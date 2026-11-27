"""Скачивание и проверка весов по models/manifest.json.

  python models/fetch_models.py                 # все модели
  python models/fetch_models.py --only rtmpose-t mediapipe-lite
  python models/fetch_models.py --skip-yolo     # без экспорта YOLO (нужен ultralytics + torch)

Веса -> models/cache/<id>/<файл>. Итог (SHA-256, размер, дата, версия инструментов) ->
models/manifest.lock.json. Скачанные файлы не исполняются; zip распаковывается только
по имени нужного члена архива.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
MANIFEST = ROOT / "manifest.json"
LOCK = ROOT / "manifest.lock.json"
# TF Hub (-> Kaggle) отдаёт 404 клиентам без браузерного User-Agent
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"


def http_get(url: str, retries: int = 4) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 — сеть: повторяем с паузой
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"не удалось скачать {url}: {last}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(name: str) -> str:
    name = Path(name).name
    if not name or name in {".", ".."}:
        raise ValueError(f"недопустимое имя файла: {name!r}")
    return name


def fetch_plain(item: dict, out_dir: Path) -> list[Path]:
    dst = out_dir / safe_name(item["name"])
    if dst.exists() and dst.stat().st_size > 0:
        return [dst]
    data = http_get(item["url"])
    if "zip_member" in item:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            member = next((n for n in z.namelist() if n.endswith("/" + item["zip_member"]) or n == item["zip_member"]), None)
            if member is None:
                raise RuntimeError(f"в архиве {item['url']} нет {item['zip_member']}: {z.namelist()}")
            data = z.read(member)
    dst.write_bytes(data)
    return [dst]


def fetch_tfjs(model: dict, out_dir: Path) -> list[Path]:
    """model.json + шарды весов по путям из weightsManifest."""
    base = model["tfjs_graph"].rstrip("/")
    mj = out_dir / "model.json"
    if not mj.exists():
        mj.write_bytes(http_get(base + "/model.json?tfjs-format=file"))
    doc = json.loads(mj.read_text(encoding="utf-8"))
    files = [mj]
    for group in doc.get("weightsManifest", []):
        for p in group["paths"]:
            dst = out_dir / safe_name(p)
            if not dst.exists():
                dst.write_bytes(http_get(base + "/" + urllib.parse.quote(p) + "?tfjs-format=file"))
            files.append(dst)
    return files


def export_yolo(model: dict, out_dir: Path) -> tuple[list[Path], dict]:
    pt = fetch_plain(model["files"][0], out_dir)[0]
    exp = model["export"]
    dst = out_dir / exp["out"]
    import ultralytics  # noqa: PLC0415 — нужен только для экспорта
    from ultralytics import YOLO  # noqa: PLC0415

    if not dst.exists():
        produced = YOLO(str(pt)).export(format=exp["format"], imgsz=exp["imgsz"], opset=exp["opset"],
                                        simplify=exp["simplify"], dynamic=exp["dynamic"], nms=exp["nms"])
        shutil.move(str(produced), dst)
    return [pt, dst], {"ultralytics": ultralytics.__version__}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="id моделей")
    ap.add_argument("--role", choices=["browser", "detector", "reference", "all"], default="all")
    ap.add_argument("--skip-yolo", action="store_true")
    a = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {"models": {}}
    failed = []
    for m in manifest["models"]:
        if a.only and m["id"] not in a.only:
            continue
        if a.role != "all" and m["role"] != a.role:
            continue
        if a.skip_yolo and "export" in m:
            continue
        out_dir = CACHE / m["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"· {m['id']} ...", flush=True)
        try:
            tools = {}
            if "export" in m:
                files, tools = export_yolo(m, out_dir)
            elif "tfjs_graph" in m:
                files = fetch_tfjs(m, out_dir)
            else:
                files = [p for item in m["files"] for p in fetch_plain(item, out_dir)]
            entry = {
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "license": m["license"],
                "tools": tools,
                "files": [{"path": str(p.relative_to(ROOT.parent)).replace("\\", "/"),
                           "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files],
            }
            # размер того, что реально грузит браузер (без .pt)
            entry["serve_bytes"] = sum(f["bytes"] for f in entry["files"] if not f["path"].endswith(".pt"))
            lock["models"][m["id"]] = entry
            print(f"  ok: {entry['serve_bytes'] / 1e6:.1f} МБ")
        except Exception as e:  # noqa: BLE001
            failed.append((m["id"], str(e)))
            print(f"  ОШИБКА: {e}", file=sys.stderr)
    LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    if failed:
        print("\nне скачаны:", *[f"{i}: {e}" for i, e in failed], sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()

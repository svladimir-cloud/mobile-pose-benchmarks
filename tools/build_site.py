"""Статическая сборка стенда для GitHub Pages (или любого статического хостинга).

  python tools/build_site.py [--out _site] [--media bench/media]

Нужно заранее: `npm ci && npm run vendor`, `python models/fetch_models.py --skip-reference` (веса в models/cache/).
Результат: _site/ = bench/web (страница, JS, vendor, service worker для COOP/COEP) + models/<id>/ (браузерные модели
и детектор, без .pt) + media/ (видео для режима «видео») + config.json (то, что сервер отдаёт как /api/config).
Результаты на статическом хостинге не отправляются на сервер: участник нажимает «Поделиться»/«Скачать».
Если задан --results-endpoint, страница дополнительно отправляет каждый результат POST-запросом (text/plain) туда.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import server  # noqa: E402  — те же функции, что и у /api/config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_site")
    ap.add_argument("--media", default="bench/media", help="папка с видео (mp4/webm) и наборами кадров")
    ap.add_argument("--results-endpoint", default="")
    a = ap.parse_args()
    out = ROOT / a.out
    if out.exists():
        shutil.rmtree(out)
    web = ROOT / "bench" / "web"
    if not (web / "vendor").exists():
        sys.exit("нет bench/web/vendor — выполните npm ci && npm run vendor")
    shutil.copytree(web, out)

    models = [m for m in server.available_models() if m["ready"]]
    for m in models:
        dst = out / "models" / m["id"]
        dst.mkdir(parents=True, exist_ok=True)
        for f in m["files"]:
            shutil.copy2(ROOT / "models" / "cache" / m["id"] / f, dst / f)

    media_src = ROOT / a.media
    media = server.list_media(media_src) if media_src.exists() else []
    if media:
        shutil.copytree(media_src, out / "media", ignore=shutil.ignore_patterns("*.avi", "*.mkv", ".gitkeep"))
        media = [m for m in media if (out / m["url"]).exists()]

    cfg = {"models": models, "media": media, "coco": {"count": 0, "items": []},
           "results_endpoint": a.results_endpoint or None, "static": True}
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    (out / ".nojekyll").write_text("")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    big = [p for p in out.rglob("*") if p.is_file() and p.stat().st_size > 95e6]
    print(f"{out}: {len(models)} моделей, {len(media)} медиа, {total / 1e6:.0f} МБ")
    if big:
        sys.exit("файлы > 95 МБ (лимит GitHub Pages 100 МБ): " + ", ".join(str(p) for p in big))


if __name__ == "__main__":
    main()

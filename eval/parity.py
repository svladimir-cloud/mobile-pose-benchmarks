"""Сверка браузерной реализации с эталонной Python-реализацией на одних и тех же изображениях (R1.5).

  python eval/parity.py --browser results/raw/<браузер>_images.json --python results/raw/<python>_images.json
  python eval/parity.py --auto [--out data/parity_coco.csv]   # все пары модель: браузер (desktop-headless) × python-reference

Метрики по изображениям, где обе реализации нашли человека: средняя и p95 разница координат точки
(px и в долях диаметра бокса GT не нужен — нормировка на диагональ бокса точек эталона), доля точек
с разницей ≤ 2 px; доля изображений, где человек найден только одной из реализаций.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np


def load(path: str) -> tuple[dict, dict]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d, {p["image_id"]: p["kps"] for p in d.get("predictions", [])}


def compare(b_path: str, p_path: str) -> dict:
    bd, b = load(b_path)
    pd_, p = load(p_path)
    common = sorted(set(b) & set(p))
    both, only = [], 0
    for i in common:
        if (b[i] is None) != (p[i] is None):
            only += 1
        elif b[i] is not None:
            both.append(i)
    diffs, norm = [], []
    for i in both:
        kb, kp = np.asarray(b[i], float), np.asarray(p[i], float)
        d = np.hypot(*(kb[:, :2] - kp[:, :2]).T)
        diag = float(np.hypot(*(kp[:, :2].max(0) - kp[:, :2].min(0)))) or 1.0
        diffs.append(d)
        norm.append(d / diag)
    d = np.concatenate(diffs) if diffs else np.array([])
    n = np.concatenate(norm) if norm else np.array([])
    return {
        "model": bd["model"]["id"], "browser_backend": bd["model"].get("backend_actual"),
        "browser_runtime": bd["model"].get("runtime"), "browser_runtime_version": bd["model"].get("runtime_version"),
        "n_images": len(common), "n_both_found": len(both),
        "found_by_one_only_pct": round(100 * only / len(common), 2) if common else None,
        "mean_diff_px": round(float(d.mean()), 3) if d.size else None,
        "p95_diff_px": round(float(np.percentile(d, 95)), 3) if d.size else None,
        "within_2px_pct": round(100 * float((d <= 2).mean()), 2) if d.size else None,
        "mean_diff_norm": round(float(n.mean()), 5) if n.size else None,
        "browser_file": Path(b_path).name, "python_file": Path(p_path).name,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--browser")
    ap.add_argument("--python")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    pairs = []
    if a.auto:
        def latest(pattern):
            fs = sorted(glob.glob(f"{a.raw}/{pattern}"))
            return fs[-1] if fs else None
        models = sorted({json.loads(Path(f).read_text(encoding="utf-8"))["model"]["id"]
                         for f in glob.glob(f"{a.raw}/*_python_*_images.json")})
        for m in models:
            p = latest(f"*_python_{m}_cpu_images.json")
            b = [f for f in sorted(glob.glob(f"{a.raw}/*_{m}_*_images.json")) if "_python_" not in f]
            if p and b:
                pairs.append((b[-1], p))
    else:
        pairs = [(a.browser, a.python)]
    rows = [compare(b, p) for b, p in pairs]
    if not rows:
        sys.exit("нет пар браузер/python")
    out = open(a.out, "w", newline="", encoding="utf-8") if a.out else sys.stdout
    w = csv.DictWriter(out, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


if __name__ == "__main__":
    main()

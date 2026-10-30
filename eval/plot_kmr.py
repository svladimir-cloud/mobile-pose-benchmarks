"""Кривые KMR(τ) для выбранных прогонов (≤ 8 серий) из data/missing_*.csv.

  python3 -I eval/plot_kmr.py data/missing_stage1_desktop-headless.csv --backend wasm \
      --models rtmo-s,rtmpose-t,yolov8n-pose,movenet-thunder,blazepose-full --out docs/img/kmr_curve_stage1.png
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5cd6", "#8a6d3b"]

ap = argparse.ArgumentParser()
ap.add_argument("csv"); ap.add_argument("--backend", default="wasm"); ap.add_argument("--models", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--title", default="")
a = ap.parse_args()
models = a.models.split(",")
assert len(models) <= len(COLORS), "не больше 8 серий"
rows = [r for r in csv.DictReader(open(a.csv)) if r["keypoint"] == "all" and r["backend"] == a.backend]
fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=120)
for m, c in zip(models, COLORS):
    pts = sorted((float(r["tau"]), float(r["frames_missing"]) / float(r["frames_total"]) * 100) for r in rows if r["model"] == m)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, lw=2, marker="o", ms=4, label=m)
ax.set_xlabel("порог τ (score точки)"); ax.set_ylabel("KMR(τ), % (среднее по 17 точкам)")
ax.set_xlim(0.05, 0.95); ax.set_ylim(0, 100)
ax.grid(axis="y", color="#e5e5e5", lw=0.8); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title(a.title or f"COCO-SinglePerson (300), {a.backend}, env=desktop-headless", fontsize=9)
fig.tight_layout(); fig.savefig(a.out)

"""Оценка прогонов этапа 0: метрики → data/*.csv, рисунки → docs/img/.

  python3 -I eval/evaluate.py --gt datasets/coco/single/gt_sub300_seed20261030.json \
      --images datasets/coco/val2017 --tag prototype results/raw/<run>.json [...]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COMMON = ["run_id", "date", "env", "model", "variant", "runtime", "runtime_version", "backend", "input_size", "raw_file"]
ACC = COMMON + ["dataset", "n_images", "AP", "AP50", "AP75", "AR", "PCK@0.2", "err_median_norm", "err_p95_norm",
                "score_source", "ci_low", "ci_high",
                # Дополнительно к схеме data/README.md:
                "PCK@0.2_found", "n_pck_images", "kmr_auc", "FMR"]
MISS = COMMON + ["dataset", "video_id", "exercise", "position", "view", "light", "keypoint", "tau", "frames_total",
                 "frames_no_person", "frames_missing", "out_of_frame", "ckmr_k", "ckmr", "gap_mean_frames",
                 "gap_max_frames", "lr_swaps"]
PERF = COMMON + ["device", "soc", "os", "os_version", "browser", "browser_version", "wasm_simd", "wasm_threads",
                 "webgl2", "webgpu", "n_warmup", "n_measure", "latency_p50_ms", "latency_p95_ms", "e2e_fps_median",
                 "dropped_frames_pct", "cold_load_ms", "cold_init_ms", "first_infer_ms", "model_size_mb",
                 "download_mb", "peak_mem_mb", "fps_min1", "fps_min10", "fps_ratio_10_1", "battery_pct", "temp_c",
                 "status", "gpu_renderer"]


def r4(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else (round(float(x), 4) if not isinstance(x, (int, str, bool)) else x)


def bootstrap_ap(gt_json, results, ids, n=2000, seed=0):
    """Бутстреп AP по изображениям: выборка id с возвращением, дубликаты получают новые id."""
    import contextlib, io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    rng = np.random.default_rng(seed)
    imgs = {i["id"]: i for i in gt_json["images"]}
    anns = {}
    for a in gt_json["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    dts = {}
    for d in results:
        dts.setdefault(d["image_id"], []).append(d)
    aps = []
    for _ in range(n):
        pick = rng.choice(ids, size=len(ids), replace=True)
        g = {"images": [], "annotations": [], "categories": gt_json["categories"]}
        dl = []
        aid = 1
        for new_id, old in enumerate(pick, start=1):
            g["images"].append({**imgs[old], "id": new_id})
            for a in anns.get(old, []):
                g["annotations"].append({**a, "id": aid, "image_id": new_id}); aid += 1
            for d in dts.get(old, []):
                dl.append({**d, "image_id": new_id})
        with contextlib.redirect_stdout(io.StringIO()):
            cg = COCO(); cg.dataset = g; cg.createIndex()
            if not dl:
                aps.append(0.0); continue
            cd = cg.loadRes(dl)
            ev = COCOeval(cg, cd, "keypoints"); ev.evaluate(); ev.accumulate(); ev.summarize()
        aps.append(ev.stats[0])
    return float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))


def common_cols(r, raw_file):
    return {"run_id": r["run_id"], "date": r["date"][:10], "env": r["env"], "model": r["model"],
            "variant": r["variant"], "runtime": r["runtime"], "runtime_version": r["runtime_version"],
            "backend": r["backend"], "input_size": r["input_size"], "raw_file": raw_file}


def draw_worst(r, rows, gt_json, images_dir, out_dir, k=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    preds = {p["image_id"]: p for p in r["predictions"]}
    anns = {a["image_id"]: a for a in gt_json["annotations"]}
    det = sorted([x for x in rows if x["detected"]], key=lambda x: x["oks"])[:k]
    sk = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
    out = []
    for x in det:
        iid = x["image_id"]
        img = Image.open(Path(images_dir) / f"{iid:012d}.jpg").convert("RGB")
        g = np.asarray(anns[iid]["keypoints"], float).reshape(17, 3)
        p = np.asarray(preds[iid]["keypoints17"], float)
        fig, ax = plt.subplots(figsize=(5, 5 * img.height / img.width), dpi=100)
        ax.imshow(img); ax.axis("off")
        for a, b in sk:
            if g[a, 2] > 0 and g[b, 2] > 0:
                ax.plot(g[[a, b], 0], g[[a, b], 1], color="#1baf7a", lw=2)
            ax.plot(p[[a, b], 0], p[[a, b], 1], color="#eb6834", lw=2, ls="--")
        vis = g[:, 2] > 0
        ax.scatter(g[vis, 0], g[vis, 1], s=14, c="#1baf7a", edgecolors="white", linewidths=0.8, zorder=3)
        ax.scatter(p[:, 0], p[:, 1], s=14, c="#eb6834", edgecolors="white", linewidths=0.8, zorder=3)
        ax.set_title(f"{iid}: OKS={x['oks']:.2f} (зелёный — GT, оранжевый — модель)", fontsize=8)
        fig.tight_layout()
        name = f"worst_{r['backend']}_{iid}.jpg"
        fig.savefig(out_dir / name); plt.close(fig)
        out.append((iid, x["oks"], x["oks_swapped"], x["num_gt_kp"], name))
    return out


def plot_kmr(curves, out_file):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#2a78d6", "#eb6834"]
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=120)
    for (label, kmr, fmr), c in zip(curves, colors):
        t = M.TAUS; y = [kmr[x] * 100 for x in t]
        ax.plot(t, y, color=c, lw=2, marker="o", ms=4, label=label)
    ax.axhline(curves[0][2] * 100, color="#999", lw=1, ls=":")
    ax.annotate(f"FMR {curves[0][2]*100:.1f} % ({curves[0][0]})", (0.1, curves[0][2] * 100), xytext=(0, -11),
                textcoords="offset points", fontsize=7, color="#666")
    ax.set_xlabel("порог τ (visibility)"); ax.set_ylabel("KMR(τ), % (среднее по 17 точкам)")
    ax.set_xlim(0.05, 0.95); ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#e5e5e5", lw=0.8); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("BlazePose Lite, COCO-SinglePerson (300), env=desktop-headless", fontsize=9)
    fig.tight_layout(); fig.savefig(out_file); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--tag", default="prototype")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    gt_json = json.loads(Path(a.gt).read_text())
    img_dir = ROOT / "docs/img"; img_dir.mkdir(parents=True, exist_ok=True)
    acc_rows, miss_rows, perf_rows, curves, summary = [], [], [], [], {}
    for f in a.runs:
        r = json.loads(Path(f).read_text())
        raw_file = str(Path(f).resolve().relative_to(ROOT))
        cc = common_cols(r, raw_file)
        P = r["predictions"]; ids = [p["image_id"] for p in P]
        res = M.to_coco_results(P)
        s, _ = M.oks_ap(a.gt, res, ids)
        rows = M.per_image(P, gt_json)
        pck_all, pck_found, _, no_l = M.pck(rows)
        emed, ep95 = M.err_stats(rows)
        mi = M.missing(P)
        kmr_mean = {t: float(np.mean(mi["kmr"][t])) for t in M.TAUS}
        auc = M.kmr_auc(kmr_mean)
        ci = M_ci = bootstrap_ap(gt_json, res, ids, n=a.bootstrap) if a.bootstrap else (None, None)
        swaps = sum(1 for x in rows if x["detected"] and x["oks_swapped"] > x["oks"])
        acc_rows.append({**cc, "dataset": "coco-single", "n_images": len(P), **s, "PCK@0.2": pck_all,
                         "err_median_norm": emed, "err_p95_norm": ep95, "score_source": "visibility",
                         "ci_low": ci[0], "ci_high": ci[1], "PCK@0.2_found": pck_found,
                         "n_pck_images": len(P) - no_l, "kmr_auc": auc, "FMR": mi["FMR"]})
        for t in M.TAUS:
            for j in range(17):
                miss_rows.append({**cc, "dataset": "coco-single", "keypoint": M.COCO_KP[j], "tau": t,
                                  "frames_total": mi["frames_total"], "frames_no_person": mi["frames_no_person"],
                                  "frames_missing": mi["frames_missing"][t][j], "out_of_frame": mi["out_of_frame"][j]})
            miss_rows.append({**cc, "dataset": "coco-single", "keypoint": "all", "tau": t,
                              "frames_total": mi["frames_total"], "frames_no_person": mi["frames_no_person"],
                              "frames_missing": round(float(np.mean(mi["frames_missing"][t])), 2),
                              "out_of_frame": int(sum(mi["out_of_frame"])), "lr_swaps": swaps})
        e = r["browser_env"]
        p50, p95 = M.latency_stats(r["latency_ms"])
        perf_rows.append({**cc, "device": r["device"], "os": e.get("ua_data", {}).get("platform", ""),
                          "browser": "HeadlessChrome", "browser_version": r["harness"]["browser_version"],
                          "wasm_simd": e["wasm_simd"], "wasm_threads": e["wasm_threads"], "webgl2": e["webgl2"],
                          "webgpu": e["webgpu"], "n_warmup": r["n_warmup"], "n_measure": r["n_measure"],
                          "latency_p50_ms": p50, "latency_p95_ms": p95, **{k: r["cold"][k] for k in r["cold"]},
                          "model_size_mb": r["model_bytes"] / 1e6, "status": r["status"],
                          "gpu_renderer": e.get("webgl_renderer", "") if r["backend"] == "webgl" else ""})
        label = {"wasm": "CPU (WASM)", "webgl": "GPU (WebGL, SwiftShader)"}.get(r["backend"], r["backend"])
        curves.append((label, kmr_mean, mi["FMR"]))
        worst = draw_worst(r, rows, gt_json, a.images, img_dir)
        missed = [x["image_id"] for x in rows if not x["detected"]]
        summary[r["run_id"]] = {"acc": acc_rows[-1], "kmr_mean": kmr_mean, "kmr_by_kp_0.5": dict(zip(M.COCO_KP, mi["kmr"][0.5])),
                                "perf": perf_rows[-1], "worst": worst, "swaps": swaps, "n_detected": len(P) - len(missed),
                                "lat_min_max": [min(r["latency_ms"]), max(r["latency_ms"])]}
    plot_kmr(curves, img_dir / f"kmr_curve_{a.tag}.png")
    for name, cols, rows in (("accuracy", ACC, acc_rows), ("missing", MISS, miss_rows), ("perf", PERF, perf_rows)):
        with open(ROOT / f"data/{name}_{a.tag}_desktop-headless.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: r4(row.get(k)) for k in cols})
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()

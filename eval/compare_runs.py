"""Сверка двух прогонов на одних изображениях (R1.5): совпадение факта детекции и расхождение координат.

  python3 -I eval/compare_runs.py <прогон A .json> <прогон B .json>
Расхождение — по 17 точкам COCO, в пикселях и в долях диагонали торса по GT нет (эталона здесь нет): только px.
"""
import json
import sys

import numpy as np


def main(a_file, b_file):
    a = {p["image_id"]: p for p in json.load(open(a_file))["predictions"]}
    b = {p["image_id"]: p for p in json.load(open(b_file))["predictions"]}
    common = sorted(set(a) & set(b))
    both = [i for i in common if a[i]["keypoints17"] and b[i]["keypoints17"]]
    only_a = sum(1 for i in common if a[i]["keypoints17"] and not b[i]["keypoints17"])
    only_b = sum(1 for i in common if b[i]["keypoints17"] and not a[i]["keypoints17"])
    d = np.array([np.hypot(*(np.array(a[i]["keypoints17"])[:, :2] - np.array(b[i]["keypoints17"])[:, :2]).T) for i in both])
    ds = np.array([abs(np.array(a[i]["keypoints17"])[:, 2] - np.array(b[i]["keypoints17"])[:, 2]) for i in both])
    out = {"n_common": len(common), "both_detected": len(both), "only_a": only_a, "only_b": only_b,
           "px_mean": float(d.mean()) if len(both) else None, "px_median": float(np.median(d)) if len(both) else None,
           "px_p95": float(np.percentile(d, 95)) if len(both) else None,
           "imgs_max_px_gt_10": int((d.max(1) > 10).sum()) if len(both) else None,
           "score_abs_diff_mean": float(ds.mean()) if len(both) else None}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

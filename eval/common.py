"""Общие чистые функции для скриптов eval/ (без ввода-вывода, кроме load_json)."""
import json
import math
import os
import sys

# Скрипты запускаются и с `python -I` (он не добавляет каталог скрипта в sys.path).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

KP_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
            "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
            "l_knee", "r_knee", "l_ankle", "r_ankle"]
KP = {n: i for i, n in enumerate(KP_NAMES)}
# Пары «левый/правый» для перестановок сторон
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
TAUS = [round(0.1 * i, 1) for i in range(1, 10)]

META_COLS = ["run_id", "date", "env", "model", "variant", "runtime",
             "runtime_version", "backend", "input_size", "raw_file"]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def meta_from_result(res, raw_file=""):
    """Общие столбцы CSV из result-json (docs/result_format.md)."""
    m = res.get("model") or {}
    return {
        "run_id": res.get("run_id", ""),
        "date": str(res.get("created_at", ""))[:10],
        "env": res.get("env", ""),
        "model": m.get("id", m.get("family", "")),
        "variant": m.get("variant", ""),
        "runtime": m.get("runtime", ""),
        "runtime_version": m.get("runtime_version", ""),
        "backend": m.get("backend_actual", m.get("backend_requested", "")),
        "input_size": m.get("input_size", ""),
        "raw_file": raw_file,
    }


def kps_array(kps):
    """kps из json -> ndarray (17,3) float или None."""
    if kps is None:
        return None
    a = np.asarray(kps, dtype=float)
    if a.shape != (17, 3):
        raise ValueError(f"ожидалось 17x3 точек, получено {a.shape}")
    return a


def torso_diameter(gt):
    """Диаметр торса по GT (17,3): левое плечо — правое бедро; если нет —
    правое плечо — левое бедро; если обе пары недоступны — None.
    Точка доступна при v>0."""
    for a, b in ((KP["l_shoulder"], KP["r_hip"]), (KP["r_shoulder"], KP["l_hip"])):
        if gt[a, 2] > 0 and gt[b, 2] > 0:
            d = float(np.hypot(*(gt[a, :2] - gt[b, :2])))
            if d > 0:
                return d
    return None


def lost_mask(pred, tau, width, height):
    """Булев (17,) вектор «точка потеряна»: нет человека / score<tau / вне кадра."""
    if pred is None:
        return np.ones(17, dtype=bool)
    x, y, s = pred[:, 0], pred[:, 1], pred[:, 2]
    outside = (x < 0) | (x >= width) | (y < 0) | (y >= height) | ~np.isfinite(x) | ~np.isfinite(y)
    return (s < tau) | outside


def percentile(a, q):
    """Линейный перцентиль (как numpy), устойчивый к +inf (потерянные точки)."""
    a = np.sort(np.asarray(a, dtype=float))
    if a.size == 0:
        return float("nan")
    pos = (a.size - 1) * q / 100.0
    lo, hi = int(np.floor(pos)), int(np.ceil(pos))
    if a[lo] == a[hi]:
        return float(a[lo])
    if np.isinf(a[hi]) or np.isinf(a[lo]):
        return float(a[hi])
    return float(a[lo] + (a[hi] - a[lo]) * (pos - lo))


def fmt(v, nd=4):
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        if math.isinf(v):
            return "inf"
        return f"{v:.{nd}f}"
    return str(v)


def write_csv(path, cols, rows):
    """path=None или '-' -> stdout. Столбцы, которых нет в строке, пустые."""
    import csv
    f = sys.stdout if path in (None, "-") else open(path, "w", newline="", encoding="utf-8")
    try:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([fmt(r.get(c)) for c in cols])
    finally:
        if f is not sys.stdout:
            f.close()


ACCURACY_COLS = META_COLS + ["dataset", "n_images", "AP", "AP50", "AP75", "AR", "PCK@0.2",
                             "err_median_norm", "err_p95_norm", "score_source", "ci_low", "ci_high",
                             # дополнительные столбцы (сверх схемы data/README.md)
                             "err_median_norm_lostinf", "err_p95_norm_lostinf", "fmr",
                             "n_pck_images", "kmr_auc", "tau_star_10"]
MISSING_COLS = META_COLS + ["dataset", "video_id", "exercise", "position", "view", "light",
                            "keypoint", "tau", "frames_total", "frames_no_person",
                            "frames_missing", "out_of_frame", "ckmr_k", "ckmr",
                            "gap_mean_frames", "gap_max_frames", "lr_swaps",
                            # дополнительные
                            "kmr", "kmr_auc"]


def nanmean(a):
    """Среднее без NaN; nan для пустого/целиком-NaN входа (без предупреждений)."""
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")

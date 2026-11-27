#!/usr/bin/env python3
"""Сравнение покадровых точек прогона (mode=frames|video) с эталоном (mode=frames).

(а) углы суставов 2D: MAE и p95 по сопоставленным кадрам;
(б) подсчёт повторов (сглаживание + гистерезис по размаху эталона);
(в) амплитуда повторов — ошибка по сопоставленным повторам;
(г) CKMR(tau), FMR, длины разрывов;
(д) перестановки лево/право.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import common as C  # noqa: E402

K = C.KP
# сустав -> (точка A, вершина B, точка C); для каждой стороны префикс l_/r_
JOINT_DEF = {"knee": ("hip", "knee", "ankle"), "hip": ("shoulder", "hip", "knee"),
             "elbow": ("shoulder", "elbow", "wrist"), "shoulder": ("hip", "shoulder", "elbow")}
EXERCISE_JOINT = {"squat": "knee", "knee_flexion": "knee", "arm_raise": "shoulder",
                  "elbow_flexion": "elbow"}


# ---------------------------------------------------------------- чистые функции
def frames_to_arrays(frames):
    """frames (json) -> (t (N,), kps (N,17,3) с NaN для null, found (N,) bool); сортировка по времени."""
    fr = sorted(frames, key=lambda f: f["media_time_s"])
    t = np.array([f["media_time_s"] for f in fr], dtype=float)
    kps = np.full((len(fr), 17, 3), np.nan)
    found = np.zeros(len(fr), dtype=bool)
    for i, f in enumerate(fr):
        if f.get("kps") is not None:
            kps[i] = C.kps_array(f["kps"])
            found[i] = True
    return t, kps, found


def frame_period(t):
    return float(np.median(np.diff(t))) if len(t) > 1 else float("nan")


def match_frames(t_run, t_ref, tol=None):
    """Для каждого кадра прогона — индекс ближайшего кадра эталона и маска допуска
    (допуск по умолчанию — половина периода кадра эталона)."""
    if tol is None:
        tol = 0.5 * frame_period(t_ref)
    idx = np.searchsorted(t_ref, t_run)
    idx = np.clip(idx, 1, len(t_ref) - 1)
    left, right = t_ref[idx - 1], t_ref[idx]
    idx = np.where(np.abs(t_run - left) <= np.abs(right - t_run), idx - 1, idx)
    ok = np.abs(t_ref[idx] - t_run) <= tol + 1e-9
    return idx, ok


def joint_triple(joint, side):
    a, b, c = JOINT_DEF[joint]
    return K[f"{side}_{a}"], K[f"{side}_{b}"], K[f"{side}_{c}"]


def angle_series(kps, triple, min_score=0.0):
    """Угол в вершине B между лучами BA и BC (градусы, 2D) по кадрам; NaN, если точки
    нет или score < min_score."""
    a, b, c = triple
    A, B, Cc = kps[:, a, :2], kps[:, b, :2], kps[:, c, :2]
    v1, v2 = A - B, Cc - B
    dot = (v1 * v2).sum(1)
    n = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ang = np.degrees(np.arccos(np.clip(dot / n, -1, 1)))
    ok = np.all(kps[:, [a, b, c], 2] >= min_score, axis=1)
    ang[~ok | ~np.isfinite(ang)] = np.nan
    return ang


def all_angles(kps, min_score=0.0):
    return {f"{s}_{j}": angle_series(kps, joint_triple(j, s), min_score)
            for j in JOINT_DEF for s in ("l", "r")}


def angle_errors(run_ang, ref_ang, idx, ok):
    """Модуль ошибки угла по сопоставленным кадрам, где угол есть и у прогона, и у эталона."""
    r = ref_ang[idx]
    e = np.abs(run_ang - r)
    e[~ok] = np.nan
    return e[np.isfinite(e)]


def main_angle(kps, joint, side, min_score=0.0, triple=None):
    """Основной угол упражнения: сторона l/r, both = среднее по доступным сторонам."""
    if triple is not None:
        return angle_series(kps, triple, min_score)
    sides = ("l", "r") if side == "both" else (side,)
    arr = np.vstack([angle_series(kps, joint_triple(joint, s), min_score) for s in sides])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(arr, axis=0)


def smooth(x, window):
    """Скользящее среднее по окну window (нечётное округление), NaN пропускаются."""
    x = np.asarray(x, float)
    w = max(1, int(round(window)))
    if w <= 1:
        return x.copy()
    m = np.isfinite(x)
    k = np.ones(w)
    num = np.convolve(np.where(m, x, 0.0), k, "same")
    den = np.convolve(m.astype(float), k, "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~m] = np.nan  # не «достраиваем» пропущенные кадры
    return out


def thresholds_from_ref(ref_smooth, low_frac=0.3, high_frac=0.7, pct=(2, 98)):
    """Пороги low/high в градусах как доли размаха эталона (перцентили 2–98)."""
    v = ref_smooth[np.isfinite(ref_smooth)]
    if v.size < 2:
        return None
    lo, hi = np.percentile(v, pct)
    rng = hi - lo
    return {"low": lo + low_frac * rng, "high": lo + high_frac * rng, "range": rng,
            "min": lo, "max": hi}


def count_reps(t, ang, low, high):
    """Гистерезисный подсчёт. Начальная зона — первая достигнутая (low или high);
    повтор = выход в противоположную зону и возврат в начальную.
    Возвращает список {t_start,t_end,t_ext,ext} (ext — экстремум угла за повтор:
    min, если начало в high, max, если начало в low)."""
    reps = []
    zone0 = None
    in_exc = False
    seg_t, seg_a = [], []
    t_start = None
    for ti, a in zip(t, ang):
        if not np.isfinite(a):
            continue
        z = "low" if a <= low else ("high" if a >= high else None)
        if zone0 is None:
            zone0 = z
            continue
        opp = "low" if zone0 == "high" else "high"
        if not in_exc:
            if z == opp:
                in_exc = True
                seg_t, seg_a = [ti], [a]
        else:
            seg_t.append(ti)
            seg_a.append(a)
            if z == zone0:
                seg_a_arr = np.array(seg_a)
                k = int(np.argmin(seg_a_arr) if zone0 == "high" else np.argmax(seg_a_arr))
                reps.append({"t_start": float(seg_t[0]), "t_end": float(ti),
                             "t_ext": float(seg_t[k]), "ext": float(seg_a_arr[k])})
                in_exc = False
    return reps


def match_reps(run_reps, ref_reps):
    """Жадное сопоставление повторов: повтор прогона привязывается к повтору эталона,
    в интервал которого [t_start,t_end] (с расширением на 25% длины) попадает его экстремум;
    при нескольких кандидатах — ближайший по t_ext; каждый повтор эталона — один раз."""
    used, pairs = set(), []
    for r in run_reps:
        best, bd = None, None
        for j, f in enumerate(ref_reps):
            if j in used:
                continue
            pad = 0.25 * (f["t_end"] - f["t_start"])
            if f["t_start"] - pad <= r["t_ext"] <= f["t_end"] + pad:
                d = abs(r["t_ext"] - f["t_ext"])
                if bd is None or d < bd:
                    best, bd = j, d
        if best is not None:
            used.add(best)
            pairs.append((r, ref_reps[best]))
    return pairs


def missing_any(kps, found, tau, w, h, key_idx, k=1):
    """(N,) bool: потеряно не менее k ключевых точек упражнения (нет человека / score<tau / вне кадра)."""
    out = np.zeros(len(kps), dtype=bool)
    for i in range(len(kps)):
        arr = kps[i] if found[i] else None
        lost = C.lost_mask(arr, tau, w, h)[key_idx]
        out[i] = lost.sum() >= k
    return out


def out_of_frame_frames(kps, found, w, h, key_idx):
    x, y = kps[:, key_idx, 0], kps[:, key_idx, 1]
    o = (x < 0) | (x >= w) | (y < 0) | (y >= h)
    return int(np.count_nonzero(found & np.any(o, axis=1)))


def gap_lengths(mask):
    """Длины серий True подряд."""
    out, n = [], 0
    for v in mask:
        if v:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def lr_swaps(run_kps, ref_kps, idx, ok, ratio=2.0):
    """Кадры, где при обмене левых/правых точек ошибка относительно эталона меньше
    в `ratio` раз. Сравниваются 8 симметричных пар (точки без NaN)."""
    n_eval = swaps = 0
    flags = np.zeros(len(run_kps), dtype=bool)
    for i in range(len(run_kps)):
        if not ok[i]:
            continue
        p, r = run_kps[i], ref_kps[idx[i]]
        e_n = e_s = 0.0
        cnt = 0
        for a, b in C.LR_PAIRS:
            pts = [p[a, :2], p[b, :2], r[a, :2], r[b, :2]]
            if not np.all(np.isfinite(pts)):
                continue
            e_n += np.linalg.norm(p[a, :2] - r[a, :2]) + np.linalg.norm(p[b, :2] - r[b, :2])
            e_s += np.linalg.norm(p[a, :2] - r[b, :2]) + np.linalg.norm(p[b, :2] - r[a, :2])
            cnt += 1
        if cnt == 0:
            continue
        n_eval += 1
        if e_n > 0 and e_s < e_n / ratio:
            swaps += 1
            flags[i] = True
    return swaps, n_eval, flags


def stats_abs(e):
    e = np.asarray(e, float)
    if e.size == 0:
        return {"n": 0, "mae": float("nan"), "p95": float("nan")}
    return {"n": int(e.size), "mae": float(e.mean()), "p95": float(np.percentile(e, 95))}


def exercise_keypoints(exercise, side, custom=None):
    """Индексы ключевых точек упражнения и (joint, triple) основного угла."""
    if exercise == "custom":
        tr = tuple(K[n.strip()] for n in custom.split(","))
        assert len(tr) == 3, "--custom-angle: три имени точек через запятую"
        return sorted(set(tr)), None, tr
    joint = EXERCISE_JOINT[exercise]
    sides = ("l", "r") if side == "both" else (side,)
    idx = sorted({i for s in sides for i in joint_triple(joint, s)})
    return idx, joint, None


def analyze(run, ref, exercise="squat", side="both", custom=None, low_frac=0.3, high_frac=0.7,
            smooth_s=0.15, min_score=0.3, taus=C.TAUS, ckmr_k=1, tol=None):
    """Полный анализ пары (прогон, эталон) -> dict."""
    t_r, k_r, f_r = frames_to_arrays(run["frames"])
    t_e, k_e, f_e = frames_to_arrays(ref["frames"])
    src = run.get("source") or {}
    w, h = src.get("width", 1e9), src.get("height", 1e9)
    idx, ok = match_frames(t_r, t_e, tol)
    key_idx, joint, triple = exercise_keypoints(exercise, side, custom)
    out = {"n_run_frames": len(t_r), "n_ref_frames": len(t_e), "n_matched": int(ok.sum()),
           "run_fps": (len(t_r) - 1) / (t_r[-1] - t_r[0]) if len(t_r) > 1 and t_r[-1] > t_r[0] else None}
    # (а) углы
    ar, ae = all_angles(k_r, min_score), all_angles(k_e, min_score)
    per, pooled = {}, []
    for name in ar:
        e = angle_errors(ar[name], ae[name], idx, ok)
        per[name] = stats_abs(e)
        pooled.append(e)
    out["angles"] = {"per_angle": per, "all": stats_abs(np.concatenate(pooled))}
    for j in JOINT_DEF:
        out["angles"][j] = stats_abs(np.concatenate(
            [angle_errors(ar[f"{s}_{j}"], ae[f"{s}_{j}"], idx, ok) for s in ("l", "r")]))
    # (б)(в) повторы
    m_r = main_angle(k_r, joint, side, min_score, triple)
    m_e = main_angle(k_e, joint, side, min_score, triple)
    fps_r = out["run_fps"] or 1.0
    fps_e = (len(t_e) - 1) / (t_e[-1] - t_e[0]) if len(t_e) > 1 and t_e[-1] > t_e[0] else 1.0
    s_r, s_e = smooth(m_r, smooth_s * fps_r), smooth(m_e, smooth_s * fps_e)
    th = thresholds_from_ref(s_e, low_frac, high_frac)
    if th is None or th["range"] < 10:
        out["reps"] = {"error": "размах угла эталона < 10° или нет данных", "thresholds": th}
    else:
        rr, re = count_reps(t_r, s_r, th["low"], th["high"]), count_reps(t_e, s_e, th["low"], th["high"])
        pairs = match_reps(rr, re)
        amp = [abs(a["ext"] - b["ext"]) for a, b in pairs]
        out["reps"] = {"thresholds": th, "n_run": len(rr), "n_ref": len(re),
                       "count_error": len(rr) - len(re), "abs_count_error": abs(len(rr) - len(re)),
                       "n_matched": len(pairs), "amplitude_abs_error": stats_abs(amp),
                       "run_amplitudes": [r["ext"] for r in rr], "ref_amplitudes": [r["ext"] for r in re]}
    # (г) полнота
    period_ms = 1000 * frame_period(t_r) if len(t_r) > 1 else float("nan")
    out["fmr"] = float(1 - f_r.mean()) if len(f_r) else float("nan")
    out["frame_period_ms"] = period_ms
    out["key_keypoints"] = [C.KP_NAMES[i] for i in key_idx]
    ck = {}
    for tau in taus:
        mk = missing_any(k_r, f_r, tau, w, h, key_idx, ckmr_k)
        g = gap_lengths(mk)
        per_kp = {}
        for i in key_idx:
            per_kp[C.KP_NAMES[i]] = int(np.count_nonzero(missing_any(k_r, f_r, tau, w, h, [i], 1)))
        ck[tau] = {"ckmr": float(mk.mean()) if len(mk) else float("nan"),
                   "frames_missing": int(mk.sum()),
                   "gap_count": len(g), "gap_mean_frames": float(np.mean(g)) if g else 0.0,
                   "gap_median_frames": float(np.median(g)) if g else 0.0,
                   "gap_max_frames": int(max(g)) if g else 0,
                   "gap_median_ms": float(np.median(g)) * period_ms if g else 0.0,
                   "gap_max_ms": max(g) * period_ms if g else 0.0,
                   "kmr_frames_missing": per_kp}
    out["ckmr"] = ck
    out["out_of_frame"] = out_of_frame_frames(k_r, f_r, w, h, key_idx)
    # (д) лево/право
    sw, n_eval, _ = lr_swaps(k_r, k_e, idx, ok)
    out["lr_swaps"] = {"frames": sw, "evaluated": n_eval, "share": sw / n_eval if n_eval else float("nan")}
    return out


def missing_rows(meta, res, ex, video_id, exercise, extra):
    rows = []
    n = ex["n_run_frames"]
    for tau, c in ex["ckmr"].items():
        base = dict(meta)
        base.update({"dataset": extra.get("dataset", "own-frames"), "video_id": video_id,
                     "exercise": exercise, "position": extra.get("position", ""),
                     "view": extra.get("view", ""), "light": extra.get("light", ""),
                     "tau": tau, "frames_total": n,
                     "frames_no_person": int(round(ex["fmr"] * n)),
                     "lr_swaps": ex["lr_swaps"]["frames"]})
        r = dict(base)
        r.update({"keypoint": "all", "frames_missing": c["frames_missing"],
                  "out_of_frame": ex["out_of_frame"], "ckmr_k": extra.get("ckmr_k", 1),
                  "ckmr": c["ckmr"], "gap_mean_frames": c["gap_mean_frames"],
                  "gap_max_frames": c["gap_max_frames"],
                  "kmr": c["frames_missing"] / n if n else float("nan")})
        rows.append(r)
        for name, miss in c["kmr_frames_missing"].items():
            r = dict(base)
            r.update({"keypoint": name, "frames_missing": miss, "kmr": miss / n if n else float("nan")})
            rows.append(r)
    return rows


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", help="result-json прогона (frames/video)")
    ap.add_argument("ref", help="result-json эталона (frames)")
    ap.add_argument("--exercise", default="squat",
                    choices=["squat", "knee_flexion", "arm_raise", "elbow_flexion", "custom"])
    ap.add_argument("--side", default="both", choices=["both", "l", "r"])
    ap.add_argument("--custom-angle", default=None, help="для custom: три точки A,B,C (имена COCO)")
    ap.add_argument("--low-frac", type=float, default=0.3)
    ap.add_argument("--high-frac", type=float, default=0.7)
    ap.add_argument("--smooth-s", type=float, default=0.15, help="окно сглаживания, с")
    ap.add_argument("--min-score", type=float, default=0.3, help="минимальный score точки для углов")
    ap.add_argument("--ckmr-k", type=int, default=1, help="CKMR: потеряно не менее k точек")
    ap.add_argument("--position", default="")
    ap.add_argument("--view", default="")
    ap.add_argument("--light", default="")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--out-missing", default=None, help="CSV missing_*.csv (по умолчанию stdout)")
    a = ap.parse_args(argv)
    run, ref = C.load_json(a.run), C.load_json(a.ref)
    for r, n in ((run, a.run), (ref, a.ref)):
        if "frames" not in r:
            sys.exit(f"{n}: нет поля frames")
    ex = analyze(run, ref, a.exercise, a.side, a.custom_angle, a.low_frac, a.high_frac,
                 a.smooth_s, a.min_score, ckmr_k=a.ckmr_k)
    meta = C.meta_from_result(run, os.path.basename(a.run))
    # идентификация прогона для eval/aggregate.py
    dev = run.get("device") or {}
    ex["run"] = {**meta, "device": dev.get("user_label") or dev.get("ua_model") or "",
                 "exercise": a.exercise}
    vid = (run.get("source") or {}).get("name", "")
    rows = missing_rows(meta, run, ex, vid, a.exercise,
                        {"position": a.position, "view": a.view, "light": a.light, "ckmr_k": a.ckmr_k})
    C.write_csv(a.out_missing, C.MISSING_COLS, rows)
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(ex, f, indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())

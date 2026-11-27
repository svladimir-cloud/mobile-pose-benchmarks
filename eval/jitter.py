#!/usr/bin/env python3
"""Стабильность точек: J_acc (средняя величина второй разности координат, px/кадр²)
на статичном сегменте и на движении, до и после фильтра 1€ (Casiez 2012),
с оценкой вносимой задержки (сдвиг максимума кросс-корреляции)."""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import common as C  # noqa: E402
import exercise as X  # noqa: E402

# Параметры 1€ фиксированы протоколом
MIN_CUTOFF, BETA, D_CUTOFF = 1.0, 0.007, 1.0

JITTER_COLS = C.META_COLS + ["video_id", "segment", "keypoint", "filter",
                             "jacc_px_per_frame2", "speed_px_per_frame", "filter_lag_ms",
                             # дополнительные
                             "jacc_norm", "speed_norm", "n_frames", "torso_px"]


# ---------------------------------------------------------------- чистые функции
def _alpha(cutoff, dt):
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def one_euro(t, x, min_cutoff=MIN_CUTOFF, beta=BETA, d_cutoff=D_CUTOFF):
    """Фильтр 1€ для одного скалярного ряда x(t) (t в секундах). NaN пропускаются
    (выход NaN, состояние сохраняется)."""
    out = np.full(len(x), np.nan)
    x_prev = dx_prev = t_prev = None
    for i, (ti, xi) in enumerate(zip(t, x)):
        if not np.isfinite(xi):
            continue
        if x_prev is None:
            out[i] = xi
            x_prev, dx_prev, t_prev = xi, 0.0, ti
            continue
        dt = max(ti - t_prev, 1e-6)
        dx = (xi - x_prev) / dt
        a_d = _alpha(d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * dx_prev
        cutoff = min_cutoff + beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * xi + (1 - a) * x_prev
        out[i] = x_hat
        x_prev, dx_prev, t_prev = x_hat, dx_hat, ti
    return out


def filter_keypoints(t, kps, **kw):
    """Фильтр 1€ по каждой координате каждой точки: kps (N,17,3) -> (N,17,3), score сохраняется."""
    out = kps.copy()
    for j in range(17):
        for c in range(2):
            out[:, j, c] = one_euro(t, kps[:, j, c], **kw)
    return out


def torso_length(kps):
    """Медианная длина торса (среднее двух диагоналей плечо-противоположное бедро), px."""
    d1 = np.linalg.norm(kps[:, C.KP["l_shoulder"], :2] - kps[:, C.KP["r_hip"], :2], axis=1)
    d2 = np.linalg.norm(kps[:, C.KP["r_shoulder"], :2] - kps[:, C.KP["l_hip"], :2], axis=1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = np.nanmean(np.vstack([d1, d2]), axis=0)
    d = d[np.isfinite(d)]
    return float(np.median(d)) if d.size else float("nan")


def second_diff_mag(xy):
    """Величина второй разности (px/кадр²) для ряда xy (N,2); NaN, если нужна пропущенная точка."""
    if len(xy) < 3:
        return np.array([])
    d2 = xy[2:] - 2 * xy[1:-1] + xy[:-2]
    return np.linalg.norm(d2, axis=1)


def first_diff_mag(xy):
    if len(xy) < 2:
        return np.array([])
    return np.linalg.norm(xy[1:] - xy[:-1], axis=1)


def segment_stats(kps_seg):
    """По сегменту kps_seg (n,17,3) -> {j: (jacc, speed)} (nanmean величин по кадрам)."""
    res = {}
    for j in range(17):
        a = second_diff_mag(kps_seg[:, j, :2])
        v = first_diff_mag(kps_seg[:, j, :2])
        res[j] = (C.nanmean(a), C.nanmean(v))
    return res


def joint_speed_norm(t, kps, torso):
    """Медиана по суставам скорости в длинах торса в секунду (центральная разность) по кадрам."""
    n = len(t)
    sp = np.full((n, 17), np.nan)
    if n >= 3:
        dt = (t[2:] - t[:-2])[:, None]
        sp[1:-1] = np.linalg.norm(kps[2:, :, :2] - kps[:-2, :, :2], axis=2) / dt / torso
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(sp, axis=1)


def find_static(t, speed, thr=0.1, min_dur=3.0, smooth_s=0.5):
    """Самый длинный отрезок >= min_dur с, где сглаженная скорость < thr (длин торса/с).
    Возвращает (t0, t1) или None."""
    t = np.asarray(t, float)
    if len(t) < 3:
        return None
    fps = (len(t) - 1) / (t[-1] - t[0])
    s = X.smooth(speed, max(1, round(smooth_s * fps)))
    ok = np.isfinite(s) & (s < thr)
    best, i, n = None, 0, len(t)
    while i < n:
        if ok[i]:
            j = i
            while j + 1 < n and ok[j + 1]:
                j += 1
            dur = t[j] - t[i]
            if dur >= min_dur and (best is None or dur > best[1] - best[0]):
                best = (float(t[i]), float(t[j]))
            i = j + 1
        else:
            i += 1
    return best


def estimate_lag_frames(raw, filt, max_lag=15):
    """Сдвиг (в кадрах, дробный по параболе), на который filt запаздывает относительно raw
    (максимум кросс-корреляции). Положительный = задержка. NaN интерполируются."""
    raw = _interp_nan(np.asarray(raw, float))
    filt = _interp_nan(np.asarray(filt, float))
    if raw is None or filt is None or len(raw) < 2 * max_lag + 3:
        return float("nan")
    a, b = raw - raw.mean(), filt - filt.mean()
    if np.allclose(a, 0) or np.allclose(b, 0):
        return float("nan")
    lags = np.arange(-max_lag, max_lag + 1)
    cc = []
    for L in lags:
        if L >= 0:
            cc.append(np.dot(a[:len(a) - L], b[L:]))
        else:
            cc.append(np.dot(a[-L:], b[:len(b) + L]))
    cc = np.array(cc)
    k = int(np.argmax(cc))
    off = 0.0
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            off = 0.5 * (y0 - y2) / den
    return float(lags[k] + off)


def _interp_nan(x):
    m = np.isfinite(x)
    if m.sum() < 2:
        return None
    idx = np.arange(len(x))
    return np.interp(idx, idx[m], x[m])


def analyze(t, kps, static=None, motion=None, ref_t=None, ref_kps=None,
            static_thr=0.1, min_static_s=3.0):
    """Результат: {"torso_px", "static": (t0,t1)|None, "motion": (t0,t1)|None, "rows": [...]}.
    Строки: segment, keypoint(имя|all), filter(none|one-euro), jacc, speed, lag_ms, n_frames."""
    torso = torso_length(kps)
    det_t, det_k = (ref_t, ref_kps) if ref_kps is not None else (t, kps)
    if static is None:
        sp = joint_speed_norm(det_t, det_k, torso_length(det_k))
        static = find_static(det_t, sp, static_thr, min_static_s)
    if motion is None:
        motion = (float(t[0]), float(t[-1]))
        if static is not None:
            # движение — всё, что вне статичного отрезка: берём более длинную из двух частей
            before, after = (t[0], static[0]), (static[1], t[-1])
            motion = max((before, after), key=lambda p: p[1] - p[0])
            if motion[1] - motion[0] < 1.0:
                motion = (float(t[0]), float(t[-1]))
    filt = filter_keypoints(t, kps)
    period = float(np.median(np.diff(t))) if len(t) > 1 else float("nan")
    rows = []
    for seg_name, seg in (("static", static), ("motion", motion)):
        if seg is None:
            continue
        m = (t >= seg[0]) & (t <= seg[1])
        if m.sum() < 3:
            continue
        for fname, data in (("none", kps), ("one-euro", filt)):
            st = segment_stats(data[m])
            lag_ms = {}
            for j in range(17):
                if fname == "one-euro":
                    lg = C.nanmean([estimate_lag_frames(kps[:, j, c], filt[:, j, c]) for c in (0, 1)])
                    lag_ms[j] = float(lg) * period * 1000 if np.isfinite(lg) else float("nan")
                else:
                    lag_ms[j] = 0.0
            for j in range(17):
                rows.append({"segment": seg_name, "keypoint": C.KP_NAMES[j], "filter": fname,
                             "jacc": st[j][0], "speed": st[j][1], "lag_ms": lag_ms[j],
                             "n_frames": int(m.sum())})
            ja = np.array([st[j][0] for j in range(17)])
            sp = np.array([st[j][1] for j in range(17)])
            lg = np.array([lag_ms[j] for j in range(17)])
            rows.append({"segment": seg_name, "keypoint": "all", "filter": fname,
                         "jacc": C.nanmean(ja), "speed": C.nanmean(sp),
                         "lag_ms": C.nanmean(lg), "n_frames": int(m.sum())})
    return {"torso_px": torso, "static": static, "motion": motion, "rows": rows}


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="result-json с полем frames")
    ap.add_argument("--ref", default=None, help="эталон (frames) для автопоиска статичного отрезка")
    ap.add_argument("--static", type=float, nargs=2, metavar=("T0", "T1"), default=None)
    ap.add_argument("--motion", type=float, nargs=2, metavar=("T0", "T1"), default=None)
    ap.add_argument("--static-speed", type=float, default=0.1,
                    help="порог скорости суставов для статики, длин торса/с")
    ap.add_argument("--min-static-s", type=float, default=3.0)
    ap.add_argument("--out", default=None, help="CSV jitter_*.csv (по умолчанию stdout)")
    a = ap.parse_args(argv)
    res = C.load_json(a.input)
    t, kps, _ = X.frames_to_arrays(res["frames"])
    rt = rk = None
    if a.ref:
        rt, rk, _ = X.frames_to_arrays(C.load_json(a.ref)["frames"])
    r = analyze(t, kps, tuple(a.static) if a.static else None, tuple(a.motion) if a.motion else None,
                rt, rk, a.static_speed, a.min_static_s)
    meta = C.meta_from_result(res, os.path.basename(a.input))
    vid = (res.get("source") or {}).get("name", "")
    rows = []
    for x in r["rows"]:
        row = dict(meta)
        row.update({"video_id": vid, "segment": x["segment"], "keypoint": x["keypoint"],
                    "filter": x["filter"], "jacc_px_per_frame2": x["jacc"],
                    "speed_px_per_frame": x["speed"], "filter_lag_ms": x["lag_ms"],
                    "jacc_norm": x["jacc"] / r["torso_px"], "speed_norm": x["speed"] / r["torso_px"],
                    "n_frames": x["n_frames"], "torso_px": r["torso_px"]})
        rows.append(row)
    C.write_csv(a.out, JITTER_COLS, rows)
    print(f"статичный отрезок: {r['static']}; движение: {r['motion']}; торс {r['torso_px']:.1f} px",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

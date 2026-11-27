#!/usr/bin/env python3
"""Минимальная частота кадров f_min: прореживание полнокадровых точек (эталон или прогон
mode=frames) до 5..30 кадр/с с несколькими фазовыми сдвигами; ошибка числа повторов и
амплитуды относительно полной частоты."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import common as C  # noqa: E402
import exercise as X  # noqa: E402

FREQS = [5, 8, 10, 12, 15, 20, 30]


# ---------------------------------------------------------------- чистые функции
def decimate_indices(t, f_hz, phase_frac=0.0):
    """Индексы кадров, ближайших к моментам t0 + (phase_frac + n)/f_hz.
    phase_frac в [0,1) — доля периода прореженной сетки. Кадры не повторяются;
    момент без кадра ближе чем в полпериода исходной частоты пропускается."""
    t = np.asarray(t, float)
    if len(t) < 2:
        return np.arange(len(t))
    base = float(np.median(np.diff(t)))
    grid = np.arange(t[0] + phase_frac / f_hz, t[-1] + 1e-9, 1.0 / f_hz)
    idx = np.searchsorted(t, grid)
    idx = np.clip(idx, 1, len(t) - 1)
    idx = np.where(np.abs(t[idx - 1] - grid) <= np.abs(t[idx] - grid), idx - 1, idx)
    ok = np.abs(t[idx] - grid) <= 0.5 * base + 1e-9
    return np.unique(idx[ok])


def reps_and_amps(t, kps, joint, side, triple, min_score, smooth_s, th=None,
                  low_frac=0.3, high_frac=0.7):
    """(повторы, пороги) для траектории; если th не задан — пороги из неё же."""
    ang = X.main_angle(kps, joint, side, min_score, triple)
    fps = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 1.0
    s = X.smooth(ang, smooth_s * fps)
    if th is None:
        th = X.thresholds_from_ref(s, low_frac, high_frac)
    if th is None:
        return [], None
    return X.count_reps(t, s, th["low"], th["high"]), th


def fmin_table(t, kps, joint, side, triple, freqs=FREQS, n_phases=4, min_score=0.3,
               smooth_s=0.15, amp_thr=5.0, low_frac=0.3, high_frac=0.7):
    """Возвращает (строки, f_min|None, справка). Эталон для сравнения — полная частота
    той же серии; пороги low/high берутся с полной частоты."""
    full, th = reps_and_amps(t, kps, joint, side, triple, min_score, smooth_s,
                             low_frac=low_frac, high_frac=high_frac)
    if th is None or th["range"] < 10:
        return [], None, {"error": "размах угла < 10° или нет данных", "n_reps_full": len(full)}
    base_fps = (len(t) - 1) / (t[-1] - t[0])
    rows = []
    for f in freqs:
        if f > base_fps + 1e-6:
            continue
        for p in range(n_phases):
            idx = decimate_indices(t, f, p / n_phases)
            reps, _ = reps_and_amps(t[idx], kps[idx], joint, side, triple, min_score, smooth_s, th)
            pairs = X.match_reps(reps, full)
            amp = [abs(a["ext"] - b["ext"]) for a, b in pairs]
            # неспаренные повторы эталона — потерянные; ошибка амплитуды считается по спаренным
            rows.append({"f_hz": f, "phase": p, "n_frames": len(idx), "n_reps_full": len(full),
                         "n_reps": len(reps), "count_err": len(reps) - len(full),
                         "n_matched": len(pairs),
                         "amp_mae_deg": float(np.mean(amp)) if amp else float("nan"),
                         "amp_max_deg": float(np.max(amp)) if amp else float("nan")})
    fmin = None
    summary = {}
    for f in sorted({r["f_hz"] for r in rows}):
        rs = [r for r in rows if r["f_hz"] == f]
        worst_cnt = max(abs(r["count_err"]) for r in rs)
        amps = [r["amp_mae_deg"] for r in rs]
        worst_amp = float("nan") if any(np.isnan(amps)) else max(amps)
        good = worst_cnt == 0 and np.isfinite(worst_amp) and worst_amp <= amp_thr
        summary[f] = {"worst_abs_count_err": worst_cnt, "worst_amp_mae_deg": worst_amp, "ok": bool(good)}
    # f_min — минимальная частота, начиная с которой условие выполняется и для всех выше неё
    fs = sorted(summary)
    for i, f in enumerate(fs):
        if all(summary[g]["ok"] for g in fs[i:]):
            fmin = f
            break
    return rows, fmin, {"summary": summary, "thresholds": th, "n_reps_full": len(full)}


FMIN_COLS = ["source_file", "exercise", "f_hz", "phase", "n_frames", "n_reps_full", "n_reps",
             "count_err", "n_matched", "amp_mae_deg", "amp_max_deg"]


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="result-json с полем frames (полная частота)")
    ap.add_argument("--exercise", default="squat",
                    choices=["squat", "knee_flexion", "arm_raise", "elbow_flexion", "custom"])
    ap.add_argument("--side", default="both", choices=["both", "l", "r"])
    ap.add_argument("--custom-angle", default=None)
    ap.add_argument("--freqs", type=float, nargs="+", default=FREQS)
    ap.add_argument("--phases", type=int, default=4, help="число фазовых сдвигов")
    ap.add_argument("--amp-thr", type=float, default=5.0, help="порог ошибки амплитуды, градусы")
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--smooth-s", type=float, default=0.15)
    ap.add_argument("--out", default=None, help="CSV (по умолчанию stdout)")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args(argv)
    res = C.load_json(a.input)
    t, kps, _ = X.frames_to_arrays(res["frames"])
    _, joint, triple = X.exercise_keypoints(a.exercise, a.side, a.custom_angle)
    rows, fmin, info = fmin_table(t, kps, joint, a.side, triple, a.freqs, a.phases, a.min_score,
                                  a.smooth_s, a.amp_thr)
    for r in rows:
        r["source_file"] = os.path.basename(a.input)
        r["exercise"] = a.exercise
    C.write_csv(a.out, FMIN_COLS, rows)
    if "error" in info:
        print(f"f_min: не определена ({info['error']})", file=sys.stderr)
    else:
        print(f"повторов на полной частоте: {info['n_reps_full']}", file=sys.stderr)
        for f, s in info["summary"].items():
            print(f"  {f:>4} Гц: худшая ошибка числа повторов {s['worst_abs_count_err']}, "
                  f"худшая MAE амплитуды {s['worst_amp_mae_deg']:.2f}° -> {'ок' if s['ok'] else 'нет'}",
                  file=sys.stderr)
        print(f"f_min (ошибка повторов = 0, амплитуда <= {a.amp_thr}°): "
              f"{fmin if fmin is not None else 'не достигнута на проверенных частотах'}", file=sys.stderr)
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"fmin": fmin, **info}, f, indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())

import numpy as np
import pytest

import exercise as X
import fmin as F
from synth import knee_frames, squat_series


def test_decimate_indices():
    t = np.arange(0, 10, 1 / 30)
    i = F.decimate_indices(t, 10)
    assert len(i) == 100 and np.all(np.diff(i) == 3)
    j = F.decimate_indices(t, 10, 0.34)
    assert j[0] == 1 and len(j) >= 99
    assert len(F.decimate_indices(t, 5)) == 50
    assert len(np.unique(F.decimate_indices(t, 30))) == len(t)


def table(period, **kw):
    t, a = squat_series(30, 12, period)
    _, k, _ = X.frames_to_arrays(knee_frames(a, t))
    return F.fmin_table(t, k, "knee", "both", None, **kw)


def test_slow_movement_low_fmin():
    rows, fmin, info = table(4.0)   # период 4 с
    assert fmin == 5
    assert info["n_reps_full"] == 3
    assert all(r["count_err"] == 0 for r in rows)


def test_fast_movement_needs_higher_fmin():
    rows, fmin, info = table(0.8, n_phases=4, amp_thr=5.0)
    s = info["summary"]
    assert s[5]["ok"] is False and s[5]["worst_amp_mae_deg"] > 5
    assert fmin is not None and 5 < fmin <= 15
    assert s[30]["ok"] and s[30]["worst_amp_mae_deg"] == pytest.approx(0, abs=1e-6)
    # более мягкий порог даёт не больший f_min
    _, fmin2, _ = table(0.8, amp_thr=15.0)
    assert fmin2 <= fmin


def test_too_flat_signal():
    t = np.arange(0, 5, 1 / 30)
    _, k, _ = X.frames_to_arrays(knee_frames(np.full(t.size, 150.0), t))
    rows, fmin, info = F.fmin_table(t, k, "knee", "both", None)
    assert fmin is None and "error" in info

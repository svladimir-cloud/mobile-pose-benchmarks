import numpy as np
import pytest

import common as C
import exercise as X
import jitter as J
from synth import base_pose, knee_frames, result


def static_kps(n, sigma, seed=0, fps=30):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    k = np.repeat(base_pose()[None], n, axis=0)
    k[:, :, :2] += rng.normal(0, sigma, (n, 17, 2))
    return t, k


def test_jacc_of_white_noise():
    t, k = static_kps(3000, 1.0)
    st = J.segment_stats(k)
    # |вторая разность| для 2D гауссова шума: std на ось sqrt(6)*sigma, среднее модуля = sqrt(6)*sqrt(pi/2)
    expected = np.sqrt(6) * np.sqrt(np.pi / 2)
    assert np.mean([v[0] for v in st.values()]) == pytest.approx(expected, rel=0.05)


def test_jacc_zero_for_constant_and_linear_motion():
    n = 50
    k = np.repeat(base_pose()[None], n, axis=0)
    assert J.segment_stats(k)[0][0] == 0
    k[:, :, 0] += np.arange(n)[:, None] * 2.0
    st = J.segment_stats(k)
    assert st[0][0] == pytest.approx(0, abs=1e-9) and st[0][1] == pytest.approx(2.0)


def test_one_euro_reduces_jitter_static():
    t, k = static_kps(600, 1.0)
    f = J.filter_keypoints(t, k)
    before = J.segment_stats(k)[5][0]
    after = J.segment_stats(f)[5][0]
    assert after < 0.4 * before
    assert np.all(f[:, :, 2] == k[:, :, 2])


def test_one_euro_basic_properties():
    t = np.arange(100) / 30
    assert np.allclose(J.one_euro(t, np.full(100, 7.0)), 7.0)
    x = np.full(100, 0.0); x[50:] = 100.0   # шаг
    y = J.one_euro(t, x)
    assert 0 < y[52] < 100 and y[-1] == pytest.approx(100, abs=1.0)
    x2 = x.copy(); x2[10] = np.nan
    y2 = J.one_euro(t, x2)
    assert np.isnan(y2[10]) and np.isfinite(y2[11])


def test_lag_estimation():
    t = np.arange(600) / 30
    x = 50 * np.sin(2 * np.pi * 0.5 * t)
    shifted = np.roll(x, 3)
    assert J.estimate_lag_frames(x, shifted) == pytest.approx(3, abs=0.2)
    assert J.estimate_lag_frames(x, x) == pytest.approx(0, abs=0.05)
    y = J.one_euro(t, x)
    assert J.estimate_lag_frames(x, y) > 0.3   # фильтр вносит задержку


def test_find_static_auto():
    t = np.arange(0, 20, 1 / 30)
    speed = np.where((t > 6) & (t < 15), 0.01, 0.8)   # статика 6..15 с
    seg = J.find_static(t, speed, thr=0.1, min_dur=3.0)
    assert seg is not None and seg[0] == pytest.approx(6, abs=0.7) and seg[1] == pytest.approx(15, abs=0.7)
    assert J.find_static(t, np.full(t.size, 0.8)) is None
    short = np.where((t > 6) & (t < 8), 0.01, 0.8)
    assert J.find_static(t, short, min_dur=3.0) is None


def test_analyze_static_and_motion_end_to_end():
    fps = 30
    t = np.arange(0, 16, 1 / fps)
    k = np.repeat(base_pose()[None], t.size, axis=0)
    rng = np.random.default_rng(0)
    mv = t < 6
    k[mv, :, 0] += 100 * np.sin(2 * np.pi * 0.5 * t[mv])[:, None]   # движение 0..6 с, дальше статика
    clean = k.copy()
    k[:, :, :2] += rng.normal(0, 1.0, k[:, :, :2].shape)
    r = J.analyze(t, k, ref_t=t, ref_kps=clean)
    assert r["static"] is not None and r["static"][0] >= 5.5
    rows = {(x["segment"], x["keypoint"], x["filter"]): x for x in r["rows"]}
    s_none, s_f = rows[("static", "all", "none")], rows[("static", "all", "one-euro")]
    assert s_f["jacc"] < s_none["jacc"] and s_none["speed"] < rows[("motion", "all", "none")]["speed"]
    assert s_f["lag_ms"] == s_f["lag_ms"]  # не NaN
    # явный сегмент
    r2 = J.analyze(t, k, static=(8.0, 14.0), motion=(0.0, 5.0))
    assert r2["static"] == (8.0, 14.0)

import numpy as np
import pytest

import common as C
import exercise as X
from synth import knee_frames, result, squat_series, pose_with_knee_angle


def run_ex(frames_run, frames_ref, **kw):
    return X.analyze(result(frames_run, mode="video"), result(frames_ref), **kw)


def test_angle_series_known():
    k = np.array([pose_with_knee_angle(a) for a in (90, 120, 170)])
    ang = X.angle_series(k, X.joint_triple("knee", "l"))
    np.testing.assert_allclose(ang, [90, 120, 170], atol=1e-6)
    # NaN при отсутствующей точке и при низком score
    k[0, C.KP["l_ankle"], 2] = 0.1
    assert np.isnan(X.angle_series(k, X.joint_triple("knee", "l"), 0.3)[0])
    k[1, C.KP["l_knee"], :] = np.nan
    assert np.isnan(X.angle_series(k, X.joint_triple("knee", "l"))[1])


def test_other_joint_angles_in_standing_pose():
    k = pose_with_knee_angle(170)[None]
    a = X.all_angles(k)
    assert a["l_hip"][0] == pytest.approx(180, abs=1e-6) or a["l_hip"][0] > 150
    assert a["l_elbow"][0] > 150      # рука почти прямая
    assert 0 <= a["l_shoulder"][0] <= 180


def test_match_frames_tolerance():
    t_ref = np.arange(0, 1, 1 / 30)
    t_run = np.array([0.0, 0.1, 0.1 + 0.02, 0.5])
    idx, ok = X.match_frames(t_run, t_ref)
    assert list(idx[:2]) == [0, 3] and ok.all()
    _, ok2 = X.match_frames(np.array([0.5 + 0.02, 5.0]), t_ref)
    assert not ok2[1]


def test_perfect_run_zero_error_and_reps():
    t, a = squat_series(30, 10, 2.0)  # 5 повторов, минимум 90, максимум 170
    fr = knee_frames(a, t)
    ex = run_ex(fr, fr)
    assert ex["angles"]["all"]["mae"] == pytest.approx(0, abs=1e-9)
    assert ex["reps"]["n_ref"] == 5 and ex["reps"]["n_run"] == 5 and ex["reps"]["count_error"] == 0
    assert ex["reps"]["amplitude_abs_error"]["mae"] == pytest.approx(0, abs=1e-9)
    assert min(ex["reps"]["ref_amplitudes"]) == pytest.approx(90, abs=1.0)
    assert ex["fmr"] == 0 and ex["ckmr"][0.5]["ckmr"] == 0 and ex["lr_swaps"]["frames"] == 0


def test_noisy_run_close():
    t, a = squat_series(30, 10, 2.0)
    rng = np.random.default_rng(0)
    ex = run_ex(knee_frames(a + rng.normal(0, 1.0, a.size), t), knee_frames(a, t))
    assert ex["angles"]["all"]["mae"] < 2.0
    assert ex["reps"]["count_error"] == 0 and ex["reps"]["amplitude_abs_error"]["mae"] < 3.0


def test_low_fps_run_as_is():
    t, a = squat_series(30, 10, 2.0)
    t10, a10 = squat_series(10, 10, 2.0)
    ex = run_ex(knee_frames(a10, t10), knee_frames(a, t))
    assert ex["n_run_frames"] == 100 and ex["reps"]["count_error"] == 0
    assert ex["n_matched"] == 100           # кадры прогона сопоставлены с ближайшими кадрами эталона
    assert ex["angles"]["all"]["mae"] < 1e-6


def test_amplitude_error_and_missed_reps():
    t, a = squat_series(30, 10, 2.0)
    shallow = 170 - (170 - a) * 0.875    # минимум 100 вместо 90
    ex = run_ex(knee_frames(shallow, t), knee_frames(a, t))
    assert ex["reps"]["amplitude_abs_error"]["mae"] == pytest.approx(10, abs=1.0)
    # очень мелкие приседы — повторы не засчитываются
    tiny = 170 - (170 - a) * 0.2
    ex2 = run_ex(knee_frames(tiny, t), knee_frames(a, t))
    assert ex2["reps"]["n_run"] == 0 and ex2["reps"]["count_error"] == -5


def test_count_reps_start_low_and_hysteresis():
    t = np.arange(0, 12, 0.05)
    ang = 90 + 80 * (0.5 - 0.5 * np.cos(2 * np.pi * t / 4))  # старт 90 (низко), максимум 170
    reps = X.count_reps(t, ang, low=110, high=150)
    assert len(reps) == 3 and reps[0]["ext"] == pytest.approx(170, abs=0.5)
    # дребезг около порога не создаёт ложных повторов
    rng = np.random.default_rng(0)
    noisy = ang + rng.normal(0, 3, ang.size)
    assert len(X.count_reps(t, noisy, 110, 150)) == 3


def test_missing_ckmr_gaps_fmr():
    t, a = squat_series(30, 10, 2.0)
    nulls = set(range(30, 40)) | set(range(100, 103))
    ex = run_ex(knee_frames(a, t, nulls=nulls), knee_frames(a, t))
    n = len(t)
    assert ex["fmr"] == pytest.approx(13 / n)
    c = ex["ckmr"][0.5]
    assert c["ckmr"] == pytest.approx(13 / n) and c["gap_max_frames"] == 10
    assert c["gap_count"] == 2 and c["gap_mean_frames"] == pytest.approx(6.5)
    assert c["gap_max_ms"] == pytest.approx(10 * 1000 / 30, rel=1e-6)
    # score < tau: ключевые точки с score 0.4 теряются при tau=0.5, не теряются при 0.3
    fr = knee_frames(a, t)
    for f in fr[:20]:
        f["kps"][C.KP["l_ankle"]][2] = 0.4
    ex2 = run_ex(fr, knee_frames(a, t), min_score=0.0)
    assert ex2["ckmr"][0.3]["ckmr"] == 0 and ex2["ckmr"][0.5]["ckmr"] == pytest.approx(20 / n)
    # вне кадра
    fr = knee_frames(a, t)
    for f in fr[:5]:
        f["kps"][C.KP["r_knee"]][0] = 9999
    ex3 = run_ex(fr, knee_frames(a, t))
    assert ex3["out_of_frame"] == 5 and ex3["ckmr"][0.1]["frames_missing"] == 5


def test_lr_swaps():
    t, a = squat_series(30, 10, 2.0)
    ex = run_ex(knee_frames(a, t, swaps={10, 11, 50}), knee_frames(a, t))
    assert ex["lr_swaps"]["frames"] == 3
    assert ex["lr_swaps"]["evaluated"] == len(t)


def test_arm_raise_exercise_keypoints_and_custom():
    idx, joint, tr = X.exercise_keypoints("arm_raise", "l")
    assert joint == "shoulder" and idx == sorted([C.KP["l_hip"], C.KP["l_shoulder"], C.KP["l_elbow"]])
    idx, joint, tr = X.exercise_keypoints("custom", "both", "l_hip,l_knee,l_ankle")
    assert joint is None and tr == (C.KP["l_hip"], C.KP["l_knee"], C.KP["l_ankle"])
    t, a = squat_series(30, 10, 2.0)
    fr = knee_frames(a, t)
    ex = run_ex(fr, fr, exercise="custom", custom="l_hip,l_knee,l_ankle")
    assert ex["reps"]["n_ref"] == 5


def test_missing_rows_schema():
    t, a = squat_series(30, 4, 2.0)
    fr = knee_frames(a, t)
    ex = run_ex(fr, fr)
    rows = X.missing_rows(C.meta_from_result(result(fr)), None, ex, "squat01", "squat", {"view": "0"})
    assert all(set(r) <= set(C.MISSING_COLS) for r in rows)
    assert {r["keypoint"] for r in rows} >= {"all", "l_knee", "r_ankle"}
    assert len(rows) == 9 * (1 + len(ex["key_keypoints"]))

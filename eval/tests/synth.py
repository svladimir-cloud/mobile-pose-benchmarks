"""Синтетические данные для тестов."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import common as C  # noqa: E402

K = C.KP


def base_pose(cx=300.0, cy=0.0):
    """Поза стоя (17,3), v=1: левая сторона слева (x меньше), правая — справа."""
    p = np.zeros((17, 3))
    p[:, 2] = 1.0
    def put(name, x, y):
        p[K[name], :2] = (cx + x, cy + y)
    put("nose", 0, -260); put("l_eye", -10, -270); put("r_eye", 10, -270)
    put("l_ear", -20, -262); put("r_ear", 20, -262)
    put("l_shoulder", -50, -200); put("r_shoulder", 50, -200)
    put("l_elbow", -70, -130); put("r_elbow", 70, -130)
    put("l_wrist", -75, -60); put("r_wrist", 75, -60)
    put("l_hip", -40, 0); put("r_hip", 40, 0)
    put("l_knee", -40, 100); put("r_knee", 40, 100)
    put("l_ankle", -40, 200); put("r_ankle", 40, 200)
    return p


def pose_with_knee_angle(angle_deg, cx=300.0, cy=300.0):
    """Поза, у которой угол в колене (бедро-колено-голеностоп) равен angle_deg (обе ноги)."""
    p = base_pose(cx, cy)
    a = math.radians(angle_deg)
    for s in ("l", "r"):
        knee = p[K[f"{s}_knee"], :2]
        p[K[f"{s}_ankle"], :2] = knee + 100 * np.array([math.sin(a), -math.cos(a)])
    return p


def knee_frames(angles, times, nulls=(), swaps=(), width=640, height=640):
    """Список frames по серии углов; nulls — индексы кадров без человека, swaps — кадры с обменом сторон."""
    out = []
    for i, (a, t) in enumerate(zip(angles, times)):
        if i in nulls:
            out.append({"i": i, "media_time_s": float(t), "kps": None})
            continue
        p = pose_with_knee_angle(a)
        if i in swaps:
            for l, r in C.LR_PAIRS:
                p[[l, r]] = p[[r, l]]
        out.append({"i": i, "media_time_s": float(t), "kps": p.tolist()})
    return out


def squat_series(fps=30.0, seconds=10.0, period=2.0, lo=90.0, hi=170.0):
    t = np.arange(0, seconds, 1.0 / fps)
    mid, amp = (hi + lo) / 2, (hi - lo) / 2
    return t, mid + amp * np.cos(2 * np.pi * t / period)  # старт в «высоко»


def result(frames, mode="frames", width=640, height=640, **extra):
    r = {"schema": "mpb-result/2", "run_id": "r1", "created_at": "2026-11-27T10:00:00Z",
         "mode": mode, "env": "desktop", "status": "ok",
         "model": {"id": "m", "variant": "v", "runtime": "x", "runtime_version": "1",
                   "backend_actual": "wasm", "input_size": "256x192", "score_source": "score"},
         "source": {"kind": "video", "name": "squat01", "width": width, "height": height, "fps": 30},
         "frames": frames}
    r.update(extra)
    return r

"""Эталонный прогон BlazePose в Python (MediaPipe Tasks, IMAGE, CPU) для сверки с браузером (R1.5).
Пишет JSON в формате раннера (predictions с keypoints17) — его можно подать в eval/evaluate.py.

  <venv с mediapipe==1.1.1>/bin/python -I eval/reference_mediapipe.py --variant lite \
      --ids eval/ids_sub300_seed20261030.txt --images datasets/coco/val2017 --out results/raw/<run_id>.json
"""
import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

COCO_FROM_BLAZEPOSE = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # Таблица 4 статьи
ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="lite")
    ap.add_argument("--ids", default="eval/ids_sub300_seed20261030.txt")
    ap.add_argument("--images", default="datasets/coco/val2017")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model = ROOT / f"models/cache/pose_landmarker_{a.variant}.task"
    opts = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model), delegate=BaseOptions.Delegate.CPU),
                                 running_mode=RunningMode.IMAGE, num_poses=1, min_pose_detection_confidence=0.5,
                                 min_pose_presence_confidence=0.5, min_tracking_confidence=0.5,
                                 output_segmentation_masks=False)
    ids = [int(x) for x in (ROOT / a.ids).read_text().split()]
    preds = []
    with PoseLandmarker.create_from_options(opts) as lm:
        for iid in ids:
            # Как в браузере: JPEG (в т. ч. одноканальный) → RGB.
            bgr = cv2.imread(str(ROOT / a.images / f"{iid:012d}.jpg"), cv2.IMREAD_COLOR)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            w, h = img.width, img.height
            t = time.perf_counter()
            res = lm.detect(img)
            ms = (time.perf_counter() - t) * 1000
            if not res.pose_landmarks:
                preds.append({"image_id": iid, "keypoints17": None, "instance_score": None, "width": w, "height": h,
                              "timings": {"infer_ms": ms}})
                continue
            p = res.pose_landmarks[0]
            kp = [[p[i].x * w, p[i].y * h, p[i].visibility] for i in COCO_FROM_BLAZEPOSE]
            preds.append({"image_id": iid, "keypoints17": kp, "instance_score": sum(k[2] for k in kp) / 17,
                          "width": w, "height": h, "timings": {"infer_ms": ms}})
    date = datetime.now(timezone.utc).isoformat()
    rec = {"run_id": Path(a.out).stem, "date": date, "env": "python-reference", "device": platform.node(),
           "model": f"blazepose-{a.variant}", "variant": a.variant, "backend": "python-cpu", "runtime": "mediapipe (Python)",
           "runtime_version": mp.__version__, "input_size": "256x256", "score_source": "mean_visibility",
           "model_files": [{"file": str(model.relative_to(ROOT)), "sha256": hashlib.sha256(model.read_bytes()).hexdigest()}],
           "dataset": "coco-single", "ids_file": a.ids, "n_images": len(ids), "status": "ok",
           "harness": {"python": platform.python_version(), "opencv": cv2.__version__, "decode": "cv2.imread IMREAD_COLOR → RGB"}, "predictions": preds}
    Path(a.out).write_text(json.dumps(rec))
    print(a.out, sum(p["keypoints17"] is None for p in preds), "без человека из", len(ids))


if __name__ == "__main__":
    main()

"""Эталонный прогон MoveNet SinglePose (Lightning/Thunder v4) в TensorFlow (Python, CPU) для сверки с браузером (R1.5).
Модель — официальная SavedModel TF2 (Kaggle google/movenet/tensorFlow2/singlepose-<variant>/4), т. е. те же веса,
из которых сделана TF.js-модель стенда. Пишет JSON в формате раннера (predictions с keypoints17).

  <python с tensorflow>/python -I eval/reference_movenet.py --variant lightning \
      --model-dir models/cache/tf2/movenet-lightning --ids eval/ids_sub300_seed20261030.txt \
      --images datasets/coco/val2017 --out results/raw/<run_id>.json

Предобработка повторяет bench/adapters/tfjs.js (один проход, без кропа): RGB → центрированный pad нулями
до квадрата → bilinear resize до 192/256 с семантикой tf.js resizeBilinear (alignCorners=false,
halfPixelCenters=false) → int32 с отбрасыванием дробной части (toInt). Выход (y, x, score) → пиксели.
"""
import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
SIZE = {"lightning": 192, "thunder": 256}


def tree_sha256(d):
    h = hashlib.sha256()
    for p in sorted(x for x in d.rglob("*") if x.is_file()):
        h.update(str(p.relative_to(d)).replace("\\", "/").encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(SIZE), required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ids", default="eval/ids_sub300_seed20261030.txt")
    ap.add_argument("--images", default="datasets/coco/val2017")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="local", help="метка устройства в JSON (не имя компьютера: репозиторий публичный)")
    a = ap.parse_args()
    S = SIZE[a.variant]
    model_dir = (ROOT / a.model_dir).resolve()
    fn = tf.saved_model.load(str(model_dir)).signatures["serving_default"]
    ids = [int(x) for x in (ROOT / a.ids).read_text().split()]
    preds = []
    for iid in ids:
        bgr = cv2.imread(str(ROOT / a.images / f"{iid:012d}.jpg"), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        side = max(w, h)
        top, left = (side - h) // 2, (side - w) // 2
        t = time.perf_counter()
        px = tf.constant(rgb[None], dtype=tf.int32)
        padded = tf.pad(px, [[0, 0], [top, side - h - top], [left, side - w - left], [0, 0]])
        x = tf.compat.v1.image.resize_bilinear(tf.cast(padded, tf.float32), [S, S],
                                              align_corners=False, half_pixel_centers=False)
        out = fn(input=tf.cast(x, tf.int32))["output_0"].numpy()[0, 0]  # [17, 3]: y, x, score
        ms = (time.perf_counter() - t) * 1000
        kp = [[float(xx * side - left), float(yy * side - top), float(sc)] for yy, xx, sc in out]
        preds.append({"image_id": iid, "keypoints17": kp, "instance_score": float(np.mean([k[2] for k in kp])),
                      "width": w, "height": h, "timings": {"infer_ms": ms}})
    rec = {"run_id": Path(a.out).stem, "date": datetime.now(timezone.utc).isoformat(), "env": "python-reference",
           "device": a.device, "model": f"movenet-{a.variant}", "variant": a.variant, "backend": "python-cpu",
           "runtime": "TensorFlow (Python), SavedModel TF2", "runtime_version": tf.__version__,
           "input_size": f"{S}x{S}", "score_source": "mean_keypoint_score",
           "model_files": [{"file": str(model_dir.relative_to(ROOT)).replace("\\", "/"), "sha256_tree": tree_sha256(model_dir),
                            "source_url": f"https://www.kaggle.com/api/v1/models/google/movenet/tensorFlow2/singlepose-{a.variant}/4/download"}],
           "dataset": "coco-single", "ids_file": a.ids, "n_images": len(ids), "status": "ok",
           "harness": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__,
                       "decode": "cv2.imread IMREAD_COLOR → RGB",
                       "resize": "tf.compat.v1.image.resize_bilinear(align_corners=False, half_pixel_centers=False)"},
           "predictions": preds}
    Path(a.out).write_text(json.dumps(rec))
    print(a.out, "изображений:", len(ids))


if __name__ == "__main__":
    main()

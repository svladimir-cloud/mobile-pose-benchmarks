"""Эталонный прогон RTMPose-t/-s (+ RTMDet-nano) и RTMO-s в Python через rtmlib (onnxruntime, CPU)
для сверки с браузером (R1.5). Те же ONNX, что в models/manifest.json. Пишет JSON в формате раннера
(predictions с keypoints17), его можно подать в eval/evaluate.py и eval/compare_runs.py.

  <venv с rtmlib, onnxruntime>/python -I eval/reference_rtmlib.py --model rtmpose-t --pose-input rgb \
      --ids eval/ids_sub300_seed20261030.txt --images datasets/coco/val2017 --out results/raw/<run_id>.json

Предобработка и декодирование — из rtmlib (RTMDet.preprocess, RTMPose.preprocess, RTMO.preprocess,
get_simcc_maximum). Отличия от rtmlib «как есть», чтобы условия совпали с браузерным адаптером:
- выбирается один экземпляр с максимальной оценкой (детектор RTMDet ≥ 0,3; RTMO ≥ 0,1), как в bench/adapters/ort.js;
- детектор RTMDet-nano — вход 320×320 (в rtmlib по умолчанию 640);
- --pose-input: rtmlib подаёт в RTMPose изображение как есть (cv2 → BGR) с mean/std в порядке RGB;
  pipeline.json (to_rgb: true) и браузер подают RGB. Оба варианта — отдельными прогонами;
- оценка точки SimCC: rtmlib (эта версия) — среднее max_x и max_y; браузер и mmpose — минимум.
  Пишутся обе: keypoints17[:, 2] — минимум (как в браузере), raw.score_mean — среднее.
"""
import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
from rtmlib import RTMDet, RTMO, RTMPose

ROOT = Path(__file__).resolve().parents[1]
DET_SCORE_THR = 0.3   # как DET_SCORE_THR в bench/adapters/ort.js
RTMO_SCORE_THR = 0.1  # как RTMO_SCORE_THR в bench/adapters/ort.js (pipeline.json)
FILES = {"rtmpose-t": ["onnx/rtmdet-nano.onnx", "onnx/rtmpose-t.onnx"],
         "rtmpose-s": ["onnx/rtmdet-nano.onnx", "onnx/rtmpose-s.onnx"],
         "rtmo-s": ["onnx/rtmo-s.onnx"]}


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_rtmpose(files, pose_input):
    det = RTMDet(str(files[0]), model_input_size=(320, 320), score_thr=DET_SCORE_THR)
    pose = RTMPose(str(files[1]), model_input_size=(192, 256))

    def predict(bgr):
        img, ratio = det.preprocess(bgr)
        dets = det.inference(img)[0][0]          # [N, 5]: x1, y1, x2, y2, score (NMS внутри ONNX)
        if len(dets) == 0 or dets[:, 4].max() < DET_SCORE_THR:
            return None, None, {}
        bi = int(dets[:, 4].argmax())
        box = dets[bi, :4] / ratio
        src = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if pose_input == "rgb" else bgr
        crop, center, scale = pose.preprocess(src, box.tolist())
        simcc_x, simcc_y = pose.inference(crop)
        kpts, score_mean = pose.postprocess([simcc_x, simcc_y], center, scale)
        score_min = np.minimum(simcc_x.max(-1), simcc_y.max(-1))[0]  # как decodeSimCC в браузере
        kp = [[float(x), float(y), float(s)] for (x, y), s in zip(kpts[0], score_min)]
        return kp, float(dets[bi, 4]), {"box": box.tolist(), "score_mean": score_mean[0].tolist()}

    return predict


def make_rtmo(files):
    m = RTMO(str(files[0]), model_input_size=(640, 640), score_thr=RTMO_SCORE_THR)

    def predict(bgr):
        img, ratio = m.preprocess(bgr)           # letterbox в левый верхний угол, pad 114, BGR, без нормировки
        dets, kps = m.inference(img)
        dets, kps = dets[0], kps[0]              # [N, 5], [N, 17, 3]
        if len(dets) == 0 or dets[:, 4].max() < RTMO_SCORE_THR:
            return None, None, {}
        bi = int(dets[:, 4].argmax())
        kp = [[float(x / ratio), float(y / ratio), float(s)] for x, y, s in kps[bi]]
        return kp, float(dets[bi, 4]), {"n_dets": int(len(dets))}

    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(FILES), required=True)
    ap.add_argument("--pose-input", choices=["rgb", "bgr"], default="rgb", help="только для rtmpose-*")
    ap.add_argument("--ids", default="eval/ids_sub300_seed20261030.txt")
    ap.add_argument("--images", default="datasets/coco/val2017")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="local", help="метка устройства в JSON (не имя компьютера: репозиторий публичный)")
    a = ap.parse_args()
    files = [ROOT / "models/cache" / f for f in FILES[a.model]]
    if a.model == "rtmo-s":
        predict, input_note = make_rtmo(files), "BGR (rtmlib и браузер)"
    else:
        predict = make_rtmpose(files, a.pose_input)
        input_note = f"детектор BGR; поза {a.pose_input.upper()}"
    ids = [int(x) for x in (ROOT / a.ids).read_text().split()]
    preds = []
    for iid in ids:
        # Как в браузере и reference_mediapipe.py: JPEG (в т. ч. одноканальный) → 3 канала.
        bgr = cv2.imread(str(ROOT / a.images / f"{iid:012d}.jpg"), cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        t = time.perf_counter()
        kp, score, raw = predict(bgr)
        ms = (time.perf_counter() - t) * 1000
        preds.append({"image_id": iid, "keypoints17": kp, "instance_score": score, "width": w, "height": h,
                      "timings": {"infer_ms": ms}, "raw": raw})
    rec = {"run_id": Path(a.out).stem, "date": datetime.now(timezone.utc).isoformat(), "env": "python-reference",
           "device": a.device, "model": a.model, "variant": a.model, "backend": "python-cpu",
           "runtime": "rtmlib + onnxruntime (Python)", "runtime_version": f"rtmlib {version('rtmlib')}, onnxruntime {onnxruntime.__version__}",
           "input_size": "640x640" if a.model == "rtmo-s" else "256x192 (детектор 320x320)",
           "score_source": "det_score", "pose_input": input_note,
           "model_files": [{"file": str(f.relative_to(ROOT)), "sha256": sha256(f)} for f in files],
           "dataset": "coco-single", "ids_file": a.ids, "n_images": len(ids), "status": "ok",
           "harness": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__,
                       "decode": "cv2.imread IMREAD_COLOR"}, "predictions": preds}
    Path(a.out).write_text(json.dumps(rec))
    print(a.out, sum(p["keypoints17"] is None for p in preds), "без человека из", len(ids))


if __name__ == "__main__":
    main()

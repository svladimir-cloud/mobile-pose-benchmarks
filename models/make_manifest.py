"""Сборка models/manifest.json: источники весов, хеши, размеры, лицензии, конвертация (R1.2).
Запуск из корня: python3 -I models/make_manifest.py  (файлы должны лежать в models/cache/)."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "models/cache"
MMP = "https://download.openmmlab.com/mmpose/v1/projects"
DATE = "2026-10-30"
YOLO_CONV = ("ultralytics 8.3.253, torch 2.15.0 (PyPI), onnx 1.23.2, onnxslim 0.1.98: "
             "yolo export model=<id>.pt format=onnx imgsz=640 opset=17 simplify=True dynamic=False half=False")

MODELS = [
    # id, family, files [(path, source url, note)], license, input, runtime, conversion
    ("blazepose-lite", "BlazePose (MediaPipe Pose Landmarker) Lite", [("pose_landmarker_lite.task",
     "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task", "float16")],
     "Apache-2.0", "256x256", "@mediapipe/tasks-vision 1.1.1", "нет (официальный .task)"),
    ("blazepose-full", "BlazePose Full", [("pose_landmarker_full.task",
     "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task", "float16")],
     "Apache-2.0", "256x256", "@mediapipe/tasks-vision 1.1.1", "нет (официальный .task)"),
    ("blazepose-heavy", "BlazePose Heavy", [("pose_landmarker_heavy.task",
     "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task", "float16")],
     "Apache-2.0", "256x256", "@mediapipe/tasks-vision 1.1.1", "нет (официальный .task)"),
    ("movenet-lightning", "MoveNet SinglePose Lightning v4 (TF.js)", [(f"tfjs/movenet-lightning/{f}",
     "https://www.kaggle.com/api/v1/models/google/movenet/tfJs/singlepose-lightning/4/download", "архив Kaggle (= tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4)")
     for f in ("model.json", "group1-shard1of2.bin", "group1-shard2of2.bin")],
     "Apache-2.0", "192x192", "TF.js", "нет (официальная TF.js-модель)"),
    ("movenet-thunder", "MoveNet SinglePose Thunder v4 (TF.js)", [(f"tfjs/movenet-thunder/{f}",
     "https://www.kaggle.com/api/v1/models/google/movenet/tfJs/singlepose-thunder/4/download", "архив Kaggle (= tfhub.dev/google/tfjs-model/movenet/singlepose/thunder/4)")
     for f in ("model.json", "group1-shard1of3.bin", "group1-shard2of3.bin", "group1-shard3of3.bin")],
     "Apache-2.0", "256x256", "TF.js", "нет (официальная TF.js-модель)"),
    ("rtmpose-t", "RTMPose-t (SimCC, body7) + RTMDet-nano", [
     ("onnx/rtmpose-t.onnx", f"{MMP}/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip", "end2end.onnx из архива"),
     ("onnx/rtmdet-nano.onnx", f"{MMP}/rtmposev1/onnx_sdk/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.zip", "end2end.onnx из архива, NMS внутри")],
     "Apache-2.0", "256x192 (детектор 320x320)", "onnxruntime-web", "нет (официальный ONNX OpenMMLab, mmdeploy)"),
    ("rtmpose-s", "RTMPose-s (SimCC, body7) + RTMDet-nano", [
     ("onnx/rtmpose-s.onnx", f"{MMP}/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip", "end2end.onnx из архива"),
     ("onnx/rtmdet-nano.onnx", f"{MMP}/rtmposev1/onnx_sdk/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.zip", "end2end.onnx из архива, NMS внутри")],
     "Apache-2.0", "256x192 (детектор 320x320)", "onnxruntime-web", "нет (официальный ONNX OpenMMLab, mmdeploy)"),
    ("rtmo-s", "RTMO-s (одноэтапная, body7)", [
     ("onnx/rtmo-s.onnx", f"{MMP}/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip", "end2end.onnx из архива, NMS внутри")],
     "Apache-2.0", "640x640", "onnxruntime-web", "нет (официальный ONNX OpenMMLab, mmdeploy)"),
    ("yolov8n-pose", "YOLOv8n-pose (Ultralytics, COCO)", [
     ("onnx/yolov8n-pose.onnx", "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt", "экспорт из .pt")],
     "AGPL-3.0", "640x640", "onnxruntime-web", YOLO_CONV),
    ("yolo11n-pose", "YOLO11n-pose (Ultralytics, COCO)", [
     ("onnx/yolo11n-pose.onnx", "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt", "экспорт из .pt")],
     "AGPL-3.0", "640x640", "onnxruntime-web", YOLO_CONV),
]


def h(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


out = {"generated": DATE, "note": "файлы — в models/cache/ (не в git); пути относительно models/cache/", "models": []}
for mid, fam, files, lic, inp, rt, conv in MODELS:
    fl = []
    for path, url, note in files:
        p = C / path
        fl.append({"path": path, "source_url": url, "note": note, "downloaded": DATE,
                   "bytes": p.stat().st_size, "sha256": h(p)})
    src = {}
    for path, url, note in files:
        if url.endswith(".pt"):
            pt = C / "dl" / url.rsplit("/", 1)[1]
            src = {"pt_file": f"dl/{pt.name}", "pt_bytes": pt.stat().st_size, "pt_sha256": h(pt)}
    out["models"].append({"id": mid, "family": fam, "license": lic, "input": inp, "runtime": rt,
                          "conversion": conv, "files": fl, **src,
                          "total_mb": round(sum(f["bytes"] for f in fl) / 1e6, 3)})
(ROOT / "models/manifest.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
for m in out["models"]:
    print(m["id"], m["total_mb"], "МБ", m["license"])

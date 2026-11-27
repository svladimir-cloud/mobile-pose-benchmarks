"""Python-реализации тех же моделей: сверка браузера с эталонной реализацией (R1.5) и эталон для своих видео.

  # COCO-SinglePerson (режим images)
  python bench/python/run_python.py --model rtmpose-t --coco [--limit 200]
  # все кадры ролика (режим frames); эталон для eval/exercise.py — ref-rtmpose-x
  python bench/python/run_python.py --model ref-rtmpose-x --frames datasets/media/squat01/frames.json

Модели и предобработка:
  rtmpose-*/ref-rtmpose-x — ONNX из onnx_sdk + ONNX Runtime CPU, предобработка по pipeline.json
                            (TopDownAffine, RGB, mean/std); детектор RTMDet-nano (ref: YOLOX-m), раз в N кадров;
  rtmo-s                  — ONNX, letterbox 640, BGR без нормировки;
  yolov8n-pose/yolo11n-pose — исходный фреймворк Ultralytics (.pt, predict), conf 0.25;
  mediapipe-*             — MediaPipe Tasks Python с тем же .task, делегат CPU.
MoveNet здесь не реализован (нужен TensorFlow) — его сверка не выполняется.
Результат — results/raw/*.json в формате docs/result_format.md, env=python-reference.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "models" / "cache"
MANIFEST = json.loads((ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
MODELS = {m["id"]: m for m in MANIFEST["models"]}
MP33_TO_COCO17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
DET_THR = 0.3


def onnx_file(mid: str) -> Path:
    return next((CACHE / mid).glob("*.onnx"))


def ort_session(path: Path):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def letterbox_tl(img: np.ndarray, size: int):
    h, w = img.shape[:2]
    ratio = min(size / h, size / w)
    out = np.full((size, size, 3), 114, np.uint8)
    rs = cv2.resize(img, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR)
    out[: rs.shape[0], : rs.shape[1]] = rs
    return out, ratio


def best_det(dets: np.ndarray, ratio: float, thr: float = DET_THR):
    d = dets[0]
    if not len(d):
        return None
    i = int(np.argmax(d[:, 4]))
    if d[i, 4] < thr:
        return None
    return d[i, :4] / ratio, float(d[i, 4]), i


class Detector:
    """RTMDet-nano (320, BGR, mean/std BGR) или YOLOX-m (640, BGR, без нормировки)."""

    def __init__(self, mid: str):
        self.sess = ort_session(onnx_file(mid))
        self.size = self.sess.get_inputs()[0].shape[2]
        self.norm = mid == "rtmdet-nano"

    def __call__(self, bgr: np.ndarray):
        x, ratio = letterbox_tl(bgr, self.size)
        x = x.astype(np.float32)
        if self.norm:
            x = (x - np.array([103.53, 116.28, 123.675], np.float32)) / np.array([57.375, 57.12, 58.395], np.float32)
        x = x.transpose(2, 0, 1)[None]
        dets = self.sess.run(None, {self.sess.get_inputs()[0].name: x})[0]
        return best_det(dets, ratio)


def pose_to_box(kps: np.ndarray, expansion: float = 1.25):
    pts = kps[kps[:, 2] >= 0.3]
    if len(pts) < 4:
        pts = kps
    x1, y1 = pts[:, :2].min(0)
    x2, y2 = pts[:, :2].max(0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return np.array([cx - (cx - x1) * expansion, cy - (cy - y1) * expansion, cx + (x2 - cx) * expansion, cy + (y2 - cy) * expansion])


class RtmPose:
    def __init__(self, mid: str, det_interval: int = 1):
        m = MODELS[mid]
        self.sess = ort_session(onnx_file(mid))
        _, _, self.H, self.W = self.sess.get_inputs()[0].shape
        self.det = Detector(m.get("detector", "rtmdet-nano"))
        self.det_interval = max(1, det_interval)
        self.reset()

    def reset(self):
        self.frame, self.last_box = 0, None

    def pose(self, bgr, box):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        sw, sh = (x2 - x1) * 1.25, (y2 - y1) * 1.25
        aspect = self.W / self.H
        if sw > sh * aspect:
            sh = sw / aspect
        else:
            sw = sh * aspect
        s = self.W / sw
        M = np.array([[s, 0, self.W / 2 - s * cx], [0, s, self.H / 2 - s * cy]], np.float32)
        crop = cv2.warpAffine(bgr, M, (self.W, self.H), flags=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = ((rgb - np.array([123.675, 116.28, 103.53], np.float32)) / np.array([58.395, 57.12, 57.375], np.float32))
        sx, sy = self.sess.run(None, {self.sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})
        bx, by = sx[0].argmax(1), sy[0].argmax(1)
        score = 0.5 * (sx[0].max(1) + sy[0].max(1))
        px, py = bx / 2.0, by / 2.0
        return np.stack([px / self.W * sw + cx - sw / 2, py / self.H * sh + cy - sh / 2, score], 1)

    def __call__(self, bgr, image_mode=False):
        box = None
        if image_mode or self.frame % self.det_interval == 0 or self.last_box is None:
            d = self.det(bgr)
            box = d[0] if d else None
        else:
            box = self.last_box
        self.frame += 1
        if box is None:
            self.last_box = None
            return None, None
        kps = self.pose(bgr, box)
        self.last_box = pose_to_box(kps) if kps[:, 2].mean() >= 0.3 else None
        return kps, box


class Rtmo:
    def __init__(self, mid: str, **_):
        self.sess = ort_session(onnx_file(mid))

    def reset(self):
        pass

    def __call__(self, bgr, image_mode=False):
        x, ratio = letterbox_tl(bgr, 640)
        dets, kp = self.sess.run(None, {self.sess.get_inputs()[0].name: x.astype(np.float32).transpose(2, 0, 1)[None]})
        b = best_det(dets, ratio)
        if b is None:
            return None, None
        k = kp[0, b[2]].copy()
        k[:, :2] /= ratio
        return k, b[0]


class Yolo:
    def __init__(self, mid: str, **_):
        from ultralytics import YOLO
        self.model = YOLO(str(next((CACHE / mid).glob("*.pt"))))

    def reset(self):
        pass

    def __call__(self, bgr, image_mode=False):
        r = self.model.predict(bgr, imgsz=640, conf=0.25, device="cpu", verbose=False)[0]
        if r.keypoints is None or len(r.boxes) == 0:
            return None, None
        i = int(r.boxes.conf.argmax())
        xy = r.keypoints.xy[i].cpu().numpy()
        c = r.keypoints.conf[i].cpu().numpy() if r.keypoints.conf is not None else np.ones(len(xy))
        return np.concatenate([xy, c[:, None]], 1), r.boxes.xyxy[i].cpu().numpy()


class MediaPipe:
    def __init__(self, mid: str, image_mode: bool = False, **_):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp, self.vision = mp, vision
        task = next((CACHE / mid).glob("*.task"))
        self.image_mode = image_mode
        self.opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(task), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.IMAGE if image_mode else vision.RunningMode.VIDEO,
            num_poses=1, min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        self.reset()

    def reset(self):
        self.lm = self.vision.PoseLandmarker.create_from_options(self.opts)
        self.ts = 0

    def __call__(self, bgr, image_mode=False):
        h, w = bgr.shape[:2]
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if self.image_mode:
            r = self.lm.detect(img)
        else:
            self.ts += 33
            r = self.lm.detect_for_video(img, self.ts)
        if not r.pose_landmarks:
            return None, None
        L = r.pose_landmarks[0]
        return np.array([[L[i].x * w, L[i].y * h, L[i].visibility or 0.0] for i in MP33_TO_COCO17]), None


def make(mid: str, image_mode: bool, det_interval: int):
    fam = MODELS[mid]["family"]
    if fam == "rtmpose":
        return RtmPose(mid, det_interval=det_interval)
    if fam == "rtmo":
        return Rtmo(mid)
    if fam == "yolo":
        return Yolo(mid)
    if fam == "blazepose":
        return MediaPipe(mid, image_mode=image_mode)
    raise SystemExit(f"{mid}: нет Python-реализации")


def rk(k):
    return None if k is None else [[round(float(a), 1), round(float(b), 1), round(float(c), 3)] for a, b, c in k]


def summarize(v):
    v = np.sort(np.asarray(v, float))
    if not len(v):
        return {"mean": None, "p50": None, "p95": None, "p99": None}
    return {"mean": round(float(v.mean()), 2), **{f"p{q}": round(float(np.percentile(v, q)), 2) for q in (50, 95, 99)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--coco", action="store_true")
    g.add_argument("--frames")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--det-interval", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    m = MODELS[a.model]
    image_mode = a.coco
    est = make(a.model, image_mode, 1 if image_mode else a.det_interval)
    doc = {
        "schema": "mpb-result/2", "run_id": datetime.now().strftime("%Y%m%d-%H%M%S") + "-py",
        "batch_id": None, "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "mode": "images" if image_mode else "frames", "env": "python-reference", "status": "ok", "error": None,
        "device": {"user_label": platform.node(), "platform": platform.system(), "platform_version": platform.release(),
                   "browser": "python", "browser_version": platform.python_version(), "mobile": False},
        "capabilities": {},
        "model": {"id": a.model, "family": m["family"], "variant": m["variant"], "runtime": "python",
                  "runtime_version": "", "backend_requested": "cpu", "backend_actual": "cpu", "threads": None,
                  "input_size": m["input_size"], "detector": m.get("detector"),
                  "det_interval": (1 if image_mode else a.det_interval) if m["family"] == "rtmpose" else None,
                  "files": [], "size_mb": None, "score_source": "visibility" if m["family"] == "blazepose" else "score",
                  "license": m["license"]},
        "params": {"limit": a.limit}, "cold_start": None, "memory": {}, "validity": {},
    }
    lat = []
    if image_mode:
        items = json.loads((ROOT / "datasets/coco/single_person_list.json").read_text(encoding="utf-8"))
        if a.limit:
            items = items[: a.limit]
        preds = []
        for n, it in enumerate(items):
            bgr = cv2.imread(str(ROOT / "datasets/coco/val2017" / it["file_name"]))
            est.reset()
            t0 = time.perf_counter()
            kps, box = est(bgr, image_mode=True)
            lat.append((time.perf_counter() - t0) * 1000)
            preds.append({"image_id": it["image_id"], "kps": rk(kps),
                          "score": round(float(kps[:, 2].mean()), 4) if kps is not None else 0.0,
                          "box": [round(float(v), 1) for v in box] if box is not None else None, "infer_ms": round(lat[-1], 2)})
            if n % 100 == 0:
                print(f"  {n}/{len(items)}", flush=True)
        doc["predictions"] = preds
        doc["source"] = {"kind": "images", "name": "coco-single", "width": None, "height": None, "fps": None, "facing": None}
    else:
        fpath = Path(a.frames)
        meta = json.loads(fpath.read_text(encoding="utf-8"))
        files = meta["files"][: a.limit or None]
        frames = []
        for i, f in enumerate(files):
            bgr = cv2.imread(str(fpath.parent / f))
            t0 = time.perf_counter()
            kps, _ = est(bgr)
            lat.append((time.perf_counter() - t0) * 1000)
            frames.append({"i": i, "media_time_s": round(i / meta["fps"], 4), "t_ms": None,
                           "infer_ms": round(lat[-1], 2), "kps": rk(kps)})
        doc["frames"] = frames
        doc["source"] = {"kind": "frames", "name": fpath.parent.name, "width": meta["width"], "height": meta["height"],
                         "fps": meta["fps"], "facing": None}
    doc["timing"] = {"n_warmup": 0, "n_measure": len(lat), "latency_ms": summarize(lat)}
    out = Path(a.out) if a.out else ROOT / "results/raw" / (
        f"{doc['run_id'][:15]}_python_{a.model}_cpu_{doc['mode']}" + (f"_{doc['source']['name']}" if not image_mode else "") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(f"{a.model}: {len(lat)} -> {out}")


if __name__ == "__main__":
    main()

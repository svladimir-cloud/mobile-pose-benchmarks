"""Видео -> последовательность JPEG-кадров для режима «Точность на кадрах» и эталона.

  python tools/extract_frames.py datasets/media/squat01.mp4 [--fps 30] [--max-side 1280] [--quality 92]

Результат: datasets/media/<имя>/frames/000000.jpg ... и datasets/media/<имя>/frames.json
({count, fps, width, height, files, source}). Все модели и эталон получают одни и те же кадры.
Для показа на телефоне в реальном времени положите видео (mp4, H.264) в datasets/media/ —
при необходимости перекодируйте: ffmpeg -i in.mov -c:v libx264 -pix_fmt yuv420p -movflags +faststart out.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=0, help="0 — исходная частота")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--name", default="")
    a = ap.parse_args()

    src = Path(a.video)
    name = a.name or src.stem
    out = src.parent / name
    fdir = out / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = a.fps or in_fps
    files, t_next, idx, w, h = [], 0.0, 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / in_fps
        idx += 1
        if t + 1e-6 < t_next:
            continue
        t_next += 1.0 / fps
        fh, fw = frame.shape[:2]
        s = min(1.0, a.max_side / max(fh, fw))
        if s < 1.0:
            frame = cv2.resize(frame, (round(fw * s), round(fh * s)), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        fn = f"{len(files):06d}.jpg"
        cv2.imwrite(str(fdir / fn), frame, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        files.append("frames/" + fn)
    meta = {"count": len(files), "fps": fps, "width": w, "height": h, "files": files, "source": src.name}
    (out / "frames.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"{name}: {len(files)} кадров, {w}x{h}, {fps:.2f} кадр/с -> {out}")


if __name__ == "__main__":
    main()

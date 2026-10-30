# Модели и веса (этап 1)

Полные сведения (URL, дата, SHA-256, размер, лицензия, конвертация) — `models/manifest.json`; пересборка — `python3 -I models/make_manifest.py`. Веса лежат в `models/cache/` (не в git).

| id | Источник | Формат в браузере | Вход | Размер, МБ | Лицензия |
|---|---|---|---|---|---|
| blazepose-lite / -full / -heavy | storage.googleapis.com/mediapipe-models (float16 `.task`) | MediaPipe Tasks | 256×256 | 5,78 / 9,40 / 30,66 | Apache-2.0 |
| movenet-lightning / -thunder | Kaggle Models (= TF Hub) `tfJs/singlepose-*/4` | TF.js graph model | 192 / 256 | 4,82 / 12,65 | Apache-2.0 |
| rtmpose-t / -s | OpenMMLab `onnx_sdk` (SimCC, обучение body7) + RTMDet-nano (person) | ONNX (opset 11) | 256×192, детектор 320×320 | 17,36 / 25,90 (вкл. детектор 4,01) | Apache-2.0 |
| rtmo-s | OpenMMLab `onnx_sdk` (body7) | ONNX (opset 11, NMS внутри) | 640×640 | 39,62 | Apache-2.0 |
| yolov8n-pose / yolo11n-pose | GitHub ultralytics/assets v8.3.0 (`.pt`, COCO) → экспорт в ONNX | ONNX (opset 17) | 640×640 | 13,51 / 11,85 | **AGPL-3.0** |

Замечания:
- **RTMPose/RTMO — веса body7** (обучение на 7 наборах, включая COCO train) — это официальные ONNX OpenMMLab и выбор по умолчанию в rtmlib. В статье RTMPose (Табл. 5, COCO-SinglePerson) приведены модели, обученные только на COCO; готовых ONNX для них OpenMMLab не публикует (нужна конвертация из `.pth` через mmdeploy). `TODO(автор)`: body7 (практичный вариант) или COCO-only (прямое сравнение с публикацией).
- YOLO экспортирован командой `yolo export model=<id>.pt format=onnx imgsz=640 opset=17 simplify=True dynamic=False half=False` (ultralytics 8.3.253, torch 2.15.0 с PyPI — `download.pytorch.org` из облака закрыт). Вход 320 — `TODO(автор)`.
- MoveNet взят архивом Kaggle (TF Hub перенаправляет туда же); `model.json` + шарды.

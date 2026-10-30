# bench-ort — прогресс

## 30.10.2026 — адаптер ONNX Runtime Web (`bench/adapters/ort.js`), первый шаг
- `onnxruntime-web` 1.30.0 из `node_modules/onnxruntime-web/dist/`, импорт по URL: `ort.wasm.min.mjs` (бэкенд wasm), `ort.webgpu.min.mjs` (webgpu); `ort.env.wasm.wasmPaths` = каталог dist. Потоки: numThreads по умолчанию = 2 (4 ядра в облаке, `min(4, cores/2)`), crossOriginIsolated=true; пишется в `backendInfo`.
- Пороги (фиксированы заранее, не подгонялись): RTMDet score ≥ 0,3 (иначе `keypoints17: null`); YOLO conf 0,25, NMS IoU 0,7 (как умолчания ultralytics; для top-1 NMS результат не меняет, но выполняется); RTMO score_thr 0,1 (из pipeline.json).
- `scoreSource` = `det_score` для всех (RTMPose: score бокса RTMDet; точки несут SimCC-score = min(max_x, max_y) как `get_simcc_maximum` в mmpose).
- Предобработка: RTMDet — resize keep_ratio в 320×320, pad 114 справа/снизу, порядок каналов BGR, mean [103.53,116.28,123.675] std [57.375,57.12,58.395] (pipeline.json); RTMPose — bbox→center/scale (padding 1.25, отношение 192:256), аффин через canvas `setTransform`+`drawImage`, RGB, mean/std OpenMMLab; YOLO — letterbox по центру 114, RGB /255; RTMO — letterbox (по умолчанию в левый верхний угол как rtmlib), BGR, без нормализации.
- Первый прогон (50 изобр., wasm): `node bench/runner.mjs --models rtmpose-t,yolov8n-pose --backends wasm --limit 50 --warmup 5 --measure 20 --no-log --out-dir <scratch>/ort-runs` — оба ok. AP: rtmpose-t 0,668 (AR 0,710, PCK 0,903, FMR 0,04), yolov8n-pose 0,603 (AR 0,654, PCK 0,893, FMR 0,04).

## Вопросы и предложения

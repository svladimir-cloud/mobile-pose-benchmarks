# bench-ort — прогресс

## 30.10.2026 — адаптер ONNX Runtime Web (`bench/adapters/ort.js`), первый шаг
- `onnxruntime-web` 1.30.0 из `node_modules/onnxruntime-web/dist/`, импорт по URL: `ort.wasm.min.mjs` (бэкенд wasm), `ort.webgpu.min.mjs` (webgpu); `ort.env.wasm.wasmPaths` = каталог dist. Потоки: numThreads по умолчанию = 2 (4 ядра в облаке, `min(4, cores/2)`), crossOriginIsolated=true; пишется в `backendInfo`.
- Пороги (фиксированы заранее, не подгонялись): RTMDet score ≥ 0,3 (иначе `keypoints17: null`); YOLO conf 0,25, NMS IoU 0,7 (как умолчания ultralytics; для top-1 NMS результат не меняет, но выполняется); RTMO score_thr 0,1 (из pipeline.json).
- `scoreSource` = `det_score` для всех (RTMPose: score бокса RTMDet; точки несут SimCC-score = min(max_x, max_y) как `get_simcc_maximum` в mmpose).
- Предобработка: RTMDet — resize keep_ratio в 320×320, pad 114 справа/снизу, порядок каналов BGR, mean [103.53,116.28,123.675] std [57.375,57.12,58.395] (pipeline.json); RTMPose — bbox→center/scale (padding 1.25, отношение 192:256), аффин через canvas `setTransform`+`drawImage`, RGB, mean/std OpenMMLab; YOLO — letterbox по центру 114, RGB /255; RTMO — letterbox (по умолчанию в левый верхний угол как rtmlib), BGR, без нормализации.
- Первый прогон (50 изобр., wasm): `node bench/runner.mjs --models rtmpose-t,yolov8n-pose --backends wasm --limit 50 --warmup 5 --measure 20 --no-log --out-dir <scratch>/ort-runs` — оба ok. AP: rtmpose-t 0,668 (AR 0,710, PCK 0,903, FMR 0,04), yolov8n-pose 0,603 (AR 0,654, PCK 0,893, FMR 0,04).

## Шаг 2 — 5 вариантов, wasm, 50 изображений (облачный headless, env=desktop-headless)
Команда: `node bench/runner.mjs --models <...> --backends wasm --limit 50 --warmup 5 --measure 20 --no-log --out-dir <scratch>/ort-runs`. Метрики: `python3 -I eval/evaluate.py ... --tag ort --bootstrap 0`.
| модель | AP | AP50 | AR | PCK@0.2 | FMR | p50 мс (20 замеров) |
|---|---|---|---|---|---|---|
| rtmpose-t | 0,668 | 0,869 | 0,710 | 0,903 | 0,04 | 76,7 |
| rtmpose-s | 0,756 | 0,896 | 0,782 | 0,946 | 0,04 | 107,4 |
| yolov8n-pose | 0,603 | 0,898 | 0,654 | 0,893 | 0,04 | 323,3 |
| yolo11n-pose | 0,603 | 0,858 | 0,650 | 0,903 | 0,02 | 319,9 |
| rtmo-s | 0,768 | 0,880 | 0,770 | 0,936 | 0,02 | 815,1 |
Расхождений с ожидаемым уровнем >10 п. нет; ошибки стенда (x/y, порядок точек) не обнаружены.

## Сверка YOLOv8n-pose с Python (задел R1.5)
`<scratch>/cmp_yolo.py` в yolo-venv (ultralytics 8.3.253, onnxruntime 1.31.1) на тех же `yolov8n-pose.onnx`, те же 50 изображений, `YOLO(onnx).predict(imgsz=640)`, экземпляр с max conf: по 48 изображениям, где обе стороны нашли человека, средняя евклидова разница 17 точек 2,30 px, медиана 0,5 px, максимум 74,8 px (вероятно, выбран другой экземпляр или шумная точка); среднее |Δconf| точки 0,0019. Различия остаточные: билинейный ресайз canvas vs cv2, JPEG-декодер браузера vs OpenCV. Различие в числе найденных: 0.

## Чувствительность (не подгонка; в адаптере оставлено поведение по pipeline.json / rtmlib)
- RTMPose-t с порядком каналов BGR вместо RGB (50 изобр.): AP 0,698 против 0,668 при RGB — разница в пределах шума 50 изображений; оставлен RGB по `pipeline.json` (to_rgb: true). Опция `rgb:false` оставлена для проверки на 300.
- RTMO-s с letterbox по центру вместо левого верхнего угла (50 изобр.): AP 0,765 против 0,768 — разницы нет; оставлен левый верх (rtmlib).
- WebGPU: поднимается в headless Chromium на SwiftShader (программный адаптер), rtmpose-t дал ok на 5 изображениях; ORT пишет предупреждение «Some nodes were not assigned to the preferred EP» (shape-операции на CPU — штатно). Скорость на SwiftShader не репрезентативна.
- Потоков wasm: 2 (авто), crossOriginIsolated=true.

## Шаг 3 — 300 изображений, wasm (env=desktop-headless, `--warmup 50 --measure 200`)
JSON: `<scratch>/ort-runs-300/*_wasm.json`. Во время прогона параллельно работал раннер bench-tfjs на тех же 4 ядрах: задержки зашумлены и завышены.
| модель | AP | AP50 | AR | PCK@0.2 | FMR | p50 мс | p95 мс | cold_load/init/first_infer мс |
|---|---|---|---|---|---|---|---|---|
| rtmpose-t | 0,639 | 0,869 | 0,700 | 0,892 | 0,017 | 83,1 | 106,4 | 153/560/313 |
| rtmpose-s | 0,701 | 0,890 | 0,748 | 0,912 | 0,017 | 105,6 | 163,8 | 177/888/371 |
| yolov8n-pose | 0,587 | 0,876 | 0,646 | 0,865 | 0,040 | 309,2 | 417,5 | 210/696/534 |
| yolo11n-pose | 0,550 | 0,855 | 0,615 | 0,843 | 0,027 | 270,5 | 368,2 | 162/972/505 |
| rtmo-s | 0,755 | 0,917 | 0,775 | 0,927 | 0,003 | 771,5 | 900,3 | 428/940/797 |
RTMPose-t: 0,639 против 72,1 в статье (веса body7, иной протокол) — отставание ~8 п., меньше порога 10 п.

## WebGPU (SwiftShader, программный)
300 изображений недостижимы: ~9,4 с на инференс rtmpose-t. Прогон `--limit 20 --warmup 3 --measure 10`: `<scratch>/ort-runs-gpu20/`. rtmpose-t ok (p50 9351 мс, first_infer 18,4 с); yolov8n-pose и yolo11n-pose — timeout 10 мин (first_infer ~39 с у yolov8n). rtmo-s и rtmpose-s на момент сдачи не завершены.

Итог WebGPU (SwiftShader, 20 изображений, `--warmup 3 --measure 10`, JSON в `<scratch>/ort-runs-gpu20/`): rtmpose-t ok (p50 9351 мс, first_infer 18,4 с); rtmpose-s ok (p50 6482 мс, first_infer 18,8 с); yolov8n-pose, yolo11n-pose, rtmo-s — timeout 10 мин (first_infer 38,8 / 41,7 / 88,9 с). AP для WebGPU не считался.

## Вопросы и предложения
- runner.mjs: после последней пары процесс не завершается (server.close ждёт keep-alive); предлагается server.closeAllConnections() / process.exit().
- Прогон WebGPU в облаке на SwiftShader непригоден для скорости; нужен реальный GPU/телефон.

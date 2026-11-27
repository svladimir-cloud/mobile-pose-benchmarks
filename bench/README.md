# Стенд замеров в браузере

```
bench/
  server.py           HTTPS-сервер: страница, рантаймы, модели, видео, приём результатов (COOP/COEP -> многопоточный WASM)
  web/index.html      страница замеров (телефон и ПК)
  web/js/app.js       режимы, очередь конфигураций, отправка результатов
  web/js/runner.js    цикл замера: холодный старт, прогрев, замер, длительный прогон, кадры, COCO
  web/js/models.js    адаптеры: MediaPipe Tasks, MoveNet (TF.js), RTMPose+RTMDet, RTMO, YOLO (ONNX Runtime Web)
  web/js/caps.js      возможности браузера и сведения об устройстве
  web/vendor/         рантаймы из node_modules (npm run vendor; не коммитятся)
  python/run_python.py  те же модели на Python (сверка R1.5, эталон RTMPose-x для своих видео)
```

Запуск и порядок замеров — `docs/phone_protocol.md`. Формат результатов — `docs/result_format.md`.

## Модели и бэкенды

| Модель | Рантайм | Бэкенды | Вход | Заметки |
|---|---|---|---|---|
| mediapipe-lite / full / heavy | MediaPipe Tasks Vision 1.1.1 | gpu (WebGL), cpu (WASM SIMD) | 256×256 | режим VIDEO (трекинг), 33 → 17 точек по Табл. 4, score = visibility |
| movenet-lightning / thunder | TF.js 4.22 + pose-detection 2.1.3 | webgl, wasm (SIMD + потоки), webgpu | 192 / 256 | официальная модель TF Hub, сглаживание выключено |
| rtmpose-t / s | ONNX Runtime Web 1.30 | webgpu, wasm (SIMD + потоки), webgl | 256×192 | детектор RTMDet-nano раз в N кадров (по умолчанию 10), между ними — бокс по точкам прошлого кадра (как rtmlib PoseTracker); веса body7 |
| rtmo-s | ONNX Runtime Web | webgpu, wasm, webgl | 640×640 | одноэтапная |
| yolov8n-pose / yolo11n-pose | ONNX Runtime Web | webgpu, wasm, webgl | 640×640 | экспорт Ultralytics в ONNX; лицензия AGPL-3.0 |

Предобработка повторяет эталонные конвейеры: `pipeline.json` из ONNX-архивов OpenMMLab (RTMPose — RGB + mean/std, аффинный кроп с padding 1,25; RTMDet — BGR + mean/std; RTMO — BGR без нормировки) и LetterBox Ultralytics. «Человек найден»: score детектора ≥ 0,3 (RTMDet, RTMO), conf ≥ 0,25 (YOLO), внутренние пороги 0,5 у MediaPipe, score позы ≥ 0,25 у MoveNet.

## Что измеряется

- **Холодный старт:** скачивание файлов модели (мс, байты, из кэша или нет), инициализация (создание сессии, компиляция), первый кадр.
- **Скорость:** задержка инференса с пред- и постобработкой и чтением результата с GPU (p50/p95/p99); частота обработанных кадров; сквозная задержка «кадр источника → скелет отрисован»; доля пропущенных кадров источника (по `presentedFrames`). Прогрев ≥ 50 кадров или 15 с, замер 200 кадров или 40 с.
- **Длительный прогон:** поминутные частота и p50/p95, заряд, отношение 10-й минуты к 1-й.
- **Точность:** покадровые точки (режимы видео и кадров) для `eval/exercise.py`, `eval/fmin.py`, `eval/jitter.py`; предсказания на COCO-SinglePerson для `eval/coco_eval.py`.

## Проверка в облаке

```bash
python bench/server.py --http --port 8080 &
node tools/smoke.mjs
```

Headless Chromium в облаке не имеет GPU (WebGL/WebGPU программные), поэтому скорость отсюда — только проверка стенда (`env=desktop-headless`).

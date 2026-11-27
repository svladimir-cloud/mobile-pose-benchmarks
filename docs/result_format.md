# Формат сырых результатов (`results/raw/*.json`)

Один файл — один прогон одной конфигурации «модель × бэкенд × источник» на одном устройстве. Пишет страница стенда (`bench/web/`) или эталонные Python-скрипты (`eval/reference.py`, `eval/python_models.py`). Читают скрипты `eval/`.

Имя файла: `<дата YYYYMMDD-HHMMSS>_<устройство>_<модель>_<бэкенд>_<режим>.json` (R5.2; символы вне `[A-Za-z0-9.-]` заменяются на `-`).

## Общая структура

```jsonc
{
  "schema": "mpb-result/2",
  "run_id": "20261127-101500-ab12",        // уникален
  "batch_id": "20261127-101200-xy",        // один запуск матрицы на устройстве
  "created_at": "2026-11-27T10:15:00.000Z",
  "mode": "speed" | "long" | "frames" | "images",
  "env": "phone" | "desktop" | "desktop-headless",
  "status": "ok" | "failed",
  "error": null,                            // текст причины при failed

  "device": {
    "user_label": "Redmi Note 11",          // ввёл участник (может быть пусто)
    "ua": "...", "ua_model": "23021RAAEG",  // UA-CH model, если доступен
    "platform": "Android", "platform_version": "13.0.0",
    "browser": "Chrome", "browser_version": "131.0.6778.81",
    "mobile": true,
    "hardware_concurrency": 8, "device_memory_gb": 4,
    "screen": [1080, 2400], "dpr": 2.75,
    "battery": { "level": 0.81, "charging": false },   // null, если API нет
    "temp_c_start": null,                    // ввёл участник
    "soc": ""                                // ввёл участник или заполняется при анализе
  },

  "capabilities": {
    "cross_origin_isolated": true,
    "wasm_simd": true, "wasm_threads": true,
    "webgl2": true, "webgpu": true,
    "webgpu_adapter": "qualcomm/adreno-6xx" // vendor/architecture или "no-adapter"/"no-api"
  },

  "model": {
    "id": "rtmpose-t",                       // id из models/manifest.json
    "family": "rtmpose", "variant": "t",
    "runtime": "onnxruntime-web", "runtime_version": "1.30.0",
    "backend_requested": "webgpu",
    "backend_actual": "webgpu",              // что реально исполнялось (или "wasm" после отката)
    "threads": 4,                            // для WASM
    "input_size": "256x192",
    "detector": "rtmdet-nano", "det_interval": 10,   // только для двухэтапных
    "files": [{ "url": "/models/rtmpose-t.onnx", "bytes": 13400000 }],
    "size_mb": 13.4,
    "score_source": "score" | "visibility"
  },

  "source": {
    "kind": "camera" | "video" | "frames" | "images",
    "name": "squat01",                        // имя клипа/набора
    "width": 720, "height": 1280,
    "fps": 30,                                // для video/frames
    "facing": "user" | "environment" | null   // для камеры
  },

  "params": {
    "warmup_frames": 50, "warmup_max_s": 15,
    "measure_frames": 200, "measure_max_s": 40,
    "long_minutes": 10,
    "seed": 12345, "order_index": 3, "repeat": 1
  },

  "cold_start": {
    "download_ms": 812.4, "download_bytes": 13400000, "from_cache": false,
    "init_ms": 640.2,
    "first_infer_ms": 1830.0
  },

  "timing": {
    "n_warmup": 50, "n_measure": 200, "duration_s": 12.3,
    "latency_ms": { "mean": 41.2, "p50": 40.0, "p95": 55.1, "p99": 61.0 },   // инференс с пред-/постобработкой
    "e2e_latency_ms": { "mean": 70.0, "p50": 68.0, "p95": 90.0, "p99": 99.0 },// кадр источника -> скелет отрисован
    "model_fps": 24.3,                       // обработанные кадры / время
    "e2e_fps": 23.9,                         // отрисованные скелеты / время
    "frames_presented": 360,                 // кадров источника за время замера
    "frames_processed": 292,
    "dropped_frames_pct": 18.9,              // 100 * (1 - processed/presented)
    "per_minute": [                          // только mode=long
      { "minute": 1, "fps": 24.1, "p50": 40.2, "p95": 55.0, "battery": 0.80 }
    ],
    "fps_ratio_10_1": 0.83                   // только mode=long
  },

  "memory": { "js_heap_mb_peak": 210.5 },    // null вне Chrome
  "validity": { "hidden_during_run": false, "wake_lock": true, "notes": "" },

  // режимы video/frames/camera(по желанию): покадровые точки
  "frames": [
    { "i": 0, "media_time_s": 0.0, "t_ms": 12.3, "infer_ms": 40.1,
      "kps": [[x, y, s], ...17]   // пиксели исходного кадра, порядок COCO-17; null, если человек не найден
    }
  ],

  // режим images (COCO): одно предсказание на изображение
  "predictions": [
    { "image_id": 397133, "kps": [[x, y, s], ...17] | null, "score": 0.83, "box": [x1, y1, x2, y2] | null, "infer_ms": 41.0 }
  ]
}
```

## Правила

- Координаты — в пикселях **исходного** кадра/изображения, порядок COCO-17: nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle. «Левая/правая» — анатомические стороны человека.
- BlazePose 33 → COCO-17 — по Таблице 4 статьи: 0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28. Score точки — `visibility`.
- Score экземпляра (`predictions[].score`): среднее score 17 точек (для двухэтапных моделей — произведение score бокса и среднего score точек не используется, чтобы правило было одинаковым для всех моделей).
- `kps = null` — модель не вернула человека. Точку с низким score не обнуляем: порог τ применяется при анализе.
- Время — `performance.now()`, мс. `latency_ms` начинается до предобработки и заканчивается после постобработки и синхронизации с GPU (результат прочитан на CPU).
- `e2e_latency_ms`: от `captureTime`/`expectedDisplayTime` кадра (requestVideoFrameCallback) до конца отрисовки скелета на canvas. Для режима `frames` не считается.
- `dropped_frames_pct` считается по `presentedFrames` (requestVideoFrameCallback), то есть по кадрам, которые источник выдал, но модель не успела обработать.
- Окружение: облачный headless Chromium — только `desktop-headless`; такие числа скорости не являются результатами (AGENTS.md §4).

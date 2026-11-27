# bench-mediapipe — прогресс

## 30.10.2026 — адаптер BlazePose (этап 0)
- `bench/adapters/mediapipe.js`: `@mediapipe/tasks-vision` **1.1.1** (npm, точная версия; 1.2.0 вышел 27.10.2026 — моложе двух недель, не взят), режим `IMAGE`, `numPoses=1`, пороги детекции/присутствия/трекинга по умолчанию 0,5, без масок.
- Модель: `pose_landmarker_lite.task` (float16, `storage.googleapis.com/mediapipe-models/.../float16/latest/`), 5 777 746 Б, SHA-256 `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`, Last-Modified 27.04.2023. Лежит в `models/cache/` (не в git).
- Бэкенды: `wasm` → `delegate: 'CPU'` (XNNPACK), `webgl` → `delegate: 'GPU'`.
- Приведение 33 → 17: `COCO_FROM_BLAZEPOSE = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]` (Таблица 4). score точки = `visibility`; `presence` сохраняется в `raw`.
- **Проверка порядка/сторон (R1.4):** на 300 изображениях AP с картой по Таблице 4 = 0,311, с переставленными сторонами = 0,002; перестановка улучшает OKS лишь на 13 из 220 найденных людей (реальные ошибки модели). Вывод: «left_*» MediaPipe Tasks 1.1.1 = анатомическая левая сторона, совпадает с COCO.
- Предупреждение MediaPipe `Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI` — проверено: средний OKS не хуже на неквадратных кадрах (ш/в < 0,9: 0,71; 0,9–1,1: 0,63; > 1,1: 0,66), признаков искажения нет.
- Задержка `predict` = вызов `detect()` (синхронный; включает пред-/постобработку и чтение результата с GPU). Загрузка и декодирование JPEG — вне замера.

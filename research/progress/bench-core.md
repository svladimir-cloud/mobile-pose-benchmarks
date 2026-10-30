# bench-core — прогресс

## 30.10.2026 — стенд этапа 0
- `bench/server.mjs` — статический сервер корня репозитория (127.0.0.1, COOP/COEP → `crossOriginIsolated`, `Cache-Control: no-store`).
- `bench/index.html` + `bench/env.js` — страница стенда, `window.bench` (init / firstInfer / runImages / runLatency / probeEnv).
- `bench/runner.mjs` — Playwright 1.63.0 (глобальный), Chromium 153.0.8010.12 headless; каждый бэкенд — новый процесс браузера (холодный старт, пустой профиль). Порядок: probeEnv → init (загрузка модели, инициализация) → первый инференс → 300 изображений по одному разу → 50 прогревов + 200 замеров по кругу на первых 20 изображениях.
- WebGL в облаке — программный: `--use-angle=swiftshader --enable-unsafe-swiftshader` → `ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero)))`. WebGPU — нет адаптера.
- Запуск: `node bench/runner.mjs --ids eval/ids_sub300_seed20261030.txt --backends wasm,webgl --warmup 50 --measure 200`.
- Прогоны: `results/raw/20261030-114815_cloud-desktop_blazepose-lite_wasm.json`, `..._114857_..._webgl.json`; пробный `..._114626_smoke_...` (5/10 замеров, только проверка).
- Замечено: GPU-делегат (SwiftShader) на тех же кадрах даёт 89 пропусков против 80 у CPU, 19 изображений расходятся по факту детекции, медиана макс. расхождения координат 17,8 px — нужна сверка на реальном GPU (R1.5).

## 30.10.2026 — страница для телефона
- `bench/phone/template.html` + `bench/phone/build.mjs` → `bench/phone/dist/phone.html` (один файл, 3,79 МБ, `page_version` 7c2e00bd11): адаптер `bench/adapters/mediapipe.js` и `bench/env.js` встраиваются как есть; 20 изображений COCO с лицензиями CC BY / CC BY-SA (`bench/phone/images.json`, первые 20 подходящих по лицензии из подвыборки 300) — base64.
- Рантайм `@mediapipe/tasks-vision@1.1.1` с jsDelivr, модель — с storage.googleapis.com; `?local=1` — из репозитория.
- Адаптер: импорт рантайма стал динамическим (`visionUrl`), добавлен этап `cold_import_ms`. Проверка: прогон `20261030-120542_smoke-refactor_...` на 20 изображениях даёт точки, совпадающие бит в бит с прогоном `..._114815_..._wasm` (WASM детерминирован).
- `bench/phone/test_phone.mjs`: эмуляция Pixel 7 в headless Chromium, запросы к CDN перехватываются и отдаются локально. WASM и WebGL: «Готово», CSV скачивается (44 / 34 строки при 20 / 10 замерах).

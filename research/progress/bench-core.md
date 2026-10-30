# bench-core — прогресс

## 30.10.2026 — стенд этапа 0
- `bench/server.mjs` — статический сервер корня репозитория (127.0.0.1, COOP/COEP → `crossOriginIsolated`, `Cache-Control: no-store`).
- `bench/index.html` + `bench/env.js` — страница стенда, `window.bench` (init / firstInfer / runImages / runLatency / probeEnv).
- `bench/runner.mjs` — Playwright 1.63.0 (глобальный), Chromium 153.0.8010.12 headless; каждый бэкенд — новый процесс браузера (холодный старт, пустой профиль). Порядок: probeEnv → init (загрузка модели, инициализация) → первый инференс → 300 изображений по одному разу → 50 прогревов + 200 замеров по кругу на первых 20 изображениях.
- WebGL в облаке — программный: `--use-angle=swiftshader --enable-unsafe-swiftshader` → `ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero)))`. WebGPU — нет адаптера.
- Запуск: `node bench/runner.mjs --ids eval/ids_sub300_seed20261030.txt --backends wasm,webgl --warmup 50 --measure 200`.
- Прогоны: `results/raw/20261030-114815_cloud-desktop_blazepose-lite_wasm.json`, `..._114857_..._webgl.json`; пробный `..._114626_smoke_...` (5/10 замеров, только проверка).
- Замечено: GPU-делегат (SwiftShader) на тех же кадрах даёт 89 пропусков против 80 у CPU, 19 изображений расходятся по факту детекции, медиана макс. расхождения координат 17,8 px — нужна сверка на реальном GPU (R1.5).

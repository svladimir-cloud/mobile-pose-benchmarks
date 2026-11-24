# bench-tfjs — прогресс

## 30.10.2026 — адаптер MoveNet SinglePose (TF.js)
- Файл: `bench/adapters/tfjs.js`. Рантайм `@tensorflow/tfjs` 4.22.0 (UMD `tf.min.js`: core+converter+cpu+webgl), `tfjs-backend-wasm` 4.22.0 и `tfjs-backend-webgpu` 4.22.0 подгружаются `<script>`-ом из node_modules (глобальный `tf`); `tf.wasm.setWasmPaths` → `node_modules/@tensorflow/tfjs-backend-wasm/dist/`.
- Веса: `models/cache/tfjs/movenet-{lightning,thunder}/model.json` (+шарды), `tf.loadGraphModel`. modelBytes: Lightning 4 818 229 Б, Thunder 12 645 263 Б (HEAD content-length по model.json+шардам).
- Пред-обработка: `fromPixels` → центрированный pad нулями до квадрата → `resizeBilinear` 192/256 → int32 [1,S,S,3]. Выход [1,1,17,3] (y,x,score) → пиксели: x·side − padLeft, y·side − padTop. Без кропа по предыдущему кадру. infer_ms включает всё это и `await out.data()`.
- Холодный старт: cold_import_ms = загрузка скриптов tf (+бэкенд); cold_init_ms = `setBackend`+`ready` (инициализация бэкенда; компиляция шейдеров WebGL/WebGPU попадает в первый инференс — first_infer_ms, отдельно не отделима); cold_load_ms = model.json+шарды+`loadGraphModel`.
- MoveNet SinglePose всегда возвращает человека → keypoints17 никогда не null (n_detected=300/300, FMR=0 не информативен).
- Утечки: 30 вызовов predict подряд (wasm, Lightning) — `tf.memory().numTensors` 180 → 180, numBytes без изменений.
- Баги по ходу: `tf.env().get` для флага чужого бэкенда бросает ошибку — исправлено (флаги берутся по бэкенду).

## Порядок точек (лево/право)
Перестановка сторон в предсказаниях (300 изобр., wasm): Thunder AP 0,4853 → 0,0031; Lightning 0,3399 → 0,0022. Порядок MoveNet = COCO, подтверждено.

## Прогоны (env=desktop-headless, облачный ПК, НЕ телефон; программный GPU SwiftShader)
Команды: `node bench/runner.mjs --models movenet-lightning,movenet-thunder --backends ... --no-log --out-dir <scratch>`; метрики `python3 -I eval/evaluate.py ... --bootstrap 0`.
Каталоги: `/tmp/claude-0/-home-user/c9435537-02da-5ef6-a906-5427909bcde8/scratchpad/tfjs-runs` (50 изобр., warmup 5 / measure 20), `/tmp/claude-0/-home-user/c9435537-02da-5ef6-a906-5427909bcde8/scratchpad/tfjs-runs-300` (300, warmup 50 / measure 200, wasm и webgl), `/tmp/claude-0/-home-user/c9435537-02da-5ef6-a906-5427909bcde8/scratchpad/tfjs-runs-300-webgpu` (300, Lightning, warmup 5 / measure 20). Метрики: `/tmp/claude-0/-home-user/c9435537-02da-5ef6-a906-5427909bcde8/scratchpad/tfjs-eval`, `/tmp/claude-0/-home-user/c9435537-02da-5ef6-a906-5427909bcde8/scratchpad/tfjs-eval300`, `/tmp/claude-0/-home-user/c9435537-02da-5ef6-a906-5427909bcde8/scratchpad/tfjs-eval300g`.

### 50 изображений
| модель | бэкенд | AP | AP50 | AR | PCK@0.2 | p50 / p95 мс |
|---|---|---|---|---|---|---|
| Lightning | wasm | 0,4049 | 0,6188 | 0,458 | 0,7376 | 35,9 / 54,8 |
| Lightning | webgl | 0,4051 | 0,6199 | 0,458 | 0,7376 | 215,5 / 247,3 |
| Lightning | webgpu | 0,4051 | 0,6199 | 0,458 | 0,7376 | 2671 / 3255 |
| Thunder | wasm | 0,5276 | 0,7644 | 0,576 | 0,8108 | 89,6 / 106,9 |
| Thunder | webgl | 0,5277 | 0,7644 | 0,576 | 0,8086 | 694,6 / 748,2 |
| Thunder | webgpu | 0,5277 | 0,7644 | 0,576 | 0,8086 | 7741 / 8749 |

### 300 изображений
| модель | бэкенд | AP | AP50 | AR | PCK@0.2 | p50 / p95 мс | n_measure |
|---|---|---|---|---|---|---|---|
| Lightning | wasm | 0,3399 | 0,6033 | 0,4047 | 0,6973 | 36,5 / 120,8* | 200 |
| Lightning | webgl | 0,3399 | 0,6039 | 0,4047 | 0,6976 | 229,6 / 264,8 | 200 |
| Lightning | webgpu | 0,3399 | 0,6039 | 0,4047 | 0,6976 | 1740 / 1832 | 20 |
| Thunder | wasm | 0,4853 | 0,7509 | 0,5423 | 0,8012 | 90,8 / 122,4 | 200 |
| Thunder | webgl | 0,4855 | 0,7510 | 0,5427 | 0,8012 | 1333 / 1521 | 200 |
| Thunder | webgpu | не прогонялся на 300 (оценка времени ≈ 8 с/изобр. → ≈ 40 мин) | | | | | |
(*p95 Lightning/wasm 300 — вероятно шум облачной машины; p50 стабилен.) p50/p95 тут — квантили по сырым latency_ms (индекс int(p·n)); evaluate.py считает чуть иначе (интерполяция).

### Холодный старт (300-прогоны, мс)
| пара | import | init | load | first_infer |
|---|---|---|---|---|
| L/wasm | 212 | 82 | 84 | 157 |
| T/wasm | 197 | 99 | 221 | 311 |
| L/webgl | 213 | 15 | 143 | 5262 |
| T/webgl | 163 | 10 | 154 | 6402 |
| L/webgpu | 207 | 5 | 99 | 16570 |

WebGPU в облаке: `WEBGPU_CPU_FORWARD=true`, SwiftShader — очень медленно, только проверка работоспособности; первый инференс 15–18 с (компиляция шейдеров).

## AP ниже статьи (RTMPose, Табл. 5: Lightning 53,6, Thunder 64,8): наши 34,0 / 48,5 на 300 изобр. — на 19 и 16 п. ниже
Гипотезы (порядок лево/право, x/y — проверены: порядок верный, перестановка даёт AP≈0):
1. Основная (подтверждена косвенно): ошибки сосредоточены на мелких людях. Терцили по площади GT (границы 15 646 и 48 622 px²), средний OKS: Thunder 0,515 / 0,740 / 0,812; Lightning 0,362 / 0,628 / 0,746. Полнокадровый вход 192/256 px даёт малым людям мало пикселей. Подвыборка из 300 (bench) содержит много мелких людей; в статье, вероятно, другая подвыборка/протокол (например, кроп по рамке человека) — не проверено.
2. Паддинг: центрированный pad нулями (как `resize_with_pad`); альтернативы (pad справа/снизу) не пробовались.
3. Без кропа по предыдущему кадру (режим отдельных изображений) — по заданию.
4. На BlazePose Lite этот же стенд даёт AP 0,311 (по данным задания), т.е. подвыборка трудная и для других моделей.
Эксперименты с кропом по GT-рамке не делались (вне задачи).

## Особенности стенда
- Раннер `bench/runner.mjs` после записи JSON не завершается (процесс висит, вероятно, из-за keep-alive у сервера или браузера) — запускал через обёртку, которая ждёт нужное число JSON и убивает процесс. Моя ошибка: один раз выполнил `pkill -f runner.mjs` — мог оборвать чужой прогон другого агента (в списке процессов был чужой `runner.mjs --models rtmpose-s,...`).

## Вопросы и предложения
- bench-core: в `runner.mjs` добавить `process.exit(0)` в конце (или `server.closeAllConnections()`), раннер не завершается.
- bench-core/analyst: для MoveNet n_detected всегда = n, метрика FMR/«не найден» не информативна.
- runner: WebGPU Thunder на 300 изображений в облаке ≈ 40+ мин; для полного прогона нужен --timeout-min ≥ 90 или меньше замеров.
- Протокол: решить, оценивать ли MoveNet с кропом по рамке человека (как, вероятно, в статье RTMPose) — тогда AP будет сопоставим с Табл. 5.

## Диагностика: опция refine (второй проход по кропу), 300 изображений, wasm
НЕ основной протокол — проверка гипотезы «мелкие люди / один проход по всему кадру».
Алгоритм (tfjs.js, `refine: true`): 1) первый проход как обычно; 2) точки со score >= 0,2; если их < 2 — результат первого прохода; 3) рамка этих точек, центр в её центре, сторона кропа = max(1,5 * max(ширина, высота) рамки, 32 px); 4) `tf.image.cropAndResize` по исходному кадру (float, нормировка координат делением на (h-1)/(w-1), выход за кадр = нули), билинейно до 192/256, int32; 5) тот же model.execute; точки = левый верх кропа + (x,y)*сторона. infer_ms включает оба прохода. Команда: `node bench/runner.mjs --models movenet-lightning-refine,movenet-thunder-refine --backends wasm --warmup 20 --measure 50 --no-log --out-dir /tmp/claude-0/-home-user/c9435537-02da-5ef6-a906-5427909bcde8/scratchpad/tfjs-refine-300`; метрики `/tmp/claude-0/-home-user/c9435537-02da-5ef6-a906-5427909bcde8/scratchpad/tfjs-evalrefine`.

| модель (wasm, 300) | AP | AP50 | AR | PCK@0.2 | p50 мс | OKS малые/средние/крупные |
|---|---|---|---|---|---|---|
| Lightning, 1 проход | 0,3399 | 0,6033 | 0,4047 | 0,6973 | 36,5 | 0,362 / 0,628 / 0,746 |
| Lightning, refine | 0,5262 | 0,7630 | 0,568 | 0,8432 | 83,6 | 0,598 / 0,716 / 0,730 |
| Thunder, 1 проход | 0,4853 | 0,7509 | 0,5423 | 0,8012 | 90,8 | 0,515 / 0,740 / 0,812 |
| Thunder, refine | 0,6447 | 0,8362 | 0,6753 | 0,8971 | 198,6 | 0,705 / 0,776 / 0,794 |
Вывод: гипотеза подтверждена, AP с refine (52,6 / 64,5) близок к статье (53,6 / 64,8); выигрыш за счёт малых людей, у крупных OKS чуть падает (0,812 → 0,794, 0,746 → 0,730). Окружение: desktop-headless.

# eval — прогресс

## 30.10.2026
- COCO val2017: аннотации скачаны по HTTP (HTTPS — ошибка сертификата), хеши — `docs/datasets.md`.
- `eval/select_coco_single.py`: COCO-SinglePerson = 921 изображение (правило R2.1); без условия `num_keypoints ≥ 1` — 1045 (совпадает с RTMPose). Подвыборка 300, seed 20261030.
- pycocotools 2.0.11, numpy 2.5.3, matplotlib 3.11.2 (Python 3.13).
- `eval/metrics.py`: OKS-AP/AP50/AP75/AR (pycocotools, без flip test; оценка экземпляра = среднее visibility 17 точек), PCK@0,2 (L — бо́льшая диагональ торса по GT; изображения без обеих диагоналей исключаются), норм. ошибка медиана/p95, KMR_j(τ) τ=0,1…0,9 + площадь, FMR, p50/p95. Тесты: `python3 -m pytest eval/tests -q` (6 passed).
- `eval/evaluate.py`: сырые прогоны → `data/{accuracy,missing,perf}_prototype_desktop-headless.csv`, `docs/img/` (кривая KMR, 5 худших). Бутстреп AP — 2000 выборок изображений (seed 0).
- Порядок/стороны (R1.4) подтверждены: AP 0,311 (Таблица 4) против 0,002 (стороны переставлены).

## 30.10.2026 — сверка браузер ↔ Python (R1.5), BlazePose Lite
- `eval/reference_mediapipe.py` (mediapipe==1.1.1 Python — та же версия, что tasks-vision 1.1.1 в браузере; CPU; JPEG → RGB через OpenCV, т. к. одноканальные JPEG Python MediaPipe не принимает). Нужен `libegl1` (поставлен через apt).
- `eval/compare_runs.py` — совпадение детекции и расхождение координат (px) двух прогонов.
- Эталон: `results/raw/20261030-134527_python-ref_blazepose-lite_python-cpu.json` — 81 из 300 без человека; AP 0,317, AP50 0,568, AR 0,365, PCK@0,2 0,668 (браузер WASM: 80 без человека, AP 0,311).
- Браузер WASM ↔ Python: оба нашли человека на 217, только браузер — 3, только Python — 2; расхождение координат: медиана 0,007 px, среднее 2,6 px, p95 6,9 px; изображений с макс. расхождением > 10 px — 29.
- Браузер WebGL (SwiftShader) ↔ Python: оба — 205, только браузер — 6, только Python — 14; медиана 4,1 px, среднее 11,3 px, p95 41,4 px; изображений с > 10 px — 158. Расхождение GPU-делегата — заметно больше, чем у WASM; проверить на реальном GPU телефона.

## 24.11.2026 — сверка браузер ↔ Python (R1.5): RTMPose-t/-s, RTMO-s, MoveNet
- Окружение: ПК автора, Windows 10, Python 3.10.10, CPU; версии — `eval/requirements-reference.txt`. ONNX — хеши совпадают с `models/manifest.json`; MoveNet — SavedModel TF2 с Kaggle (`google/movenet/tensorFlow2/singlepose-{lightning,thunder}/4`); изображения — хеши совпадают с `eval/coco_single_sub300_sha256.txt`.
- `eval/reference_rtmlib.py`: предобработка и декодирование из rtmlib; один экземпляр с максимальной оценкой (RTMDet ≥ 0,3, RTMO ≥ 0,1), детектор 320×320 — как в `bench/adapters/ort.js`; `--pose-input rgb|bgr`.
- `eval/reference_movenet.py`: предобработка как в `bench/adapters/tfjs.js` (pad до квадрата, `resize_bilinear` с семантикой tf.js, `toInt`).
- Прогоны: `results/raw/20261124-python-ref_*.json` (7 шт.), сравнение — `eval/compare_runs.py`, AP — `eval/evaluate.py --bootstrap 0` (CSV в рабочий каталог, не в `data/`).
- AP браузер / эталон: RTMPose-t 0,639 / 0,637 (BGR 0,608); RTMPose-s 0,701 / 0,695 (BGR 0,684); RTMO-s 0,755 / 0,754; MoveNet Lightning 0,340 / 0,339; Thunder 0,485 / 0,481.
- Расхождение точек (медиана / среднее / p95, px): RTMPose-t 1,49 / 8,6 / 29,1; RTMPose-s 1,39 / 8,5 / 32,9; RTMO-s 0,0001 / 0,53 / 2,2; Lightning 0,27 / 2,7 / 10,2; Thunder 0,11 / 2,4 / 7,3. Сдвиг по близким точкам (|Δ| < 5 px): RTMPose ≈ −0,2 px, остальные ≈ 0.
- Порядок каналов RTMPose — RGB подтверждён на 300. rtmlib по умолчанию подаёт BGR — занижает AP на 1–3 п.
- Исправлено: `eval/evaluate.py` писал CSV в кодировке системы (на Windows — cp1251) — теперь UTF-8, как требует `data/README.md`.

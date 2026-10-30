# Состояние — 30.10.2026

Ветка работы: `claude/wizardly-thompson-g1vm38` (ветка `ccr-00c41dd7-kbg0lj` уже влита в `main` через PR #1).

## Сделано — этап 0 (прототип), кроме замеров на телефонах
- **Доступ к доменам** (проверка 30.10.2026, `curl` через прокси среды):

  | Домен | Результат | Примечание |
  |---|---|---|
  | `images.cocodataset.org` | **открыт по HTTP** | по HTTPS — ошибка сертификата (сертификат S3 не покрывает имя бакета); качаем по HTTP, фиксируем SHA-256 |
  | `storage.googleapis.com` | открыт | модели MediaPipe `.task` (200) |
  | `tfhub.dev` | открыт | редирект на GCS, `model.json` MoveNet → 200 |
  | `www.kaggle.com` | открыт | API загрузки моделей → 200 |
  | `github.com` | **частично** (повторно 30.10.2026 после открытия) | загрузки релизов (`/releases/download/...`) → 200; HTML-страницы чужих репозиториев → 403 (вероятно, ограничение GitHub-доступа сессии) |
  | `raw.githubusercontent.com` | открыт | 200 (не был в списке) |
  | `objects.githubusercontent.com` | открыт (TLS) | релизы не проверены — ссылки на них выдаёт github.com |
  | `download.openmmlab.com` | открыт (после открытия автором) | ONNX RTMPose-t → 200 |
  | `huggingface.co` | открыт (после открытия автором) | API → 200 |
  | `cdn.jsdelivr.net` | открыт | 200 |
  | `registry.npmjs.org`, `pypi.org` | открыты | |

  Окружение: Node 22.22, Python 3.13, Playwright 1.63 + Chromium 153 (предустановлены).
- **COCO-SinglePerson** (`docs/datasets.md`, `eval/select_coco_single.py`): правило R2.1 даёт **921** изображение (выбрано автором для статьи); без условия `num_keypoints ≥ 1` — 1045 (= число у RTMPose). Прототип — подвыборка 300, seed 20261030. Все 921 изображение скачаны, хеши — `eval/coco_single_921_sha256.txt`.
- **Стенд** (`bench/`): страница + адаптер BlazePose (`@mediapipe/tasks-vision` 1.1.1, IMAGE), раннер Playwright, бэкенды CPU/WASM и GPU/WebGL (SwiftShader). Прогоны — `results/raw/`, журнал — `logs/runs.md`.
- **Метрики** (`eval/`, pycocotools 2.0.11, 6 юнит-тестов): OKS-AP/AP50/AP75/AR, PCK@0,2, KMR_j(τ), FMR, p50/p95, холодный старт; бутстреп-ДИ для AP.
- **Результаты** (`env=desktop-headless`, 300 изобр., `docs/prototype_report.md`, `data/*_prototype_desktop-headless.csv`):

  | | CPU (WASM) | GPU (WebGL, SwiftShader) |
  |---|---|---|
  | OKS-AP (95 % ДИ) | 0,311 (0,271–0,355) | 0,306 (0,266–0,348) |
  | AP50 / AP75 / AR | 0,566 / 0,293 / 0,360 | 0,543 / 0,305 / 0,355 |
  | PCK@0,2 (пропуск = ошибка / только найденные) | 0,669 / 0,785 | 0,647 / 0,793 |
  | FMR | 26,7 % | 29,7 % |
  | KMR(0,5) / площадь под KMR(τ) | 40,1 % / 0,408 | 42,5 % / 0,430 |
  | Задержка p50 / p95 (ПК в облаке, не телефон) | 65,0 / 90,5 мс | 601,0 / 698,4 мс |
  | Холодный старт: загрузка / иниц. / 1-й инференс | 47 / 362 / 370 мс | 41 / 370 / 4384 мс |

  AP сопоставим с опубликованным для BlazePose Lite на COCO-SinglePerson (29,3 — RTMPose, Табл. 5). Порядок и стороны точек по Таблице 4 подтверждены (с переставленными сторонами AP = 0,002).
- **Страница для телефона**: `bench/phone/dist/phone.html` (20 встроенных изображений, 50 + 200 замеров, «Скачать CSV/JSON»), инструкция — `docs/phone_quickstart.md`, конвертер — `eval/phone_to_perf.py`. Проверена в эмуляции Pixel 7 в headless Chromium (не замер).

- **Первый замер на телефоне** (POCO X8 Pro, Яндекс Браузер, по скриншотам — CSV не скачался; `data/perf_phone_prototype.csv`, отчёт — разд. 5):

  | | CPU (WASM) | GPU (WebGL) |
  |---|---|---|
  | Задержка p50 / p95 | 94,6 / 118,6 мс | 84,6 / 106,9 мс |
  | Загрузка модели (4G) / иниц. / 1-й инференс | 66 880 / 444 / 209 мс | 7 835 / 299 / 1 943 мс |

  ≈ 11–12 кадр/с на инференс в IMAGE-режиме; холодный старт упирается в сеть — нужен кеш модели.

## Этап 1 (30.10.2026) — стенд для нескольких моделей: сделан, кроме части сверки с эталоном
- 10 моделей (BlazePose Lite/Full/Heavy, MoveNet Lightning/Thunder, RTMPose-t/-s + RTMDet-nano, RTMO-s, YOLOv8n/11n-pose) работают в браузере: MediaPipe Tasks, TF.js 4.22.0, onnxruntime-web 1.30.0 (`bench/adapters/`, `models/manifest.json`, `docs/models.md`).
- Матрица на 300 изобр. (WASM — все; WebGL — BlazePose, MoveNet), 17 прогонов → `data/*_stage1_desktop-headless.csv`, отчёт `docs/stage1_report.md`.
- AP (WASM): RTMO-s 0,755; RTMPose-s 0,701; RTMPose-t 0,639; YOLOv8n 0,587; YOLO11n 0,550; MoveNet Thunder 0,485 (2 прохода 0,645); BlazePose Heavy 0,433; Full 0,375; MoveNet Lightning 0,340 (2 прохода 0,526); BlazePose Lite 0,311.
- Сверка с эталоном: BlazePose (MediaPipe Python) и YOLOv8n (ultralytics) — сделана; MoveNet, RTMPose — TODO.
- WebGPU в облаке (SwiftShader) — непригоден для замеров (секунды на кадр), мерить на телефоне.

## Решения автора (30.10.2026)
- Репозиторий сделан публичным → страница для телефона через GitHub Pages.
- Выборка для статьи — 921.
- Домены для этапа 1 открыты.
- RTMPose/RTMO — официальные ONNX с весами body7; в статье оговорить отличие от COCO-only моделей из публикации RTMPose.
- Полную выборку пока не прогоняем: сначала примерные числа с телефонов (прогон 921 был начат и прерван, см. `logs/runs.md`).

## Следующие шаги
1. **Автор:** протокол MoveNet в статье — один проход (как видеорежим библиотеки) или второй проход по кропу (как в публикации RTMPose); YOLO/RTMO — добавлять ли вход 320.
2. Сверка с эталоном для MoveNet (TF Python) и RTMPose/RTMO (rtmlib) — закрыть R1.5; проверить RGB/BGR у RTMPose на 300 изобр.
3. Страница телефона для всех моделей (этап 4): выбор модели и бэкенда, кеш моделей, WebGPU; заодно исправить скачивание CSV в Яндекс Браузере (если автор захочет).
4. Этап 2: прогон всей выборки 921 для всех моделей (в облаке на WASM ≈ 1,5–2 ч последовательно).
5. Статья: закрыть `TODO` о порядке выхода MediaPipe (Таблица 4); оговорить веса body7; KMR среди найденных; пороги детекции.
6. Ответы автора на вопросы из `PLAN.md` (блокируют этапы 3–4).

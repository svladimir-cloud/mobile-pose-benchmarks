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
  | `github.com` | **закрыт** | 403 от прокси |
  | `raw.githubusercontent.com` | открыт | 200 (не был в списке) |
  | `objects.githubusercontent.com` | открыт (TLS) | релизы не проверены — ссылки на них выдаёт github.com |
  | `download.openmmlab.com` | **закрыт** | 403 CONNECT |
  | `huggingface.co` | **закрыт** | 403 CONNECT |
  | `cdn.jsdelivr.net` | открыт | 200 |
  | `registry.npmjs.org`, `pypi.org` | открыты | |

  Окружение: Node 22.22, Python 3.13, Playwright 1.63 + Chromium 153 (предустановлены).
- **COCO-SinglePerson** (`docs/datasets.md`, `eval/select_coco_single.py`): правило R2.1 даёт **921** изображение; без условия `num_keypoints ≥ 1` — 1045 (= число у RTMPose). Подвыборка 300, seed 20261030.
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

## Следующие шаги
1. **Автор:** замеры на 1–2 телефонах по `docs/phone_quickstart.md` (CPU и GPU, перезагрузка страницы перед каждым запуском) → CSV в чат или в `results/raw/phone/`. Это последнее условие готовности этапа 0.
2. **Автор:** выбрать выборку для статьи — 921 (правило R2.1) или 1045 (как у RTMPose).
3. Прогон на полной COCO-SinglePerson (921 и 1045) → финальные CSV этапа 0/2.
4. Статья: закрыть `TODO` о порядке выхода MediaPipe в примечании к Таблице 4 (проверено); добавить в протокол KMR среди кадров с найденным человеком (на COCO τ* по 10 % не определён, т. к. FMR > 10 %); зафиксировать пороги детекции MediaPipe.
5. **Автор:** для этапа 1 открыть `github.com`, `download.openmmlab.com`, `huggingface.co` (RTMPose/RTMO/YOLO).
6. Ответы автора на вопросы из `PLAN.md` (блокируют этапы 3–4).

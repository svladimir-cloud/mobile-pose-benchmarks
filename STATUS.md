# Состояние — 30.10.2026

## Сделано
- Репозиторий заведён: требования (`REQUIREMENTS.md`), план (`PLAN.md`), правила агентов (`AGENTS.md`), схемы CSV (`data/README.md`).
- Проверка доступности из облачной среды (30.10.2026):
  - открыто: `storage.googleapis.com` (модели MediaPipe `.task` скачиваются), `registry.npmjs.org`, `pypi.org`; есть Node 22, Python 3.13, Playwright + Chromium;
  - **закрыто**: `images.cocodataset.org` (COCO), `tfhub.dev`, `www.kaggle.com` (MoveNet), `github.com` (403), `download.openmmlab.com`, `huggingface.co`, `cdn.jsdelivr.net`.

## Следующие шаги
1. **Автор:** добавить закрытые домены в Allowed domains среды (список — `REQUIREMENTS.md`, разд. 6). Минимум для прототипа — `images.cocodataset.org`.
2. Этап 0 (прототип): BlazePose Lite в headless Chromium на 300 изображениях COCO-SinglePerson → первые OKS-AP, PCK@0,2, KMR(τ), FMR, задержка (см. `PLAN.md`).
3. Страница для телефона (этап 0, шаг 4) → первые замеры автора на своих телефонах.
4. Ответы автора на вопросы из `PLAN.md` (блокируют этапы 3–4).

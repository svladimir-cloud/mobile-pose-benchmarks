# Состояние — 30.10.2026

## Сделано
- Репозиторий заведён: требования (`REQUIREMENTS.md`), план (`PLAN.md`), правила агентов (`AGENTS.md`), схемы CSV (`data/README.md`).
- Проверка доступности доменов из `REQUIREMENTS.md`, разд. 6 (повторная, 30.10.2026, `curl` через прокси среды):

  | Домен | Результат | Примечание |
  |---|---|---|
  | `images.cocodataset.org` | **открыт по HTTP** | по HTTPS — ошибка сертификата (сертификат S3 не покрывает имя бакета); `http://images.cocodataset.org/...` → 200, скачиваем по HTTP и фиксируем SHA-256 |
  | `storage.googleapis.com` | открыт | модели MediaPipe `.task` скачиваются (200) |
  | `tfhub.dev` | открыт | редирект на `storage.googleapis.com/kagglesdsdata/...`, `model.json` MoveNet → 200 |
  | `www.kaggle.com` | открыт | API загрузки моделей → 200 (редирект на GCS) |
  | `github.com` | **закрыт** | 403 от прокси |
  | `raw.githubusercontent.com` | открыт | 200 (не был в списке) |
  | `objects.githubusercontent.com` | открыт (TLS) | корень → 404, релизы не проверены, т. к. ссылки на них выдаёт github.com |
  | `download.openmmlab.com` | **закрыт** | 403 CONNECT |
  | `huggingface.co` | **закрыт** | 403 CONNECT |
  | `cdn.jsdelivr.net` | открыт | `@mediapipe/tasks-vision/package.json` → 200 |
  | `registry.npmjs.org`, `pypi.org` | открыты | |

  Окружение: Node 22.22, Python 3.13, Playwright + Chromium предустановлены.

## Следующие шаги
1. **Автор:** для этапа 1 открыть `github.com`, `download.openmmlab.com`, `huggingface.co` (RTMPose/RTMO/YOLO). Для этапа 0 доступа достаточно.
2. Этап 0 (прототип): BlazePose Lite в headless Chromium на 300 изображениях COCO-SinglePerson → первые OKS-AP, PCK@0,2, KMR(τ), FMR, задержка (см. `PLAN.md`).
3. Страница для телефона (этап 0, шаг 4) → первые замеры автора на своих телефонах.
4. Ответы автора на вопросы из `PLAN.md` (блокируют этапы 3–4).

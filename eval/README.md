# eval/ — метрики и отбор данных

Python 3.13. Зависимости: `pip install -r eval/requirements.txt`. Все команды — из корня репозитория. Формат входных результатов — `docs/result_format.md`, схемы CSV — `data/README.md`. При чтении скачанных данных запускайте `python -I`.

Тесты: `python -m pytest eval/tests -q`

| Скрипт | Назначение | Пример |
|---|---|---|
| `coco_select.py` | отбор COCO-SinglePerson (921 изображение; `--min-keypoints 0` даёт 1045), `--sample N --seed S` | `python eval/coco_select.py` |
| `coco_download.py` | скачать изображения списка, sha256 | `python eval/coco_download.py` |
| `coco_eval.py` | AP/AP50/AP75/AR, PCK@0.2, ошибки, FMR, KMR(τ), бутстреп-ДИ по result-json (mode=images) | `python eval/coco_eval.py results/raw/X.json --boot 200 --out-accuracy data/accuracy_stage0.csv --out-missing data/missing_coco_stage0.csv` |
| `exercise.py` | углы, повторы, амплитуда, CKMR, разрывы, лево/право: прогон против эталона (frames) | `python eval/exercise.py run.json ref.json --exercise squat --json-out ex.json --out-missing data/missing_x.csv` |
| `fmin.py` | f_min по прореживанию 5…30 кадр/с | `python eval/fmin.py ref.json --exercise squat --amp-thr 5 --out fmin.csv` |
| `jitter.py` | J_acc до/после 1€, задержка фильтра | `python eval/jitter.py run.json --ref ref.json --out data/jitter_x.csv` (`--static T0 T1`) |
| `aggregate.py` | perf_*.csv и Markdown пригодности по (устройство, модель) | `python eval/aggregate.py --out-csv data/perf_x.csv --out-md table.md --fmin 10 --exercise-json ex.json` |

Для финальных ДИ используйте `--boot 2000`. Вывод без `--out*` — в stdout. Принятые решения (пороги, нормировки, допуски) — в `research/progress/eval.md`, раздел «Решения».

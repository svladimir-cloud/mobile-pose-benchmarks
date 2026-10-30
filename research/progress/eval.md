# eval — прогресс

## 30.10.2026
- COCO val2017: аннотации скачаны по HTTP (HTTPS — ошибка сертификата), хеши — `docs/datasets.md`.
- `eval/select_coco_single.py`: COCO-SinglePerson = 921 изображение (правило R2.1); без условия `num_keypoints ≥ 1` — 1045 (совпадает с RTMPose). Подвыборка 300, seed 20261030.
- pycocotools 2.0.11, numpy 2.5.3, matplotlib 3.11.2 (Python 3.13).
- `eval/metrics.py`: OKS-AP/AP50/AP75/AR (pycocotools, без flip test; оценка экземпляра = среднее visibility 17 точек), PCK@0,2 (L — бо́льшая диагональ торса по GT; изображения без обеих диагоналей исключаются), норм. ошибка медиана/p95, KMR_j(τ) τ=0,1…0,9 + площадь, FMR, p50/p95. Тесты: `python3 -m pytest eval/tests -q` (6 passed).
- `eval/evaluate.py`: сырые прогоны → `data/{accuracy,missing,perf}_prototype_desktop-headless.csv`, `docs/img/` (кривая KMR, 5 худших). Бутстреп AP — 2000 выборок изображений (seed 0).
- Порядок/стороны (R1.4) подтверждены: AP 0,311 (Таблица 4) против 0,002 (стороны переставлены).

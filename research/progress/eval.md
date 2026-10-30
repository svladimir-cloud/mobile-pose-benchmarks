# eval — прогресс

## 30.10.2026
- COCO val2017: аннотации скачаны по HTTP (HTTPS — ошибка сертификата), хеши — `docs/datasets.md`.
- `eval/select_coco_single.py`: COCO-SinglePerson = 921 изображение (правило R2.1); без условия `num_keypoints ≥ 1` — 1045 (совпадает с RTMPose). Подвыборка 300, seed 20261030.
- pycocotools 2.0.11, numpy 2.5.3, matplotlib 3.11.2 (Python 3.13).

# Наборы данных

## COCO val2017 — исходник

| Файл | Источник | SHA-256 |
|---|---|---|
| `annotations_trainval2017.zip` (252 907 541 Б) | `http://images.cocodataset.org/annotations/annotations_trainval2017.zip` (загружено 30.10.2026) | `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268` |
| `annotations/person_keypoints_val2017.json` | из архива выше | `788e2dae83c86bd547be7fab269d6399df5671063d29a61360cdb2cc370d2b14` |
| изображения `val2017/<id:012d>.jpg` | `http://images.cocodataset.org/val2017/` | по файлу — `eval/coco_single_sub300_sha256.txt` |

Загрузка идёт по HTTP: по HTTPS сертификат S3 не покрывает имя `images.cocodataset.org`. Целостность фиксируется хешами. Данные лежат в `datasets/` и в git не попадают.

## COCO-SinglePerson (R2.1)

**Правило отбора** (`eval/select_coco_single.py`): изображение val2017 входит в выборку, если среди аннотаций `person_keypoints_val2017.json` у него **ровно одна** аннотация категории `person`, и у неё `iscrowd = 0` и `num_keypoints ≥ 1`.

| Величина | Значение |
|---|---|
| изображений в val2017 | 5000 |
| изображений хотя бы с одной аннотацией person | 2693 |
| **COCO-SinglePerson (наше правило)** | **921** |
| то же без условия `num_keypoints ≥ 1` (ровно одна аннотация person) | 1045 |

**Расхождение с RTMPose (1045).** Число 1045 воспроизводится правилом «ровно одна аннотация person» без требования размеченных точек. Разница — 124 изображения, где единственный человек не имеет ни одной размеченной точки (обычно мелкий или сильно перекрытый). Для pycocotools такие аннотации помечаются как игнорируемые, и предсказания на этих изображениях могут дать лишь ложные срабатывания. По R2.1 используем 921; чтобы сравнить с RTMPose напрямую, можно пересчитать на 1045 (ключ отбора менять в скрипте). **Решение автора (30.10.2026): в статье используется 921.** Списки: `eval/ids_coco_single_921.txt`, хеши изображений — `eval/coco_single_921_sha256.txt`.

## Подвыборка прототипа (этап 0)

- 300 изображений из 921, `random.Random(20261030).sample(ids, 300)` по отсортированному списку `image_id`, затем сортировка.
- Список: `eval/ids_sub300_seed20261030.txt` (SHA-256 списка `f68eaf959777030f36b7694e0a839316ed5d5cbbc7d287733ee844b4664f6865`).
- Хеши изображений: `eval/coco_single_sub300_sha256.txt`.

Воспроизведение:
```bash
python3 -I eval/select_coco_single.py --ann datasets/coco/raw/annotations/person_keypoints_val2017.json --out datasets/coco/single --subset 300 --seed 20261030
```
Скрипт пишет `datasets/coco/single/gt_all.json` (921), `gt_sub300_seed20261030.json` (300), списки id и `selection_stats.json`.

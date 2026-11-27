# Наборы данных

## COCO-SinglePerson (изображения)

Источник: COCO 2017, `annotations_trainval2017.zip` (`annotations/person_keypoints_val2017.json`), изображения val2017 с `http://images.cocodataset.org/val2017/`. Разметка — по лицензии COCO (CC BY 4.0), лицензии отдельных изображений — поле `licenses` в аннотациях. Изображения и разметка в репозиторий не коммитятся (`datasets/`).

### Правило отбора (`eval/coco_select.py`)

Изображение val2017 входит в набор, если:
1. у него **ровно одна** аннотация в `person_keypoints_val2017.json` (все аннотации там — person; crowd-аннотации тоже считаются, то есть человек + crowd исключаются);
2. эта аннотация имеет `iscrowd = 0`;
3. `num_keypoints >= 1` (размечена хотя бы одна точка).

### Размер набора

| Условие | Изображений |
|---|---|
| val2017 всего | 5000 |
| из них с хотя бы одной аннотацией person | 2693 |
| ровно одна аннотация, `iscrowd = 0` (без условия на точки) | **1045** |
| то же и `num_keypoints >= 1` (**наш набор**) | **921** |

У авторов RTMPose в наборе COCO-SinglePerson 1045 изображений — это в точности число изображений с одной аннотацией без условия на размеченные точки. Наше число меньше на 124: это изображения, у которых единственный человек не имеет ни одной размеченной точки (`num_keypoints = 0`); для оценки точек они бесполезны (COCOeval их игнорирует). Вариант авторов воспроизводится командой `python eval/coco_select.py --min-keypoints 0` (1045). В наборе 921 изображений: медиана размеченных точек — 13, все 17 размечены у 82 изображений.

### Файлы и команды

```
python eval/coco_select.py                      # datasets/coco/single_person_gt.json, single_person_list.json
python eval/coco_select.py --sample 300 --seed 0 --out-gt datasets/coco/sp300_gt.json --out-list datasets/coco/sp300_list.json
python eval/coco_download.py                    # datasets/coco/val2017/*.jpg + val2017_sha256.txt
```

- `single_person_gt.json` — валидный COCO-json (images, annotations, categories) только с отобранными изображениями; подаётся в pycocotools.
- `single_person_list.json` — список `{image_id, file_name, width, height}`.
- `val2017_sha256.txt` — SHA-256 каждого скачанного изображения (формат `sha256sum`).
- Подвыборка: `--sample N --seed S` (фиксированная, `random.Random(S)`, порядок по `image_id`).
- Скачано полностью: 921 изображение, 135 МБ, ошибок загрузки 0 (27.11.2026).

### Метрики на этом наборе
`eval/coco_eval.py` (OKS-AP/AP50/AP75/AR, PCK@0,2, нормированная ошибка по суставам, FMR, KMR(τ), бутстреп-ДИ). Подробности — `eval/README.md` и `research/progress/eval.md`.

## Собственные видео
TODO: описание появится после съёмки (см. `PLAN.md`).

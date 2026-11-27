#!/usr/bin/env python3
"""Отбор COCO-SinglePerson из person_keypoints_val2017.

Правило: изображение val2017 берётся, если у него ровно одна аннотация person
(в файле person_keypoints все аннотации — person), она с iscrowd=0 и
num_keypoints >= 1. Любые другие аннотации (в т.ч. crowd) исключают изображение.
"""
import argparse
import json
import os
import random
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MEMBER = "annotations/person_keypoints_val2017.json"


def select_single_person(coco, min_keypoints=1):
    """Чистая функция. coco — dict в формате COCO keypoints.
    Возвращает (список {image_id,file_name,width,height}, {image_id: аннотация})."""
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    imgs = {im["id"]: im for im in coco["images"]}
    out, anns = [], {}
    for iid in sorted(by_img):
        al = by_img[iid]
        if len(al) != 1:
            continue
        a = al[0]
        if a.get("iscrowd", 0) != 0 or a.get("num_keypoints", 0) < min_keypoints:
            continue
        if iid not in imgs:
            continue
        im = imgs[iid]
        out.append({"image_id": iid, "file_name": im["file_name"],
                    "width": im["width"], "height": im["height"]})
        anns[iid] = a
    return out, anns


def subsample(items, n, seed):
    """Фиксированная подвыборка n элементов (по seed), порядок — по image_id."""
    if n is None or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(items)), n))
    return [items[i] for i in idx]


def build_gt(coco, items, anns):
    """Валидный COCO-json только с выбранными изображениями и аннотациями."""
    ids = {it["image_id"] for it in items}
    return {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": [im for im in coco["images"] if im["id"] in ids],
        "annotations": [anns[i] for i in sorted(ids)],
        "categories": coco["categories"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", default="datasets/coco/annotations_trainval2017.zip")
    ap.add_argument("--out-gt", default="datasets/coco/single_person_gt.json")
    ap.add_argument("--out-list", default="datasets/coco/single_person_list.json")
    ap.add_argument("--sample", type=int, default=None, help="подвыборка N изображений")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-keypoints", type=int, default=1,
                    help="минимум размеченных точек (по умолчанию 1; 0 даёт 1045 изображений, как у авторов RTMPose)")
    a = ap.parse_args(argv)
    with zipfile.ZipFile(a.zip) as z:
        with z.open(MEMBER) as f:
            coco = json.load(f)
    items, anns = select_single_person(coco, a.min_keypoints)
    total = len(items)
    items = subsample(items, a.sample, a.seed)
    gt = build_gt(coco, items, anns)
    for p in (a.out_gt, a.out_list):
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(a.out_gt, "w") as f:
        json.dump(gt, f)
    with open(a.out_list, "w") as f:
        json.dump(items, f, indent=0)
    print(f"val2017: {len(coco['images'])} изображений; COCO-SinglePerson (полный): {total}; "
          f"записано: {len(items)} (sample={a.sample}, seed={a.seed})")


if __name__ == "__main__":
    main()

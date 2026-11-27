"""Отбор COCO-SinglePerson из val2017 (REQUIREMENTS.md R2.1, docs/datasets.md).

Правило: изображение входит в выборку, если в нём ровно один аннотированный
экземпляр категории person, этот экземпляр iscrowd=0 и num_keypoints >= 1,
и нет ни одной аннотации person с iscrowd=1.

Запуск (из корня репозитория):
  python3 -I eval/select_coco_single.py --ann datasets/coco/raw/annotations/person_keypoints_val2017.json \
      --out datasets/coco/single --subset 300 --seed 20261030
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def select(gt):
    by_img = defaultdict(list)
    for a in gt["annotations"]:
        by_img[a["image_id"]].append(a)
    ids = []
    for img in gt["images"]:
        anns = by_img.get(img["id"], [])
        if len(anns) == 1 and anns[0]["iscrowd"] == 0 and anns[0]["num_keypoints"] >= 1:
            ids.append(img["id"])
    return sorted(ids)


def subset_gt(gt, ids):
    ids = set(ids)
    return {
        "info": gt.get("info", {}),
        "licenses": gt.get("licenses", []),
        "categories": gt["categories"],
        "images": [i for i in gt["images"] if i["id"] in ids],
        "annotations": [a for a in gt["annotations"] if a["image_id"] in ids],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261030)
    args = ap.parse_args()

    gt = json.loads(Path(args.ann).read_text())
    ids = select(gt)
    rng = random.Random(args.seed)
    sub = sorted(rng.sample(ids, min(args.subset, len(ids))))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, chosen in (("all", ids), (f"sub{args.subset}_seed{args.seed}", sub)):
        g = subset_gt(gt, chosen)
        (out / f"gt_{name}.json").write_text(json.dumps(g))
        (out / f"ids_{name}.txt").write_text("\n".join(str(i) for i in chosen) + "\n")
    stats = {
        "source_sha256": hashlib.sha256(Path(args.ann).read_bytes()).hexdigest(),
        "val2017_images": len(gt["images"]),
        "images_with_person": len({a["image_id"] for a in gt["annotations"]}),
        "coco_single": len(ids),
        "subset": len(sub),
        "seed": args.seed,
        "subset_ids_sha256": hashlib.sha256(("\n".join(map(str, sub)) + "\n").encode()).hexdigest(),
    }
    (out / "selection_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

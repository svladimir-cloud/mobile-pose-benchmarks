"""Метрики прототипа (раздел 3.4 статьи; REQUIREMENTS.md R3.1, R3.2, R3.4).

Вход — сырой JSON прогона из results/raw/ (bench/runner.mjs) и GT в формате COCO.
Предсказание на изображение — не более одного человека (keypoints17: [[x, y, score]] или null).
"""
import contextlib
import io
import json

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

COCO_KP = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
           "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
           "left_knee", "right_knee", "left_ankle", "right_ankle"]
# Константы k_i из pycocotools (COCOeval.Params.kpt_oks_sigmas).
SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
# Пары лево/право в нумерации COCO.
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
TAUS = [round(0.1 * i, 1) for i in range(1, 10)]


def instance_score(kp17, mode="mean_score"):
    """Оценка экземпляра для ранжирования в AP. У BlazePose нет оценки человека,
    поэтому берётся среднее score (visibility) по 17 точкам."""
    s = np.asarray(kp17, float)[:, 2]
    if mode == "mean_score":
        return float(np.mean(s))
    if mode == "one":
        return 1.0
    raise ValueError(mode)


def to_coco_results(preds, score_mode="mean_score", swap_lr=False):
    res = []
    for p in preds:
        if p["keypoints17"] is None:
            continue
        kp = np.asarray(p["keypoints17"], float)
        if swap_lr:
            kp = kp[FLIP]
        flat = np.column_stack([kp[:, 0], kp[:, 1], np.ones(17)]).ravel().tolist()
        # Оценку экземпляра даёт адаптер (instance_score: детектор / среднее по точкам), иначе — среднее score точек.
        sc = p.get("instance_score") if score_mode == "mean_score" else None
        res.append({"image_id": p["image_id"], "category_id": 1, "keypoints": flat,
                    "score": float(sc) if sc is not None else instance_score(kp, score_mode)})
    return res


def oks_ap(gt_file, results, img_ids):
    """OKS-AP/AP50/AP75/AR (pycocotools, без flip test). Возвращает словарь и COCOeval."""
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(gt_file)
        if not results:
            return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR": 0.0}, None
        dt = gt.loadRes(results)
        ev = COCOeval(gt, dt, "keypoints")
        ev.params.imgIds = sorted(img_ids)
        ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    return {"AP": s[0], "AP50": s[1], "AP75": s[2], "AR": s[5]}, ev


def oks_single(gt_kp, gt_area, pred_xy):
    """OKS одного предсказания относительно одной разметки (формула pycocotools)."""
    g = np.asarray(gt_kp, float).reshape(17, 3)
    vis = g[:, 2] > 0
    if not vis.any():
        return np.nan
    d2 = ((pred_xy - g[:, :2]) ** 2).sum(1)
    e = d2 / ((2 * SIGMAS) ** 2) / (gt_area + np.spacing(1)) / 2
    return float(np.exp(-e)[vis].mean())


def torso_diameter(g):
    """Нормирующий размер L: наибольшая из диагоналей торса (плечо — противоположное бедро) по GT.
    None, если ни одна диагональ не размечена полностью."""
    ds = []
    for a, b in ((5, 12), (6, 11)):
        if g[a, 2] > 0 and g[b, 2] > 0:
            ds.append(float(np.hypot(*(g[a, :2] - g[b, :2]))))
    return max(ds) if ds else None


def per_image(preds, gt_json):
    """Покадровые величины: OKS, OKS с переставленными сторонами, ошибки по суставам / L."""
    anns = {a["image_id"]: a for a in gt_json["annotations"]}
    rows = []
    for p in preds:
        a = anns[p["image_id"]]
        g = np.asarray(a["keypoints"], float).reshape(17, 3)
        L = torso_diameter(g)
        row = {"image_id": p["image_id"], "detected": p["keypoints17"] is not None,
               "num_gt_kp": int((g[:, 2] > 0).sum()), "L": L, "oks": 0.0, "oks_swapped": 0.0,
               "err_norm": [None] * 17}
        if row["detected"]:
            kp = np.asarray(p["keypoints17"], float)
            row["oks"] = oks_single(a["keypoints"], a["area"], kp[:, :2])
            row["oks_swapped"] = oks_single(a["keypoints"], a["area"], kp[FLIP, :2])
            if L:
                err = np.hypot(*(kp[:, :2] - g[:, :2]).T) / L
                row["err_norm"] = [float(e) if g[j, 2] > 0 else None for j, e in enumerate(err)]
        rows.append(row)
    return rows


def pck(rows, alpha=0.2):
    """PCK@alpha по размеченным точкам изображений, где определён L.
    Возвращает (PCK с пропуском как ошибкой, PCK только по найденным, число точек, число изображений без L)."""
    hit = tot = hit_found = tot_found = 0
    no_l = 0
    for r in rows:
        if r["L"] is None:
            no_l += 1
            continue
        n = r["num_gt_kp"]
        tot += n
        if r["detected"]:
            e = [x for x in r["err_norm"] if x is not None]
            h = sum(x <= alpha for x in e)
            hit += h; hit_found += h; tot_found += len(e)
    return hit / tot if tot else np.nan, hit_found / tot_found if tot_found else np.nan, tot, no_l


def err_stats(rows):
    e = [x for r in rows if r["detected"] for x in r["err_norm"] if x is not None]
    if not e:
        return np.nan, np.nan
    return float(np.median(e)), float(np.percentile(e, 95))


def missing(preds, taus=TAUS):
    """KMR_j(tau) и FMR (раздел 3.4). Точка потеряна, если нет человека, score < tau или точка вне кадра."""
    T = len(preds)
    no_person = np.array([p["keypoints17"] is None for p in preds])
    score = np.zeros((T, 17)); oof = np.zeros((T, 17), bool)
    for t, p in enumerate(preds):
        if p["keypoints17"] is None:
            continue
        kp = np.asarray(p["keypoints17"], float)
        score[t] = kp[:, 2]
        oof[t] = (kp[:, 0] < 0) | (kp[:, 0] >= p["width"]) | (kp[:, 1] < 0) | (kp[:, 1] >= p["height"])
    out = {"frames_total": T, "frames_no_person": int(no_person.sum()), "FMR": float(no_person.mean()),
           "out_of_frame": oof.sum(0).tolist(), "kmr": {}, "frames_missing": {}}
    for tau in taus:
        m = no_person[:, None] | (score < tau) | oof
        out["frames_missing"][tau] = m.sum(0).tolist()
        out["kmr"][tau] = m.mean(0).tolist()
    return out


def kmr_auc(kmr_mean_by_tau):
    """Площадь под кривой KMR(tau) по сетке 0,1…0,9 (трапеции), нормированная на длину отрезка (0,8)."""
    t = np.array(sorted(kmr_mean_by_tau)); y = np.array([kmr_mean_by_tau[x] for x in t])
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(t)) / (t[-1] - t[0]))


def latency_stats(lat):
    a = np.asarray(lat, float)
    return float(np.percentile(a, 50)), float(np.percentile(a, 95))

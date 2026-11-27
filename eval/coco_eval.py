#!/usr/bin/env python3
"""Метрики точности и полноты на COCO-SinglePerson по result-json (mode=images).

AP/AP50/AP75/AR — pycocotools COCOeval (keypoints, без flip test);
PCK@0.2 и нормированная ошибка — нормировка по диаметру торса GT;
FMR, KMR_j(tau), площадь под KMR(tau); бутстреп-ДИ AP по изображениям.
"""
import argparse
import contextlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import common as C  # noqa: E402

FINE_TAUS = np.round(np.arange(0.0, 1.0001, 0.01), 2)


# ---------------------------------------------------------------- чистые функции
def pred_score(p, arr):
    """Score экземпляра: поле score, а если его нет — среднее score точек."""
    if p.get("score") is not None:
        return float(p["score"])
    return float(arr[:, 2].mean())


def preds_by_image(predictions, valid_ids=None):
    """{image_id: (kps_array|None, score)}; при дубликатах побеждает наибольший score."""
    out = {}
    for p in predictions:
        iid = p["image_id"]
        if valid_ids is not None and iid not in valid_ids:
            continue
        arr = C.kps_array(p.get("kps"))
        sc = pred_score(p, arr) if arr is not None else 0.0
        if iid not in out or (arr is not None and (out[iid][0] is None or sc > out[iid][1])):
            out[iid] = (arr, sc)
    return out


def make_dt(pbi):
    """Список результатов COCO. Предсказания с kps=None не подаются."""
    dts = []
    for iid, (arr, sc) in pbi.items():
        if arr is None:
            continue
        dts.append({"image_id": iid, "category_id": 1, "score": sc,
                    "keypoints": [float(v) for v in arr.reshape(-1)]})
    return dts


def _coco_from_dict(d):
    from pycocotools.coco import COCO
    c = COCO()
    c.dataset = d
    with contextlib.redirect_stdout(io.StringIO()):
        c.createIndex()
    return c


def run_cocoeval(gt_dict, dts, img_ids):
    """Возвращает (stats[0..9], COCOeval|None). Пустой dts -> нули."""
    from pycocotools.cocoeval import COCOeval
    if not dts:
        return np.zeros(10), None
    gt = _coco_from_dict(gt_dict)
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dts)
        E = COCOeval(gt, dt, "keypoints")
        E.params.imgIds = sorted(img_ids)
        E.evaluate()
        E.accumulate()
        E.summarize()
    return np.asarray(E.stats), E


def ap_from_evalimgs(eval_imgs, iou_thrs, rec_thrs, max_det=20):
    """AP (area=all, maxDets=20, одна категория) по списку evalImg (с повторами) —
    копия логики COCOeval.accumulate, нужна для быстрого бутстрепа."""
    e = [x for x in eval_imgs if x is not None]
    if not e:
        return 0.0
    scores = np.concatenate([x["dtScores"][:max_det] for x in e])
    order = np.argsort(-scores, kind="mergesort")
    dtm = np.concatenate([x["dtMatches"][:, :max_det] for x in e], axis=1)[:, order]
    dig = np.concatenate([x["dtIgnore"][:, :max_det] for x in e], axis=1)[:, order]
    gig = np.concatenate([x["gtIgnore"] for x in e])
    npig = int(np.count_nonzero(gig == 0))
    if npig == 0:
        return 0.0
    tps = np.logical_and(dtm, np.logical_not(dig))
    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dig))
    tp_sum = np.cumsum(tps, axis=1).astype(float)
    fp_sum = np.cumsum(fps, axis=1).astype(float)
    aps = []
    for t in range(len(iou_thrs)):
        tp, fp = tp_sum[t], fp_sum[t]
        nd = len(tp)
        rc = tp / npig
        pr = (tp / (fp + tp + np.spacing(1))).tolist()
        q = np.zeros(len(rec_thrs))
        for i in range(nd - 1, 0, -1):
            if pr[i] > pr[i - 1]:
                pr[i - 1] = pr[i]
        inds = np.searchsorted(rc, rec_thrs, side="left")
        for ri, pi in enumerate(inds):
            if pi < nd:
                q[ri] = pr[pi]
        aps.append(q.mean())
    return float(np.mean(aps))


def bootstrap_ap(E, img_ids, n_boot, seed):
    """Бутстреп по изображениям: ДИ 2.5–97.5% для AP. Возвращает (low, high) или (nan, nan)."""
    if E is None or n_boot <= 0:
        return float("nan"), float("nan")
    ids = list(E.params.imgIds)
    n_img = len(ids)
    ev = E.evalImgs[:n_img]  # категория 0, areaRng 'all' (индекс 0)
    iou, rec = E.params.iouThrs, E.params.recThrs
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_img, n_img)
        vals.append(ap_from_evalimgs([ev[i] for i in pick], iou, rec))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def gt_by_image(gt_dict):
    """{image_id: (kps (17,3), width, height)} для одиночных аннотаций."""
    imgs = {im["id"]: im for im in gt_dict["images"]}
    out = {}
    for a in gt_dict["annotations"]:
        k = np.asarray(a["keypoints"], dtype=float).reshape(17, 3)
        im = imgs[a["image_id"]]
        out[a["image_id"]] = (k, im["width"], im["height"])
    return out


def pck_and_errors(gts, pbi, thr=0.2):
    """PCK@thr и нормированные ошибки.
    gts: {id: (kps,w,h)}, pbi: {id: (arr|None, score)}.
    Возвращает dict: pck, n_images, n_points, err_found (массив),
    err_all (массив, потерянные = inf), err_by_joint {j: (found, all)}."""
    correct = total = n_img = 0
    found, allv = [], []
    by_j = {j: ([], []) for j in range(17)}
    for iid, (gt, w, h) in gts.items():
        D = C.torso_diameter(gt)
        if D is None:
            continue
        n_img += 1
        arr = pbi.get(iid, (None, 0.0))[0]
        for j in range(17):
            if gt[j, 2] <= 0:
                continue
            total += 1
            if arr is None:
                allv.append(np.inf)
                by_j[j][1].append(np.inf)
                continue
            e = float(np.hypot(*(arr[j, :2] - gt[j, :2]))) / D
            found.append(e)
            allv.append(e)
            by_j[j][0].append(e)
            by_j[j][1].append(e)
            if e <= thr:
                correct += 1
    return {"pck": correct / total if total else float("nan"), "n_images": n_img,
            "n_points": total, "err_found": np.array(found), "err_all": np.array(allv),
            "err_by_joint": {j: (np.array(v[0]), np.array(v[1])) for j, v in by_j.items()}}


def kmr_counts(gts, pbi, tau):
    """Для каждого сустава: (всего размеченных, нет человека, потеряно, вне кадра)."""
    res = np.zeros((17, 4), dtype=int)
    for iid, (gt, w, h) in gts.items():
        arr = pbi.get(iid, (None, 0.0))[0]
        lab = gt[:, 2] > 0
        lost = C.lost_mask(arr, tau, w, h)
        if arr is None:
            out = np.zeros(17, dtype=bool)
        else:
            out = (arr[:, 0] < 0) | (arr[:, 0] >= w) | (arr[:, 1] < 0) | (arr[:, 1] >= h)
        res[:, 0] += lab
        res[:, 1] += lab & (arr is None)
        res[:, 2] += lab & lost
        res[:, 3] += lab & out
    return res


def kmr_curve(gts, pbi, taus=C.TAUS):
    """{tau: массив (17,4)} и KMR по суставам (17,) для каждого tau."""
    return {t: kmr_counts(gts, pbi, t) for t in taus}


def auc_trapz(taus, vals):
    """Площадь под кривой KMR(tau) по трапециям, нормированная на размах tau (= среднее значение, 0..1)."""
    taus = np.asarray(taus, float)
    vals = np.asarray(vals, float)
    if len(taus) < 2:
        return float("nan")
    return float(np.trapezoid(vals, taus) / (taus[-1] - taus[0]))


def tau_star(gts, pbi, target=0.10):
    """Наибольший tau (сетка 0.01), при котором общий KMR по всем точкам ≤ target;
    nan, если даже при tau=0 доля потерь выше target."""
    best = float("nan")
    for t in FINE_TAUS:
        c = kmr_counts(gts, pbi, float(t))
        tot = c[:, 0].sum()
        if tot and c[:, 2].sum() / tot <= target:
            best = float(t)
    return best


def evaluate_images(gt_dict, predictions, n_boot=200, seed=0, restrict_to_predicted=True):
    """Полный расчёт метрик для одного набора предсказаний."""
    gts_all = gt_by_image(gt_dict)
    pred_ids = {p["image_id"] for p in predictions} & set(gts_all)
    ids = pred_ids if restrict_to_predicted else set(gts_all)
    gts = {i: gts_all[i] for i in ids}
    sub_gt = {**gt_dict, "images": [im for im in gt_dict["images"] if im["id"] in ids],
              "annotations": [a for a in gt_dict["annotations"] if a["image_id"] in ids]}
    pbi = preds_by_image(predictions, ids)
    stats, E = run_cocoeval(sub_gt, make_dt(pbi), ids)
    lo, hi = bootstrap_ap(E, ids, n_boot, seed)
    pe = pck_and_errors(gts, pbi)
    n = len(ids)
    fmr = sum(1 for i in ids if pbi.get(i, (None,))[0] is None) / n if n else float("nan")
    curve = kmr_curve(gts, pbi)
    taus = list(curve)
    kmr_j = {}
    for j in range(17):
        v = [curve[t][j, 2] / curve[t][j, 0] if curve[t][j, 0] else float("nan") for t in taus]
        kmr_j[C.KP_NAMES[j]] = {"kmr": dict(zip(taus, v)), "auc": auc_trapz(taus, np.nan_to_num(v))
                                if not np.all(np.isnan(v)) else float("nan"),
                                "counts": {t: curve[t][j].tolist() for t in taus}}
    tot = {t: curve[t].sum(axis=0) for t in taus}
    kmr_all = [tot[t][2] / tot[t][0] if tot[t][0] else float("nan") for t in taus]
    return {
        "n_images": n, "AP": float(stats[0]), "AP50": float(stats[1]), "AP75": float(stats[2]),
        "AR": float(stats[5]), "ci_low": lo, "ci_high": hi,
        "PCK@0.2": pe["pck"], "n_pck_images": pe["n_images"], "n_pck_points": pe["n_points"],
        "err_median_norm": C.percentile(pe["err_found"], 50),
        "err_p95_norm": C.percentile(pe["err_found"], 95),
        "err_median_norm_lostinf": C.percentile(pe["err_all"], 50),
        "err_p95_norm_lostinf": C.percentile(pe["err_all"], 95),
        "err_by_joint": {C.KP_NAMES[j]: {
            "median": C.percentile(f, 50), "p95": C.percentile(f, 95),
            "median_lostinf": C.percentile(a, 50), "p95_lostinf": C.percentile(a, 95)}
            for j, (f, a) in pe["err_by_joint"].items()},
        "fmr": fmr, "kmr_by_joint": kmr_j,
        "kmr_all": {"kmr": dict(zip(taus, kmr_all)), "auc": auc_trapz(taus, np.nan_to_num(kmr_all)),
                    "counts": {t: tot[t].tolist() for t in taus}},
        "tau_star_10": tau_star(gts, pbi),
    }


# ---------------------------------------------------------------- строки CSV
def accuracy_row(meta, score_source, m, dataset="coco-single"):
    r = dict(meta)
    r.update({"dataset": dataset, "score_source": score_source})
    for k in ("n_images", "AP", "AP50", "AP75", "AR", "PCK@0.2", "err_median_norm",
              "err_p95_norm", "ci_low", "ci_high", "err_median_norm_lostinf",
              "err_p95_norm_lostinf", "fmr", "n_pck_images", "tau_star_10"):
        r[k] = m[k]
    r["kmr_auc"] = m["kmr_all"]["auc"]
    return r


def missing_rows(meta, m, dataset="coco-single"):
    rows = []
    groups = dict(m["kmr_by_joint"])
    groups["all"] = m["kmr_all"]
    for name, g in groups.items():
        for t, cnt in g["counts"].items():
            total, nop, lost, out = cnt
            r = dict(meta)
            r.update({"dataset": dataset, "keypoint": name, "tau": t, "frames_total": total,
                      "frames_no_person": nop, "frames_missing": lost, "out_of_frame": out,
                      "kmr": lost / total if total else float("nan"), "kmr_auc": g["auc"]})
            rows.append(r)
    return rows


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+", help="result-json (mode=images)")
    ap.add_argument("--gt", default="datasets/coco/single_person_gt.json")
    ap.add_argument("--boot", type=int, default=200, help="число бутстреп-выборок (финал: 2000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-gt", action="store_true",
                    help="считать по всем изображениям GT (по умолчанию — только по тем, что есть в предсказаниях)")
    ap.add_argument("--out-accuracy", default=None, help="CSV accuracy_*.csv (по умолчанию stdout)")
    ap.add_argument("--out-missing", default=None, help="CSV missing_*.csv (по умолчанию не пишется)")
    ap.add_argument("--json-out", default=None, help="полные метрики в JSON")
    a = ap.parse_args(argv)
    gt = C.load_json(a.gt)
    arows, mrows, full = [], [], {}
    for path in a.results:
        res = C.load_json(path)
        if res.get("mode") != "images" or res.get("status") != "ok" or "predictions" not in res:
            print(f"пропуск {path}: mode={res.get('mode')} status={res.get('status')}", file=sys.stderr)
            continue
        meta = C.meta_from_result(res, os.path.basename(path))
        m = evaluate_images(gt, res["predictions"], a.boot, a.seed, not a.all_gt)
        src = (res.get("model") or {}).get("score_source", "")
        arows.append(accuracy_row(meta, src, m))
        mrows += missing_rows(meta, m)
        full[os.path.basename(path)] = m
    C.write_csv(a.out_accuracy, C.ACCURACY_COLS, arows)
    if a.out_missing:
        C.write_csv(a.out_missing, C.MISSING_COLS, mrows)
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(full, f, indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())

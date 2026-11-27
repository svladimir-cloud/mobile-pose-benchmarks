import numpy as np
import pytest

import coco_eval as E
import coco_select as S
import common as C
from synth import base_pose


def make_coco(n=12, w=640, h=800, seed=0):
    rng = np.random.default_rng(seed)
    images, anns = [], []
    for i in range(1, n + 1):
        images.append({"id": i, "file_name": f"{i:012d}.jpg", "width": w, "height": h})
        p = base_pose(cx=rng.uniform(150, 450), cy=rng.uniform(250, 300))
        p[:, :2] *= rng.uniform(0.6, 1.0)
        p[:, :2] += 50
        k = np.concatenate([p[:, :2], np.full((17, 1), 2.0)], axis=1)
        xs, ys = k[:, 0], k[:, 1]
        anns.append({"id": i, "image_id": i, "category_id": 1, "iscrowd": 0, "num_keypoints": 17,
                     "keypoints": k.reshape(-1).tolist(),
                     "bbox": [xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()],
                     "area": float((xs.max() - xs.min()) * (ys.max() - ys.min()))})
    return {"images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "person", "keypoints": C.KP_NAMES, "skeleton": []}]}


def gt_as_pred(coco, score=1.0):
    return [{"image_id": a["image_id"], "score": score,
             "kps": [[a["keypoints"][3 * j], a["keypoints"][3 * j + 1], 1.0] for j in range(17)]}
            for a in coco["annotations"]]


def test_select_rules():
    ann = lambda i, iid, crowd=0, nk=5: {"id": i, "image_id": iid, "category_id": 1, "iscrowd": crowd,
                                         "num_keypoints": nk, "keypoints": [0] * 51}
    coco = {"images": [{"id": k, "file_name": f"{k}.jpg", "width": 10, "height": 10} for k in range(1, 7)],
            "annotations": [ann(1, 1),                       # ок
                            ann(2, 2), ann(3, 2),            # два человека
                            ann(4, 3, crowd=1),              # crowd
                            ann(5, 4, nk=0),                 # нет точек
                            ann(6, 5), ann(7, 5, crowd=1, nk=0)],  # человек + crowd
            "categories": [{"id": 1, "name": "person"}]}
    items, anns = S.select_single_person(coco)
    assert [i["image_id"] for i in items] == [1]
    items0, _ = S.select_single_person(coco, min_keypoints=0)
    assert [i["image_id"] for i in items0] == [1, 4]
    gt = S.build_gt(coco, items, anns)
    assert len(gt["images"]) == 1 and len(gt["annotations"]) == 1


def test_subsample_fixed():
    items = [{"image_id": i} for i in range(100)]
    a, b = S.subsample(items, 10, 5), S.subsample(items, 10, 5)
    assert a == b and len(a) == 10 and a != S.subsample(items, 10, 6)
    assert [x["image_id"] for x in a] == sorted(x["image_id"] for x in a)
    assert S.subsample(items, None, 0) == items


def test_perfect_predictions_ap_one():
    coco = make_coco()
    m = E.evaluate_images(coco, gt_as_pred(coco), n_boot=20)
    assert m["AP"] == pytest.approx(1.0) and m["AP50"] == pytest.approx(1.0)
    assert m["AR"] == pytest.approx(1.0)
    assert m["PCK@0.2"] == 1.0 and m["fmr"] == 0.0
    assert m["err_median_norm"] == 0 and m["ci_low"] == pytest.approx(1.0)
    assert m["kmr_all"]["auc"] == 0.0


def test_bootstrap_function_matches_cocoeval():
    coco = make_coco(n=15)
    preds = gt_as_pred(coco)
    rng = np.random.default_rng(1)
    for p in preds:  # шум и разные score
        p["kps"] = [[x + rng.normal(0, 8), y + rng.normal(0, 8), s] for x, y, s in p["kps"]]
        p["score"] = float(rng.uniform(0.2, 1))
    ids = {a["image_id"] for a in coco["annotations"]}
    pbi = E.preds_by_image(preds, ids)
    stats, ev = E.run_cocoeval(coco, E.make_dt(pbi), ids)
    mine = E.ap_from_evalimgs(ev.evalImgs[:15], ev.params.iouThrs, ev.params.recThrs)
    assert mine == pytest.approx(stats[0], abs=1e-9)
    lo, hi = E.bootstrap_ap(ev, ids, 30, 0)
    assert lo <= hi and 0 <= lo


def test_ap_drops_for_large_shift():
    coco = make_coco()
    preds = gt_as_pred(coco)
    for p in preds:
        p["kps"] = [[x + 200, y, s] for x, y, s in p["kps"]]
    m = E.evaluate_images(coco, preds, n_boot=0)
    assert m["AP"] < 0.05 and m["PCK@0.2"] == 0


def gt_single(pose):
    k = np.concatenate([pose[:, :2], np.full((17, 1), 2.0)], axis=1)
    return k


def test_torso_diameter_rules():
    k = gt_single(base_pose(0, 0))  # l_shoulder(-50,-200) - r_hip(40,0)
    assert C.torso_diameter(k) == pytest.approx(np.hypot(90, 200))
    k2 = k.copy(); k2[C.KP["l_shoulder"], 2] = 0     # запасная пара r_shoulder - l_hip
    assert C.torso_diameter(k2) == pytest.approx(np.hypot(90, 200))
    k3 = k2.copy(); k3[C.KP["l_hip"], 2] = 0
    assert C.torso_diameter(k3) is None


def test_pck_and_errors_known():
    coco = make_coco(n=1)
    gts = E.gt_by_image(coco)
    gt, w, h = gts[1]
    D = C.torso_diameter(gt)
    pred = np.concatenate([gt[:, :2], np.ones((17, 1))], axis=1)
    pred[C.KP["l_knee"], 0] += 0.3 * D     # ошибка 0.3 D > 0.2
    pred[C.KP["nose"], 1] += 0.1 * D       # 0.1 D < 0.2
    r = E.pck_and_errors(gts, {1: (pred, 1.0)})
    assert r["pck"] == pytest.approx(16 / 17)
    assert r["err_found"].max() == pytest.approx(0.3)
    assert r["err_by_joint"][C.KP["nose"]][0][0] == pytest.approx(0.1)
    # человек не найден: все точки ошибочны, ошибка = inf
    r0 = E.pck_and_errors(gts, {})
    assert r0["pck"] == 0 and np.all(np.isinf(r0["err_all"])) and r0["err_found"].size == 0


def test_unlabeled_points_ignored_and_image_without_torso_skipped():
    coco = make_coco(n=2)
    a = coco["annotations"][0]
    for j in range(5, 17):  # размечены только голова и плечи отсутствуют -> нет торса
        a["keypoints"][3 * j + 2] = 0
    gts = E.gt_by_image(coco)
    pbi = E.preds_by_image(gt_as_pred(coco))
    r = E.pck_and_errors(gts, pbi)
    assert r["n_images"] == 1 and r["n_points"] == 17


def test_kmr_fmr_synthetic():
    coco = make_coco(n=4, w=640, h=800)
    preds = gt_as_pred(coco)
    # изобр. 1: человека нет
    preds[0]["kps"] = None
    # изобр. 2: score носа 0.5 (потеряна при tau>0.5)
    preds[1]["kps"][C.KP["nose"]][2] = 0.5
    # изобр. 3: нос вне кадра
    preds[2]["kps"][C.KP["nose"]][0] = -5
    m = E.evaluate_images(coco, preds, n_boot=0)
    assert m["fmr"] == pytest.approx(0.25)
    nose = m["kmr_by_joint"]["nose"]
    assert nose["kmr"][0.4] == pytest.approx(2 / 4)   # нет человека + вне кадра
    assert nose["kmr"][0.6] == pytest.approx(3 / 4)   # + низкий score
    assert nose["counts"][0.6][1] == 1 and nose["counts"][0.6][3] == 1
    ankle = m["kmr_by_joint"]["l_ankle"]["kmr"]
    assert ankle[0.9] == pytest.approx(1 / 4)
    # площадь: 0.25 для tau<=0.5, 0.5? -> монотонно не убывает
    v = [nose["kmr"][t] for t in C.TAUS]
    assert v == sorted(v)
    assert 0 < nose["auc"] < 1


def test_restrict_to_predicted_images():
    coco = make_coco(n=6)
    preds = gt_as_pred(coco)[:3]
    assert E.evaluate_images(coco, preds, n_boot=0)["n_images"] == 3
    assert E.evaluate_images(coco, preds, n_boot=0, restrict_to_predicted=False)["n_images"] == 6
    assert E.evaluate_images(coco, preds, n_boot=0, restrict_to_predicted=False)["fmr"] == pytest.approx(0.5)


def test_empty_predictions():
    coco = make_coco(n=3)
    m = E.evaluate_images(coco, [{"image_id": 1, "kps": None, "score": 0}], n_boot=5)
    assert m["AP"] == 0 and m["fmr"] == 1.0


def test_csv_rows():
    coco = make_coco(n=3)
    m = E.evaluate_images(coco, gt_as_pred(coco), n_boot=0)
    meta = C.meta_from_result({"run_id": "x", "created_at": "2026-01-02T00:00:00Z", "env": "desktop",
                               "model": {"id": "m"}}, "f.json")
    row = E.accuracy_row(meta, "score", m)
    assert row["dataset"] == "coco-single" and row["raw_file"] == "f.json" and row["date"] == "2026-01-02"
    rows = E.missing_rows(meta, m)
    assert len(rows) == 18 * 9 and {r["keypoint"] for r in rows} >= {"all", "nose"}

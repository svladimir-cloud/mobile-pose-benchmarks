"""Юнит-тесты метрик на синтетических примерах: python3 -m pytest eval/tests -q"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import metrics as M  # noqa: E402


def make_gt(tmp_path, kps_list, w=200, h=300):
    images, anns = [], []
    for i, kps in enumerate(kps_list, start=1):
        images.append({"id": i, "width": w, "height": h, "file_name": f"{i}.jpg"})
        g = np.asarray(kps, float)
        anns.append({"id": i, "image_id": i, "category_id": 1, "iscrowd": 0, "area": 10000.0,
                     "bbox": [0, 0, 100, 100], "num_keypoints": int((g[:, 2] > 0).sum()),
                     "keypoints": g.ravel().tolist()})
    gt = {"images": images, "annotations": anns,
          "categories": [{"id": 1, "name": "person", "keypoints": M.COCO_KP, "skeleton": []}]}
    p = tmp_path / "gt.json"; p.write_text(json.dumps(gt))
    return str(p), gt


def base_kp():
    rng = np.random.default_rng(0)
    xy = rng.uniform(20, 180, size=(17, 2))
    return np.column_stack([xy, np.full(17, 2.0)])


def pred_from(g, score=0.9, w=200, h=300, iid=1):
    return {"image_id": iid, "width": w, "height": h,
            "keypoints17": [[x, y, score] for x, y in np.asarray(g)[:, :2]]}


def test_perfect_prediction_ap_one(tmp_path):
    g = base_kp(); gf, gt = make_gt(tmp_path, [g])
    preds = [pred_from(g)]
    s, _ = M.oks_ap(gf, M.to_coco_results(preds), [1])
    assert abs(s["AP"] - 1.0) < 1e-6 and abs(s["AR"] - 1.0) < 1e-6
    rows = M.per_image(preds, gt)
    assert abs(rows[0]["oks"] - 1.0) < 1e-9
    assert M.pck(rows)[0] == 1.0


def test_missed_detection(tmp_path):
    g = base_kp(); gf, gt = make_gt(tmp_path, [g, g])
    preds = [pred_from(g, iid=1), {"image_id": 2, "width": 200, "height": 300, "keypoints17": None}]
    s, _ = M.oks_ap(gf, M.to_coco_results(preds), [1, 2])
    assert abs(s["AR"] - 0.5) < 1e-6
    m = M.missing(preds)
    assert m["FMR"] == 0.5 and m["kmr"][0.5][0] == 0.5
    p_all, p_found, _, _ = M.pck(M.per_image(preds, gt))
    assert p_all == 0.5 and p_found == 1.0


def test_kmr_threshold_and_out_of_frame():
    p = pred_from(base_kp(), score=0.35)
    p["keypoints17"][0][0] = -5.0  # нос вне кадра
    m = M.missing([p])
    assert m["kmr"][0.3][0] == 1.0 and sum(m["kmr"][0.3]) == 1.0
    assert all(x == 1.0 for x in m["kmr"][0.4])
    assert m["out_of_frame"][0] == 1


def test_swap_detected_by_oks(tmp_path):
    g = base_kp(); _, gt = make_gt(tmp_path, [g])
    sw = g[M.FLIP]
    rows = M.per_image([pred_from(sw)], gt)
    assert rows[0]["oks_swapped"] > rows[0]["oks"]


def test_pck_threshold(tmp_path):
    g = base_kp(); _, gt = make_gt(tmp_path, [g])
    L = M.torso_diameter(g)
    pr = g.copy(); pr[0, 0] += 0.25 * L
    rows = M.per_image([pred_from(pr)], gt)
    assert abs(M.pck(rows)[0] - 16 / 17) < 1e-9


def test_kmr_auc_constant():
    assert abs(M.kmr_auc({t: 0.3 for t in M.TAUS}) - 0.3) < 1e-9

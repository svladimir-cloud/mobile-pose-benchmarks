"""CSV со страницы телефона (bench/phone) → строки data/perf_*.csv (схема data/README.md).

  python3 -I eval/phone_to_perf.py results/raw/phone/*.csv --out data/perf_phone_<дата>.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import PERF, r4  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def browser_from_ua(ua):
    for name, key in (("YaBrowser", "YaBrowser/"), ("Edge", "EdgA/"), ("Firefox", "Firefox/"), ("Chrome", "Chrome/"),
                      ("Safari", "Version/")):
        if key in ua:
            return name, ua.split(key)[1].split(" ")[0]
    return "", ""


def os_from_ua(ua):
    if "Android" in ua:
        return "Android", ua.split("Android ")[1].split(";")[0]
    if "iPhone" in ua or "iPad" in ua:
        return "iOS", ua.split(" OS ")[1].split(" ")[0].replace("_", ".") if " OS " in ua else ""
    return "", ""


def convert(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    h = rows[0]
    cold = {r["phase"]: float(r["latency_ms"]) for r in rows if r["phase"].startswith(("cold_", "first_"))}
    lat = [float(r["latency_ms"]) for r in rows if r["phase"] == "measure"]
    br, brv = browser_from_ua(h["user_agent"])
    os_, osv = os_from_ua(h["user_agent"])
    try:
        raw = str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        raw = str(path)
    status = h["status"]
    if int(h["run_in_page"]) > 1:
        status += "; cold start not cold (run_in_page>1)"
    return {"run_id": h["run_id"], "date": h["date"][:10], "env": "phone", "model": h["model"], "variant": h["variant"],
            "runtime": h["runtime"], "runtime_version": h["runtime_version"], "backend": h["backend"],
            "input_size": "256x256", "raw_file": raw, "device": h["device"], "os": os_, "os_version": osv,
            "browser": br, "browser_version": brv, "wasm_simd": h["wasm_simd"], "wasm_threads": h["wasm_threads"],
            "webgl2": h["webgl2"], "webgpu": h["webgpu"], "n_warmup": h["n_warmup"], "n_measure": len(lat),
            "latency_p50_ms": np.percentile(lat, 50) if lat else None,
            "latency_p95_ms": np.percentile(lat, 95) if lat else None,
            "cold_load_ms": cold.get("cold_load_ms"), "cold_init_ms": cold.get("cold_init_ms"),
            "first_infer_ms": cold.get("first_infer_ms"), "model_size_mb": 5.7777,
            "battery_pct": h["battery_pct"], "temp_c": h["temp_c"], "status": status,
            "gpu_renderer": h["webgl_renderer"] if h["backend"] == "webgl" else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PERF, extrasaction="ignore")
        w.writeheader()
        for p in a.csv:
            w.writerow({k: r4(v) for k, v in convert(p).items()})


if __name__ == "__main__":
    main()

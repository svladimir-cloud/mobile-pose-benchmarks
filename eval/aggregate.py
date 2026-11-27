#!/usr/bin/env python3
"""Сводка прогонов производительности: results/raw/*.json (mode=speed|long) ->
perf_*.csv (одна строка на файл) и Markdown-таблица пригодности по парам (устройство, модель)."""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C  # noqa: E402

PERF_COLS = C.META_COLS + [
    "device", "soc", "os", "os_version", "browser", "browser_version",
    "wasm_simd", "wasm_threads", "webgl2", "webgpu", "n_warmup", "n_measure",
    "latency_p50_ms", "latency_p95_ms", "e2e_fps_median", "dropped_frames_pct",
    "cold_load_ms", "cold_init_ms", "first_infer_ms", "model_size_mb", "download_mb",
    "peak_mem_mb", "fps_min1", "fps_min10", "fps_ratio_10_1", "battery_pct", "temp_c", "status",
    # дополнительные
    "mode", "model_fps", "e2e_latency_p95_ms"]


def _g(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def device_label(res):
    d = res.get("device") or {}
    return d.get("user_label") or d.get("ua_model") or "unknown"


def perf_row(res, raw_file=""):
    """Одна строка perf_*.csv по result-json."""
    r = C.meta_from_result(res, raw_file)
    d, cap, t = res.get("device") or {}, res.get("capabilities") or {}, res.get("timing") or {}
    cs = res.get("cold_start") or {}
    pm = t.get("per_minute") or []
    f1 = next((x["fps"] for x in pm if x.get("minute") == 1), None)
    f10 = next((x["fps"] for x in pm if x.get("minute") == 10), pm[-1]["fps"] if len(pm) > 1 else None)
    ratio = t.get("fps_ratio_10_1")
    if ratio is None and f1 and f10 is not None and len(pm) > 1:
        ratio = f10 / f1
    wp = cap.get("webgpu")
    status = res.get("status", "failed")
    if status != "ok":
        status = f"failed: {res.get('error') or 'причина не указана'}"
    bat = d.get("battery")
    r.update({
        "device": device_label(res), "soc": d.get("soc", ""), "os": d.get("platform", ""),
        "os_version": d.get("platform_version", ""), "browser": d.get("browser", ""),
        "browser_version": d.get("browser_version", ""),
        "wasm_simd": cap.get("wasm_simd"), "wasm_threads": cap.get("wasm_threads"),
        "webgl2": cap.get("webgl2"), "webgpu": wp,
        "n_warmup": t.get("n_warmup"), "n_measure": t.get("n_measure"),
        "latency_p50_ms": _g(t, "latency_ms", "p50"), "latency_p95_ms": _g(t, "latency_ms", "p95"),
        "e2e_fps_median": t.get("e2e_fps"), "dropped_frames_pct": t.get("dropped_frames_pct"),
        "cold_load_ms": cs.get("download_ms"), "cold_init_ms": cs.get("init_ms"),
        "first_infer_ms": cs.get("first_infer_ms"),
        "model_size_mb": _g(res, "model", "size_mb"),
        "download_mb": (cs["download_bytes"] / 1e6) if cs.get("download_bytes") is not None else None,
        "peak_mem_mb": _g(res, "memory", "js_heap_mb_peak"),
        "fps_min1": f1, "fps_min10": f10, "fps_ratio_10_1": ratio,
        "battery_pct": bat.get("level") * 100 if isinstance(bat, dict) and bat.get("level") is not None else None,
        "temp_c": d.get("temp_c_start"), "status": status, "mode": res.get("mode"),
        "model_fps": t.get("model_fps"), "e2e_latency_p95_ms": _g(t, "e2e_latency_ms", "p95"),
    })
    return r


def exercise_ok(ex, amp_thr=5.0):
    """True/False по результату exercise.py (JSON): ошибка числа повторов 0, MAE амплитуды <= порога."""
    reps = ex.get("reps") or {}
    if "abs_count_error" not in reps:
        return None
    if reps["abs_count_error"] != 0 or reps.get("n_ref", 0) == 0:
        return False
    mae = _g(reps, "amplitude_abs_error", "mae")
    return bool(mae is not None and mae == mae and mae <= amp_thr)


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v else None


def build_pairs(rows, exercises=(), fmin=10.0, amp_thr=5.0):
    """Список пар (устройство, env, модель): лучший бэкенд (макс. e2e_fps среди ok) и вердикты."""
    groups = {}
    for r in rows:
        groups.setdefault((r["device"], r["env"], r["model"], r["variant"]), []).append(r)
    ex_by = {}
    for ex in exercises:
        k = ex.get("run") or {}
        ex_by.setdefault((k.get("device"), k.get("env"), k.get("model"), k.get("variant")), []).append(
            (k.get("backend"), exercise_ok(ex, amp_thr)))
    out = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        rs = groups[key]
        ok = [r for r in rs if r["status"] == "ok"]
        best = max(ok, key=lambda r: _num(r["e2e_fps_median"]) or -1) if ok else None
        works = best is not None
        e2e = _num(best["e2e_fps_median"]) if best else None
        fast = (e2e >= fmin) if e2e is not None else (False if works else None)
        acc = None
        exs = [v for b, v in ex_by.get(key, []) if v is not None and (best is None or b in (None, best["backend"]))]
        if exs:
            acc = all(exs)
        out.append({"device": key[0], "env": key[1], "model": key[2], "variant": key[3],
                    "best": best, "works": works, "fast": fast, "accurate": acc,
                    "fail_reasons": sorted({r["status"] for r in rs if r["status"] != "ok"}),
                    "n_files": len(rs)})
    return out


def suitable(p, require_accuracy=False):
    if not p["works"] or not p["fast"]:
        return False
    if p["accurate"] is False:
        return False
    return not (require_accuracy and p["accurate"] is None)


def _yn(v):
    return "н/д" if v is None else ("да" if v else "нет")


def _f(v, nd=1):
    v = _num(v)
    return "—" if v is None else f"{v:.{nd}f}"


def render_markdown(pairs, fmin=10.0, amp_thr=5.0, summary_env="phone"):
    L = [f"# Пригодность моделей по устройствам", "",
         f"Критерии: работает — есть прогон со статусом ok; достаточно быстро — e2e_fps лучшего бэкенда >= {fmin:g}; "
         f"достаточно точно — по результатам exercise.py: ошибка числа повторов 0 и MAE амплитуды <= {amp_thr:g}° "
         "(н/д — результатов нет). Лучший бэкенд — по наибольшему e2e_fps среди прогонов ok. "
         "Строки с env=desktop-headless — проверка стенда, не результат телефонов.", ""]
    L += ["| Устройство | env | Модель | Лучший бэкенд | model_fps | e2e_fps | p95, мс | fps 10-я/1-я | Холодный старт, с | Работает | Достаточно быстро | Достаточно точно |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in pairs:
        b = p["best"]
        if b:
            cold = sum(_num(b[k]) or 0 for k in ("cold_load_ms", "cold_init_ms", "first_infer_ms")) / 1000
            have = any(_num(b[k]) is not None for k in ("cold_load_ms", "cold_init_ms", "first_infer_ms"))
            cells = [b["backend"], _f(b["model_fps"]), _f(b["e2e_fps_median"]), _f(b["latency_p95_ms"]),
                     _f(b["fps_ratio_10_1"], 2), f"{cold:.1f}" if have else "—"]
        else:
            cells = ["—"] * 6
        name = f"{p['model']} {p['variant']}".strip()
        works = _yn(p["works"]) if p["works"] else "нет: " + "; ".join(x.replace("failed: ", "") for x in p["fail_reasons"])
        L.append(f"| {p['device']} | {p['env']} | {name} | " + " | ".join(cells) +
                 f" | {works} | {_yn(p['fast'])} | {_yn(p['accurate'])} |")
    L += ["", f"## Доля устройств, на которых модель пригодна (env={summary_env})", ""]
    models = sorted({(p["model"], p["variant"]) for p in pairs if p["env"] == summary_env})
    if not models:
        L.append(f"TODO: нужны данные (нет прогонов с env={summary_env}).")
    else:
        L += ["| Модель | Устройств | Работает | Достаточно быстро | Точность известна / достаточна | Пригодна (работает + быстро, точность не хуже порога) | Доля |",
              "|---|---|---|---|---|---|---|"]
        for m in models:
            ps = [p for p in pairs if p["env"] == summary_env and (p["model"], p["variant"]) == m]
            n = len(ps)
            nk = [p for p in ps if p["accurate"] is not None]
            ns = sum(suitable(p) for p in ps)
            L.append(f"| {m[0]} {m[1]} | {n} | {sum(p['works'] for p in ps)} | "
                     f"{sum(bool(p['fast']) for p in ps)} | {len(nk)} / {sum(bool(p['accurate']) for p in nk)} | "
                     f"{ns} | {ns / n:.0%} |")
        L += ["", "Малая выборка устройств: доли описательные, без доверительных интервалов."]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- CLI
def main(argv=None):
    import json
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="results/raw", help="каталог с result-json")
    ap.add_argument("--out-csv", default=None, help="perf_*.csv (по умолчанию не пишется)")
    ap.add_argument("--out-md", default=None, help="Markdown (по умолчанию stdout)")
    ap.add_argument("--fmin", type=float, default=10.0, help="порог e2e_fps «достаточно быстро»")
    ap.add_argument("--amp-thr", type=float, default=5.0)
    ap.add_argument("--exercise-json", nargs="*", default=[], help="JSON-выходы exercise.py")
    ap.add_argument("--summary-env", default="phone")
    a = ap.parse_args(argv)
    rows = []
    for path in sorted(glob.glob(os.path.join(a.raw, "*.json"))):
        try:
            res = C.load_json(path)
        except ValueError as e:
            print(f"пропуск {path}: {e}", file=sys.stderr)
            continue
        if res.get("mode") not in ("speed", "long"):
            continue
        rows.append(perf_row(res, os.path.basename(path)))
    if a.out_csv:
        C.write_csv(a.out_csv, PERF_COLS, rows)
    exs = [C.load_json(p) for p in a.exercise_json]
    md = render_markdown(build_pairs(rows, exs, a.fmin, a.amp_thr), a.fmin, a.amp_thr, a.summary_env)
    if a.out_md:
        with open(a.out_md, "w", encoding="utf-8") as f:
            f.write(md)
    else:
        sys.stdout.write(md)
    print(f"файлов speed/long: {len(rows)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

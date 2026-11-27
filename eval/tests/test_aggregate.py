import aggregate as A


def res(device, model, backend, e2e, status="ok", env="phone", error=None, ratio=0.9, soc=""):
    r = {"run_id": f"{device}-{model}-{backend}", "created_at": "2026-11-27T10:00:00Z", "mode": "long",
         "env": env, "status": status, "error": error,
         "device": {"user_label": device, "platform": "Android", "platform_version": "13",
                    "browser": "Chrome", "browser_version": "131", "soc": soc,
                    "battery": {"level": 0.8, "charging": False}},
         "capabilities": {"wasm_simd": True, "wasm_threads": True, "webgl2": True, "webgpu": False},
         "model": {"id": model, "variant": "t", "runtime": "ort", "runtime_version": "1.30",
                   "backend_actual": backend, "size_mb": 13.4},
         "cold_start": {"download_ms": 500, "download_bytes": 13_400_000, "init_ms": 300, "first_infer_ms": 700}}
    if status == "ok":
        r["timing"] = {"n_warmup": 50, "n_measure": 200, "latency_ms": {"p50": 30, "p95": 45},
                       "model_fps": e2e + 2, "e2e_fps": e2e, "dropped_frames_pct": 5.0,
                       "per_minute": [{"minute": 1, "fps": e2e}, {"minute": 10, "fps": e2e * ratio}],
                       "fps_ratio_10_1": ratio}
        r["memory"] = {"js_heap_mb_peak": 200}
    return r


def test_perf_row_ok_and_failed():
    row = A.perf_row(res("Phone A", "rtmpose", "wasm", 20.0), "f.json")
    assert row["status"] == "ok" and row["device"] == "Phone A" and row["latency_p95_ms"] == 45
    assert row["fps_min1"] == 20.0 and row["fps_min10"] == 18.0 and row["download_mb"] == 13.4
    assert row["battery_pct"] == 80 and row["cold_init_ms"] == 300 and row["raw_file"] == "f.json"
    bad = A.perf_row(res("Phone A", "rtmpose", "webgpu", 0, status="failed", error="нет WebGPU"), "g.json")
    assert bad["status"] == "failed: нет WebGPU" and bad["latency_p95_ms"] is None


def test_pairs_and_verdicts():
    rows = [A.perf_row(r) for r in [
        res("A", "m1", "wasm", 8.0), res("A", "m1", "webgl", 14.0),        # лучший бэкенд webgl, быстро
        res("A", "m2", "wasm", 6.0),                                        # медленно
        res("B", "m1", "wasm", 0, status="failed", error="OOM"),            # не работает
        res("B", "m2", "wasm", 12.0),
        res("H", "m1", "wasm", 50.0, env="desktop-headless")]]
    ex = [{"run": {"device": "B", "env": "phone", "model": "m2", "variant": "t", "backend": "wasm"},
           "reps": {"abs_count_error": 0, "n_ref": 5, "amplitude_abs_error": {"mae": 2.0}}}]
    pairs = A.build_pairs(rows, ex, fmin=10)
    d = {(p["device"], p["model"]): p for p in pairs}
    assert d[("A", "m1")]["best"]["backend"] == "webgl" and d[("A", "m1")]["fast"] is True
    assert d[("A", "m2")]["works"] and d[("A", "m2")]["fast"] is False
    assert d[("B", "m1")]["works"] is False and "failed: OOM" in d[("B", "m1")]["fail_reasons"]
    assert d[("B", "m2")]["accurate"] is True and d[("A", "m1")]["accurate"] is None
    md = A.render_markdown(pairs, 10, 5, "phone")
    assert "нет: OOM" in md and "desktop-headless" in md
    # доли только по env=phone: m1 — устройства A (пригодно), B (нет) -> 50%; m2 — A (медленно), B (да) -> 50%
    lines = [l for l in md.splitlines() if l.startswith("| m1 t") or l.startswith("| m2 t")]
    assert len(lines) == 2 and all(l.endswith("50% |") for l in lines)


def test_exercise_ok():
    assert A.exercise_ok({"reps": {"abs_count_error": 1, "n_ref": 5, "amplitude_abs_error": {"mae": 1}}}) is False
    assert A.exercise_ok({"reps": {"abs_count_error": 0, "n_ref": 5, "amplitude_abs_error": {"mae": 7}}}) is False
    assert A.exercise_ok({"reps": {"error": "x"}}) is None


def test_no_phone_data_todo():
    rows = [A.perf_row(res("H", "m1", "wasm", 30.0, env="desktop-headless"))]
    assert "TODO: нужны данные" in A.render_markdown(A.build_pairs(rows), 10, 5, "phone")

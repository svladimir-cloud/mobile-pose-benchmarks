# Итоговые данные для статьи

Один эксперимент — один CSV: `<тип>_<этап|дата>.csv`. Кодировка UTF-8, разделитель запятая, десятичная точка. Каждая строка ссылается на сырой файл (`raw_file`). Числа в статье берутся только отсюда.

Общие столбцы всех таблиц: `run_id, date, env` (`desktop-headless` / `phone`), `model, variant, runtime, runtime_version, backend, input_size, raw_file`.

## accuracy_*.csv — точность на изображениях
`dataset` (`coco-single` / `coco-val2017` / `own-frames`), `n_images`, `AP, AP50, AP75, AR`, `PCK@0.2`, `err_median_norm, err_p95_norm`, `score_source` (score / visibility / presence), `ci_low, ci_high` (для AP, бутстреп).

## missing_*.csv — полнота
`dataset, video_id, exercise, position` (лёжа/сидя/стоя), `view` (0/45/90), `light`, `keypoint` (имя COCO или `all`), `tau`, `frames_total`, `frames_no_person` (FMR), `frames_missing` (KMR), `out_of_frame`, `ckmr_k`, `ckmr`, `gap_mean_frames, gap_max_frames`, `lr_swaps`.

## jitter_*.csv — стабильность
`video_id, segment` (static/motion), `keypoint`, `filter` (none / one-euro), `jacc_px_per_frame2`, `speed_px_per_frame`, `filter_lag_ms`.

## perf_*.csv — производительность
`device, soc, os, os_version, browser, browser_version, wasm_simd, wasm_threads, webgl2, webgpu`, `n_warmup, n_measure`, `latency_p50_ms, latency_p95_ms`, `e2e_fps_median`, `dropped_frames_pct`, `cold_load_ms, cold_init_ms, first_infer_ms`, `model_size_mb, download_mb`, `peak_mem_mb`, `fps_min1, fps_min10, fps_ratio_10_1`, `battery_pct, temp_c`, `status` (ok / failed: причина).

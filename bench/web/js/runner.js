// Цикл замера одной конфигурации «модель × бэкенд × источник» (R3.4, R4.4, R5.1).
import { createAdapter } from "./models.js";
import { heapMb, readBattery } from "./caps.js";
import { Loader, SKELETON, now, round, sleep, srcSize, summarize } from "./util.js";

// Тайм-ауты: загрузка+инициализация 60 с (R4.4), один кадр 30 с. Зависшая конфигурация -> status=failed,
// матрица продолжается со следующей.
export const LOAD_TIMEOUT_MS = 60000, FRAME_TIMEOUT_MS = 30000;
function withTimeout(p, ms, what) {
  let t;
  return Promise.race([p, new Promise((_, rej) => { t = setTimeout(() => rej(new Error(`тайм-аут ${ms / 1000} с: ${what}`)), ms); })])
    .finally(() => clearTimeout(t));
}

export const DEFAULT_PARAMS = {
  warmup_frames: 50, warmup_max_s: 15,
  measure_frames: 200, measure_max_s: 40,
  long_minutes: 10, det_interval: 10, store_frames: true,
};

// ---------------------------------------------------------------- источники кадров
export class VideoSource {
  constructor(video) { this.video = video; this.hasRvfc = "requestVideoFrameCallback" in HTMLVideoElement.prototype; }
  async open({ kind, url, facing }) {
    const v = this.video;
    this.kind = kind;
    if (v.srcObject) { v.srcObject.getTracks().forEach((t) => t.stop()); v.srcObject = null; }
    if (kind === "camera") {
      // без жёстких width/height: иначе браузер пережимает портретный кадр (урок lfk-bench)
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: facing || "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
      });
      v.srcObject = stream;
      this.facing = facing || "user";
    } else {
      v.src = url; v.loop = true;
    }
    v.muted = true; v.playsInline = true;
    await v.play();
    await new Promise((r) => (v.readyState >= 2 ? r() : v.addEventListener("loadeddata", r, { once: true })));
    this.name = kind === "camera" ? "camera-" + this.facing : url.split("/").pop().replace(/\.[^.]+$/, "");
  }
  info() {
    const { w, h } = srcSize(this.video);
    const s = this.kind === "camera" ? this.video.srcObject.getVideoTracks()[0].getSettings() : {};
    return { kind: this.kind, name: this.name, width: w, height: h, fps: s.frameRate || null, facing: this.kind === "camera" ? this.facing : null };
  }
  restart() { if (this.kind === "video") this.video.currentTime = 0; }
  // следующий новый кадр: { ts, mediaTime, presented, frameTs }
  next() {
    const v = this.video;
    return new Promise((resolve) => {
      if (this.hasRvfc) {
        // если видео встало (нет новых кадров 1 с) — перезапустить воспроизведение и отдать текущий кадр
        const guard = setTimeout(() => {
          if (v.paused || v.ended) v.play().catch(() => {});
          resolve({ ts: performance.now(), mediaTime: v.currentTime, presented: null, frameTs: null, stalled: true });
        }, 1000);
        v.requestVideoFrameCallback((ts, md) => { clearTimeout(guard); resolve({
          ts, mediaTime: md.mediaTime, presented: md.presentedFrames,
          frameTs: md.captureTime || md.presentationTime || md.expectedDisplayTime || ts,
        }); });
      } else {
        requestAnimationFrame((ts) => resolve({ ts, mediaTime: v.currentTime, presented: null, frameTs: ts }));
      }
    });
  }
  close() {
    const v = this.video;
    if (v.srcObject) { v.srcObject.getTracks().forEach((t) => t.stop()); v.srcObject = null; }
    v.pause();
  }
}

function drawSkeleton(canvas, src, kps, thr = 0.3) {
  const { w, h } = srcSize(src);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  if (!kps) return;
  ctx.lineWidth = Math.max(2, w / 200);
  ctx.strokeStyle = "#22c55e"; ctx.fillStyle = "#f97316";
  ctx.beginPath();
  for (const [a, b] of SKELETON) {
    if (kps[a][2] < thr || kps[b][2] < thr) continue;
    ctx.moveTo(kps[a][0], kps[a][1]); ctx.lineTo(kps[b][0], kps[b][1]);
  }
  ctx.stroke();
  const r = Math.max(3, w / 150);
  for (const p of kps) if (p[2] >= thr) { ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 6.283); ctx.fill(); }
}

const roundKps = (kps) => (kps ? kps.map((p) => [round(p[0], 1), round(p[1], 1), round(p[2], 3)]) : null);

// ---------------------------------------------------------------- общий каркас результата
function baseDoc(ctx, model, backend, mode) {
  return {
    schema: "mpb-result/2", run_id: ctx.newId(), batch_id: ctx.batchId, created_at: new Date().toISOString(),
    mode, env: ctx.env, status: "ok", error: null,
    device: structuredClone(ctx.device), capabilities: ctx.caps,
    model: {
      id: model.id, family: model.family, variant: model.variant, runtime: model.runtime,
      runtime_version: "", backend_requested: backend, backend_actual: null, threads: null,
      input_size: model.input_size, detector: null, det_interval: null,
      files: [], size_mb: null, score_source: null, license: model.license,
    },
    source: null, params: {}, cold_start: null, timing: null,
    memory: { js_heap_mb_peak: null }, validity: { hidden_during_run: false, wake_lock: !!ctx.wakeLock, notes: "" },
  };
}

async function setupAdapter(ctx, doc, model, backend, adapterOpts, firstSrc) {
  const loader = new Loader();
  const adapter = await withTimeout(createAdapter(model, backend, adapterOpts, loader, ctx.models), LOAD_TIMEOUT_MS, "загрузка и инициализация");
  const ls = loader.summary();
  Object.assign(doc.model, {
    runtime_version: adapter.meta.runtime_version || ctx.versions[model.runtime] || "",
    backend_actual: adapter.meta.backend_actual, threads: adapter.meta.threads,
    score_source: adapter.meta.score_source, detector: adapter.meta.detector || null,
    det_interval: adapter.meta.det_interval || null, files: ls.files,
    size_mb: round(ls.download_bytes / 1e6, 2),
  });
  if (ctx.versions[vendorKey(model.runtime)]) doc.model.runtime_version = ctx.versions[vendorKey(model.runtime)];
  const t0 = now();
  await withTimeout(adapter.estimate(firstSrc, now()), LOAD_TIMEOUT_MS, "первый кадр");
  doc.cold_start = {
    download_ms: ls.download_ms, download_bytes: ls.download_bytes, from_cache: ls.from_cache,
    init_ms: round(adapter.initMs, 1), first_infer_ms: round(now() - t0, 1),
  };
  return adapter;
}

function vendorKey(runtime) {
  return { "onnxruntime-web": "onnxruntime-web", "mediapipe-tasks-vision": "@mediapipe/tasks-vision", tfjs: "@tensorflow/tfjs-core" }[runtime];
}

// ---------------------------------------------------------------- скорость: видео/камера в реальном времени
export async function runRealtime(ctx, model, backend, source, params, mode, ui) {
  const doc = baseDoc(ctx, model, backend, mode);
  doc.params = { ...params };
  doc.source = source.info();
  let adapter = null;
  const hiddenWatch = () => { if (document.hidden) doc.validity.hidden_during_run = true; };
  document.addEventListener("visibilitychange", hiddenWatch);
  try {
    adapter = await setupAdapter(ctx, doc, model, backend, { mode: "video", detInterval: params.det_interval }, source.video);
    const store = params.store_frames && source.kind !== "camera";
    const frames = [];
    const lat = [], e2e = [];
    let heap = heapMb() || 0;
    // прогрев
    ui.phase("прогрев");
    const tw = now();
    let nw = 0;
    while (nw < params.warmup_frames && now() - tw < params.warmup_max_s * 1000) {
      const f = await source.next();
      await withTimeout(adapter.estimate(source.video, now()), FRAME_TIMEOUT_MS, "кадр прогрева");
      nw++;
    }
    // замер
    ui.phase(mode === "long" ? "длительный прогон" : "замер");
    source.restart();
    let f0 = await source.next();
    const start = now();
    const firstPresented = f0.presented;
    let lastPresented = f0.presented, processed = 0, drawn = 0, pending = f0;
    const limitMs = mode === "long" ? params.long_minutes * 60000 : params.measure_max_s * 1000;
    const limitFrames = mode === "long" ? Infinity : params.measure_frames;
    const minutes = [];
    let minuteStart = start, mLat = [], mFrames = 0;
    let lastUi = 0;
    while (processed < limitFrames && now() - start < limitMs) {
      const f = pending || (await source.next());
      pending = null;
      if (f.presented !== null && f.presented === lastPresented && processed > 0) continue;
      if (f.presented !== null) lastPresented = f.presented;
      if (f.stalled) doc.validity.notes = "видео останавливалось (нет новых кадров > 1 с)";
      const t0 = now();
      const r = await withTimeout(adapter.estimate(source.video, t0), FRAME_TIMEOUT_MS, "кадр");
      const t1 = now();
      drawSkeleton(ui.overlay, source.video, r.kps);
      const t2 = now();
      processed++; drawn++;
      lat.push(t1 - t0);
      if (f.frameTs && f.frameTs <= t2) e2e.push(t2 - f.frameTs);
      if (store) frames.push({ i: processed - 1, media_time_s: round(f.mediaTime, 4), t_ms: round(t0 - start, 1), infer_ms: round(t1 - t0, 2), kps: roundKps(r.kps) });
      mLat.push(t1 - t0); mFrames++;
      if (mode === "long" && t2 - minuteStart >= 60000) {
        const s = summarize(mLat), b = await readBattery();
        minutes.push({ minute: minutes.length + 1, fps: round(mFrames / ((t2 - minuteStart) / 1000), 2), p50: s.p50, p95: s.p95, battery: b ? b.level : null });
        minuteStart = t2; mLat = []; mFrames = 0;
      }
      if (t2 - lastUi > 500) {
        heap = Math.max(heap, heapMb() || 0);
        ui.progress({ processed, elapsed: (t2 - start) / 1000, fps: processed / ((t2 - start) / 1000), lat: t1 - t0, limitMs, limitFrames });
        lastUi = t2;
      }
    }
    const dur = (now() - start) / 1000;
    if (mode === "long" && mFrames > 0 && minutes.length < params.long_minutes) {
      const s = summarize(mLat), b = await readBattery();
      minutes.push({ minute: minutes.length + 1, fps: round(mFrames / ((now() - minuteStart) / 1000), 2), p50: s.p50, p95: s.p95, battery: b ? b.level : null, partial: true });
    }
    const presented = firstPresented !== null && lastPresented !== null ? lastPresented - firstPresented + 1 : null;
    const ls = summarize(lat), es = summarize(e2e);
    doc.timing = {
      n_warmup: nw, n_measure: processed, duration_s: round(dur, 2),
      latency_ms: { mean: ls.mean, p50: ls.p50, p95: ls.p95, p99: ls.p99 },
      e2e_latency_ms: { mean: es.mean, p50: es.p50, p95: es.p95, p99: es.p99 },
      model_fps: round(processed / dur, 2), e2e_fps: round(drawn / dur, 2),
      frames_presented: presented, frames_processed: processed,
      dropped_frames_pct: presented ? round(100 * Math.max(0, 1 - processed / presented), 1) : null,
    };
    if (mode === "long") {
      doc.timing.per_minute = minutes;
      const full = minutes.filter((m) => !m.partial);
      doc.timing.fps_ratio_10_1 = full.length >= 2 ? round(full[full.length - 1].fps / full[0].fps, 3) : null;
    }
    doc.memory.js_heap_mb_peak = heap || null;
    if (store) doc.frames = frames;
  } catch (e) {
    doc.status = "failed"; doc.error = String(e && e.message ? e.message : e);
  } finally {
    document.removeEventListener("visibilitychange", hiddenWatch);
    drawSkeleton(ui.overlay, source.video, null);
    if (adapter) adapter.dispose();
  }
  return doc;
}

// ---------------------------------------------------------------- точность: последовательность кадров (все кадры)
function loadImage(url) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.decoding = "async";
    im.onload = () => res(im);
    im.onerror = () => rej(new Error("не загрузился кадр " + url));
    im.src = url;
  });
}

export async function runFrames(ctx, model, backend, set, params, ui) {
  const doc = baseDoc(ctx, model, backend, "frames");
  doc.params = { ...params };
  let adapter = null;
  try {
    const meta = await (await fetch(set.url)).json();
    const base = set.url.replace(/frames\.json$/, "");
    const files = meta.files.slice(0, params.max_frames || meta.files.length);
    doc.source = { kind: "frames", name: set.name, width: meta.width, height: meta.height, fps: meta.fps, facing: null };
    let next = loadImage(base + files[0]);
    const first = await next;
    adapter = await setupAdapter(ctx, doc, model, backend, { mode: "video", detInterval: params.det_interval }, first);
    adapter.reset();
    const frames = [], lat = [];
    for (let i = 0; i < files.length; i++) {
      const im = i === 0 ? first : await next;
      if (i + 1 < files.length) next = loadImage(base + files[i + 1]);
      const ts = (i * 1000) / meta.fps + 1;
      const t0 = now();
      const r = await withTimeout(adapter.estimate(im, ts), FRAME_TIMEOUT_MS, "кадр");
      const dt = now() - t0;
      lat.push(dt);
      frames.push({ i, media_time_s: round(i / meta.fps, 4), t_ms: null, infer_ms: round(dt, 2), kps: roundKps(r.kps) });
      if (i % 10 === 0) { drawSkeleton(ui.overlay, im, r.kps); ui.progress({ processed: i + 1, total: files.length, lat: dt }); }
    }
    const ls = summarize(lat);
    doc.timing = { n_warmup: 0, n_measure: frames.length, latency_ms: { mean: ls.mean, p50: ls.p50, p95: ls.p95, p99: ls.p99 } };
    doc.frames = frames;
  } catch (e) {
    doc.status = "failed"; doc.error = String(e && e.message ? e.message : e);
  } finally { if (adapter) adapter.dispose(); }
  return doc;
}

// ---------------------------------------------------------------- точность: COCO (отдельные изображения)
export async function runImages(ctx, model, backend, items, params, ui) {
  const doc = baseDoc(ctx, model, backend, "images");
  doc.params = { ...params, n_images: items.length };
  doc.source = { kind: "images", name: "coco-single", width: null, height: null, fps: null, facing: null };
  let adapter = null;
  try {
    let next = loadImage("/coco/" + items[0].file_name);
    const first = await next;
    adapter = await setupAdapter(ctx, doc, model, backend, { mode: "image", detInterval: 1 }, first);
    const preds = [], lat = [];
    for (let i = 0; i < items.length; i++) {
      const im = i === 0 ? first : await next;
      if (i + 1 < items.length) next = loadImage("/coco/" + items[i + 1].file_name);
      adapter.reset();
      const t0 = now();
      const r = await withTimeout(adapter.estimate(im, now()), FRAME_TIMEOUT_MS, "изображение");
      const dt = now() - t0;
      lat.push(dt);
      preds.push({ image_id: items[i].image_id, kps: roundKps(r.kps), score: round(r.score, 4), box: r.box ? r.box.map((v) => round(v, 1)) : null, infer_ms: round(dt, 2) });
      if (i % 20 === 0) { drawSkeleton(ui.overlay, im, r.kps); ui.progress({ processed: i + 1, total: items.length, lat: dt }); }
    }
    const ls = summarize(lat);
    doc.timing = { n_warmup: 0, n_measure: preds.length, latency_ms: { mean: ls.mean, p50: ls.p50, p95: ls.p95, p99: ls.p99 } };
    doc.predictions = preds;
  } catch (e) {
    doc.status = "failed"; doc.error = String(e && e.message ? e.message : e);
  } finally { if (adapter) adapter.dispose(); }
  return doc;
}

export { sleep };

// Адаптеры моделей. Единый интерфейс (R1.3):
//   const a = await createAdapter(modelInfo, backend, { mode: "video"|"image", detInterval }, loader)
//   const r = await a.estimate(src, timestampMs)   // { kps: [[x,y,s]*17] | null, box, score }
//   a.reset(); a.dispose();
// Координаты — пиксели исходного кадра, порядок COCO-17. Предобработка повторяет
// эталонные конвейеры (MMPose/MMDeploy pipeline.json, Ultralytics LetterBox), чтобы
// результат совпадал с Python-реализацией (bench/python/models.py).
import { MP33_TO_COCO17, abs, loadScript, meanScore, now, srcSize } from "./util.js";

export const BACKENDS = {
  "mediapipe-tasks-vision": ["gpu", "cpu"],
  tfjs: ["webgl", "wasm", "webgpu"],
  "onnxruntime-web": ["webgpu", "wasm", "webgl"],
};
const DET_THR = 0.3;     // порог «человек найден» для детекторов RTMDet/RTMO (как score_thr rtmlib для RTMDet)
const YOLO_THR = 0.25;   // conf по умолчанию в Ultralytics predict

export function preferredBackends(model, caps) {
  const all = BACKENDS[model.runtime] || [];
  return all.filter((b) => (b !== "webgpu" || (caps.webgpu && !caps.webgpu_fallback)) && (b !== "webgl" && b !== "gpu" || caps.webgl2));
}

export async function createAdapter(model, backend, opts, loader, all) {
  if (model.runtime === "mediapipe-tasks-vision") return createMediaPipe(model, backend, opts, loader);
  if (model.runtime === "tfjs") return createMoveNet(model, backend, opts, loader);
  if (model.family === "rtmpose") return createRtmPose(model, backend, opts, loader, all);
  if (model.family === "rtmo") return createRtmo(model, backend, opts, loader);
  if (model.family === "yolo") return createYolo(model, backend, opts, loader);
  throw new Error("нет адаптера для " + model.id);
}

// ---------------------------------------------------------------- MediaPipe
let mpVision = null, mpModule = null;
async function createMediaPipe(model, backend, opts, loader) {
  if (!mpModule) mpModule = await import("../vendor/mediapipe/vision_bundle.mjs");
  const { FilesetResolver, PoseLandmarker } = mpModule;
  const file = model.files.find((f) => f.endsWith(".task"));
  const buf = await loader.get(abs(`models/${model.id}/${file}`));
  const t0 = now();
  if (!mpVision) mpVision = await FilesetResolver.forVisionTasks(abs("vendor/mediapipe/wasm"));
  const image = opts.mode === "image";
  const lm = await PoseLandmarker.createFromOptions(mpVision, {
    baseOptions: { modelAssetBuffer: new Uint8Array(buf), delegate: backend === "gpu" ? "GPU" : "CPU" },
    runningMode: image ? "IMAGE" : "VIDEO", numPoses: 1,
    minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
    outputSegmentationMasks: false,
  });
  const initMs = now() - t0;
  let lastTs = -1;
  return {
    initMs,
    meta: { runtime_version: "tasks-vision", backend_actual: backend, threads: 1, score_source: "visibility" },
    async estimate(src, ts) {
      const { w, h } = srcSize(src);
      let r;
      if (image) r = lm.detect(src);
      else { const t = Math.max(ts, lastTs + 1); lastTs = t; r = lm.detectForVideo(src, t); }
      const L = r.landmarks && r.landmarks[0];
      if (!L) return { kps: null, box: null, score: 0 };
      const kps = MP33_TO_COCO17.map((i) => [L[i].x * w, L[i].y * h, L[i].visibility ?? 0]);
      return { kps, box: null, score: meanScore(kps) };
    },
    reset() { lastTs = -1; },
    dispose() { try { lm.close(); } catch { /* уже закрыт */ } },
  };
}

// ---------------------------------------------------------------- MoveNet (TF.js)
let tfLoaded = null;
async function loadTf() {
  if (!tfLoaded) {
    tfLoaded = (async () => {
      await loadScript(abs("vendor/tfjs/tf-core.min.js"));
      await loadScript(abs("vendor/tfjs/tf-converter.min.js"));
      await loadScript(abs("vendor/tfjs/tf-backend-webgl.min.js"));
      await loadScript(abs("vendor/tfjs/tf-backend-wasm.min.js"));
      try { await loadScript(abs("vendor/tfjs/tf-backend-webgpu.min.js")); } catch { /* нет WebGPU-сборки */ }
      await loadScript(abs("vendor/tfjs/pose-detection.min.js"));
      window.tf.wasm.setWasmPaths(abs("vendor/tfjs/wasm/"));
    })();
  }
  return tfLoaded;
}

async function createMoveNet(model, backend, opts, loader) {
  await loadTf();
  const tf = window.tf, pd = window.poseDetection;
  const base = abs(`models/${model.id}/`);
  const mj = JSON.parse(new TextDecoder().decode(await loader.get(base + "model.json")));
  const specs = [], parts = [];
  for (const g of mj.weightsManifest) {
    specs.push(...g.weights);
    for (const p of g.paths) parts.push(new Uint8Array(await loader.get(base + p)));
  }
  const total = parts.reduce((a, p) => a + p.byteLength, 0);
  const weightData = new Uint8Array(total);
  let off = 0;
  for (const p of parts) { weightData.set(p, off); off += p.byteLength; }
  const t0 = now();
  if (backend === "wasm" && tf.wasm.setThreadsCount && self.crossOriginIsolated) {
    tf.wasm.setThreadsCount(Math.min(4, navigator.hardwareConcurrency || 1));
  }
  const ok = await tf.setBackend(backend);
  if (!ok) throw new Error("TF.js: бэкенд недоступен: " + backend);
  await tf.ready();
  const handler = tf.io.fromMemory({
    modelTopology: mj.modelTopology, weightSpecs: specs, weightData: weightData.buffer,
    format: mj.format, generatedBy: mj.generatedBy, convertedBy: mj.convertedBy,
    signature: mj.signature, userDefinedMetadata: mj.userDefinedMetadata,
  });
  const detector = await pd.createDetector(pd.SupportedModels.MoveNet, {
    modelType: model.variant === "thunder" ? pd.movenet.modelType.SINGLEPOSE_THUNDER : pd.movenet.modelType.SINGLEPOSE_LIGHTNING,
    modelUrl: handler, enableSmoothing: false,
  });
  const initMs = now() - t0;
  let threads = 1;
  try { if (backend === "wasm" && tf.wasm.getThreadsCount) threads = tf.wasm.getThreadsCount(); } catch { /* нет */ }
  const image = opts.mode === "image";
  return {
    initMs,
    meta: { runtime_version: tf.version_core, backend_actual: tf.getBackend(), threads, score_source: "score" },
    async estimate(src, ts) {
      if (image) detector.reset();
      const poses = await detector.estimatePoses(src, { flipHorizontal: false }, image ? undefined : ts);
      const p = poses && poses[0];
      if (!p) return { kps: null, box: null, score: 0 };
      const kps = p.keypoints.map((k) => [k.x, k.y, k.score ?? 0]);
      return { kps, box: null, score: meanScore(kps) };
    },
    reset() { detector.reset(); },
    dispose() { try { detector.dispose(); } catch { /* нет */ } },
  };
}

// ---------------------------------------------------------------- ONNX Runtime Web
let ortModule = null;
export async function getOrt() {
  if (!ortModule) {
    ortModule = await import("../vendor/ort/ort.all.min.mjs");
    const env = ortModule.env;
    env.wasm.wasmPaths = abs("vendor/ort/");
    // многопоточность доступна только при crossOriginIsolated (заголовки COOP/COEP сервера)
    env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    env.wasm.proxy = false;
    env.logLevel = "error";
  }
  return ortModule;
}

async function ortSession(buf, backend) {
  const ort = await getOrt();
  return ort.InferenceSession.create(buf, { executionProviders: [backend], graphOptimizationLevel: "all" });
}

function ortMeta(backend) {
  return {
    runtime_version: ortModule.env.versions ? ortModule.env.versions.web : "",
    backend_actual: backend,
    threads: backend === "wasm" ? ortModule.env.wasm.numThreads : null,
    score_source: "score",
  };
}

const canvases = {};
function ctx2d(key, w, h) {
  let c = canvases[key];
  if (!c) { c = document.createElement("canvas"); canvases[key] = c; }
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  return c.getContext("2d", { willReadFrequently: true });
}

// RGBA -> CHW float32; order "rgb"|"bgr"; mean/std в порядке каналов на выходе (null — без нормировки)
function toCHW(ctx, w, h, order, mean, std, scale = 1) {
  const d = ctx.getImageData(0, 0, w, h).data;
  const n = w * h, out = new Float32Array(3 * n);
  const ci = order === "bgr" ? [2, 1, 0] : [0, 1, 2];
  for (let c = 0; c < 3; c++) {
    const src = ci[c], m = mean ? mean[c] : 0, s = std ? std[c] : 1, o = c * n;
    for (let p = 0, i = src; p < n; p++, i += 4) out[o + p] = ((d[i] * scale) - m) / s;
  }
  return out;
}

// letterbox в левый верхний угол с фоном 114 (MMDeploy Resize keep_ratio + Pad)
function letterboxTL(src, size, key) {
  const { w, h } = srcSize(src);
  const ratio = Math.min(size / h, size / w);
  const nw = Math.floor(w * ratio), nh = Math.floor(h * ratio);
  const ctx = ctx2d(key, size, size);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "rgb(114,114,114)"; ctx.fillRect(0, 0, size, size);
  ctx.drawImage(src, 0, 0, nw, nh);
  return { ctx, ratio };
}

// letterbox по центру (Ultralytics LetterBox, auto=False)
function letterboxCenter(src, size, key) {
  const { w, h } = srcSize(src);
  const r = Math.min(size / h, size / w);
  const nw = Math.round(w * r), nh = Math.round(h * r);
  const dw = (size - nw) / 2, dh = (size - nh) / 2;
  const left = Math.round(dw - 0.1), top = Math.round(dh - 0.1);
  const ctx = ctx2d(key, size, size);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "rgb(114,114,114)"; ctx.fillRect(0, 0, size, size);
  ctx.drawImage(src, left, top, nw, nh);
  return { ctx, r, left, top };
}

function bestDet(dets, ratio, thr) {
  const d = dets.data, n = dets.dims[1], k = dets.dims[2];
  let best = -1, bs = -Infinity;
  for (let i = 0; i < n; i++) { const s = d[i * k + 4]; if (s > bs) { bs = s; best = i; } }
  if (best < 0 || bs < thr) return null;
  return { box: [d[best * k] / ratio, d[best * k + 1] / ratio, d[best * k + 2] / ratio, d[best * k + 3] / ratio], score: bs, index: best };
}

// RTMDet-nano (детектор человека): BGR, mean/std в порядке BGR, без перевода в RGB (pipeline.json)
async function makeRtmDet(buf, backend) {
  const ort = await getOrt();
  const sess = await ortSession(buf, backend);
  const size = 320;
  return {
    sess,
    async detect(src) {
      const { ctx, ratio } = letterboxTL(src, size, "det");
      const x = toCHW(ctx, size, size, "bgr", [103.53, 116.28, 123.675], [57.375, 57.12, 58.395]);
      const o = await sess.run({ [sess.inputNames[0]]: new ort.Tensor("float32", x, [1, 3, size, size]) });
      return bestDet(o.dets || o[sess.outputNames[0]], ratio, DET_THR);
    },
  };
}

function poseToBox(kps, expansion = 1.25) {
  let pts = kps.filter((p) => p[2] >= 0.3);
  if (pts.length < 4) pts = kps;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x1 = Math.min(...xs), y1 = Math.min(...ys), x2 = Math.max(...xs), y2 = Math.max(...ys);
  const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
  return [cx - (cx - x1) * expansion, cy - (cy - y1) * expansion, cx + (x2 - cx) * expansion, cy + (y2 - cy) * expansion];
}

// RTMPose: детектор раз в N кадров (rtmlib PoseTracker), между ними — бокс по точкам прошлого кадра
async function createRtmPose(model, backend, opts, loader, all) {
  const ort = await getOrt();
  const det = all.find((m) => m.id === (model.detector || "rtmdet-nano"));
  if (!det || !det.ready) throw new Error("нет весов детектора " + model.detector);
  const detBuf = await loader.get(abs(`models/${det.id}/${det.files.find((f) => f.endsWith(".onnx"))}`));
  const poseBuf = await loader.get(abs(`models/${model.id}/${model.files.find((f) => f.endsWith(".onnx"))}`));
  const t0 = now();
  const detector = await makeRtmDet(detBuf, backend);
  const sess = await ortSession(poseBuf, backend);
  const initMs = now() - t0;
  const W = 192, H = 256, aspect = W / H;
  const image = opts.mode === "image";
  const N = image ? 1 : Math.max(1, opts.detInterval || 1);
  let frame = 0, lastBox = null;

  async function pose(src, box) {
    const [x1, y1, x2, y2] = box;
    const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
    let sw = (x2 - x1) * 1.25, sh = (y2 - y1) * 1.25;
    if (sw > sh * aspect) sh = sw / aspect; else sw = sh * aspect;
    const s = W / sw;
    const ctx = ctx2d("pose", W, H);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "rgb(0,0,0)"; ctx.fillRect(0, 0, W, H);   // cv2.warpAffine: граница 0
    ctx.setTransform(s, 0, 0, s, W / 2 - s * cx, H / 2 - s * cy);
    ctx.drawImage(src, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    // pipeline.json: Normalize to_rgb=true, mean/std в порядке RGB
    const x = toCHW(ctx, W, H, "rgb", [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]);
    const o = await sess.run({ [sess.inputNames[0]]: new ort.Tensor("float32", x, [1, 3, H, W]) });
    const sx = o.simcc_x || o[sess.outputNames[0]], sy = o.simcc_y || o[sess.outputNames[1]];
    const K = sx.dims[1], WX = sx.dims[2], HY = sy.dims[2];
    const kps = [];
    for (let k = 0; k < K; k++) {
      let bx = 0, by = 0, mx = -Infinity, my = -Infinity;
      for (let i = 0, o2 = k * WX; i < WX; i++) { const v = sx.data[o2 + i]; if (v > mx) { mx = v; bx = i; } }
      for (let i = 0, o2 = k * HY; i < HY; i++) { const v = sy.data[o2 + i]; if (v > my) { my = v; by = i; } }
      const score = 0.5 * (mx + my);
      const px = bx / 2.0, py = by / 2.0;          // simcc_split_ratio = 2
      kps.push([px / W * sw + cx - sw / 2, py / H * sh + cy - sh / 2, score]);
    }
    return kps;
  }

  return {
    initMs,
    meta: { ...ortMeta(backend), detector: det.id, det_interval: N },
    async estimate(src) {
      let box = null, detScore = null;
      if (N <= 1 || frame % N === 0 || !lastBox) {
        const d = await detector.detect(src);
        if (d) { box = d.box; detScore = d.score; }
      } else box = lastBox;
      frame++;
      if (!box) { lastBox = null; return { kps: null, box: null, score: 0 }; }
      const kps = await pose(src, box);
      const sc = meanScore(kps);
      lastBox = sc >= 0.3 ? poseToBox(kps) : null;   // потеря — детектор на следующем кадре
      return { kps, box, score: sc, det_score: detScore };
    },
    reset() { frame = 0; lastBox = null; },
    dispose() { try { sess.release(); detector.sess.release(); } catch { /* нет */ } },
  };
}

// RTMO-s: одноэтапная, BGR без нормировки, letterbox в угол
async function createRtmo(model, backend, opts, loader) {
  const ort = await getOrt();
  const buf = await loader.get(abs(`models/${model.id}/${model.files.find((f) => f.endsWith(".onnx"))}`));
  const t0 = now();
  const sess = await ortSession(buf, backend);
  const initMs = now() - t0;
  const size = 640;
  return {
    initMs,
    meta: ortMeta(backend),
    async estimate(src) {
      const { ctx, ratio } = letterboxTL(src, size, "rtmo");
      const x = toCHW(ctx, size, size, "bgr", null, null);
      const o = await sess.run({ [sess.inputNames[0]]: new ort.Tensor("float32", x, [1, 3, size, size]) });
      const dets = o.dets || o[sess.outputNames[0]], kp = o.keypoints || o[sess.outputNames[1]];
      const b = bestDet(dets, ratio, DET_THR);
      if (!b) return { kps: null, box: null, score: 0 };
      const K = kp.dims[2], C = kp.dims[3], base = b.index * K * C;
      const kps = [];
      for (let k = 0; k < K; k++) {
        const i = base + k * C;
        kps.push([kp.data[i] / ratio, kp.data[i + 1] / ratio, kp.data[i + 2]]);
      }
      return { kps, box: b.box, score: meanScore(kps), det_score: b.score };
    },
    reset() {},
    dispose() { try { sess.release(); } catch { /* нет */ } },
  };
}

// YOLOv8/YOLO11-pose (Ultralytics ONNX, без NMS в графе): выход (1, 56, N) = cx,cy,w,h,conf,17*(x,y,c)
async function createYolo(model, backend, opts, loader) {
  const ort = await getOrt();
  const buf = await loader.get(abs(`models/${model.id}/${model.files.find((f) => f.endsWith(".onnx"))}`));
  const t0 = now();
  const sess = await ortSession(buf, backend);
  const initMs = now() - t0;
  const size = 640;
  return {
    initMs,
    meta: ortMeta(backend),
    async estimate(src) {
      const { ctx, r, left, top } = letterboxCenter(src, size, "yolo");
      const x = toCHW(ctx, size, size, "rgb", null, null, 1 / 255);
      const o = await sess.run({ [sess.inputNames[0]]: new ort.Tensor("float32", x, [1, 3, size, size]) });
      const out = o[sess.outputNames[0]], d = out.data, N = out.dims[2];
      // один человек: лучший по conf кандидат (после NMS он же остаётся первым)
      let best = -1, bs = -Infinity;
      for (let i = 0; i < N; i++) { const s = d[4 * N + i]; if (s > bs) { bs = s; best = i; } }
      if (best < 0 || bs < YOLO_THR) return { kps: null, box: null, score: 0 };
      const kps = [];
      for (let k = 0; k < 17; k++) {
        const c = 5 + 3 * k;
        kps.push([(d[c * N + best] - left) / r, (d[(c + 1) * N + best] - top) / r, d[(c + 2) * N + best]]);
      }
      const cx = d[best], cy = d[N + best], w = d[2 * N + best], h = d[3 * N + best];
      const box = [(cx - w / 2 - left) / r, (cy - h / 2 - top) / r, (cx + w / 2 - left) / r, (cy + h / 2 - top) / r];
      return { kps, box, score: meanScore(kps), det_score: bs };
    },
    reset() {},
    dispose() { try { sess.release(); } catch { /* нет */ } },
  };
}

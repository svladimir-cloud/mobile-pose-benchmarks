// Адаптер ONNX Runtime Web: RTMPose-t/-s (+ RTMDet-nano), RTMO-s, YOLOv8n-pose, YOLO11n-pose.
// Контракт — bench/adapters/registry.js. Протокол COCO-SinglePerson: на кадре один человек →
// двухэтапные модели берут один бокс с максимальным score, одноэтапные — экземпляр с максимальным score.
export const RUNTIME = 'onnxruntime-web';
export const RUNTIME_VERSION = '1.30.0';

const DIST = new URL('../../node_modules/onnxruntime-web/dist/', import.meta.url).href;
const ONNX = (f) => new URL(`../../models/cache/onnx/${f}`, import.meta.url).href;

// Пороги (зафиксированы, не подгонялись):
export const DET_SCORE_THR = 0.3;   // RTMDet: минимальный score бокса (иначе «человек не найден» → keypoints17 = null)
export const YOLO_CONF_THR = 0.25;  // как conf по умолчанию в ultralytics predict
export const YOLO_NMS_IOU = 0.7;    // как iou по умолчанию в ultralytics predict
export const RTMO_SCORE_THR = 0.1;  // score_thr из pipeline.json RTMO (YOLOXPoseAnnotationProcessor)
const PAD_VAL = 114;

const VARIANTS = {
  'rtmpose-t': { kind: 'rtmpose', files: ['rtmdet-nano.onnx', 'rtmpose-t.onnx'], inputSize: '256x192 (det 320x320)', scoreSource: 'det_score' },
  'rtmpose-s': { kind: 'rtmpose', files: ['rtmdet-nano.onnx', 'rtmpose-s.onnx'], inputSize: '256x192 (det 320x320)', scoreSource: 'det_score' },
  'rtmo-s': { kind: 'rtmo', files: ['rtmo-s.onnx'], inputSize: '640x640', scoreSource: 'det_score' },
  'yolov8n-pose': { kind: 'yolo', files: ['yolov8n-pose.onnx'], inputSize: '640x640', scoreSource: 'det_score' },
  'yolo11n-pose': { kind: 'yolo', files: ['yolo11n-pose.onnx'], inputSize: '640x640', scoreSource: 'det_score' },
};

// ---------- общие помощники ----------
function makeCanvas(w, h) {
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'low'; // билинейная (как cv2.INTER_LINEAR)
  return { c, ctx };
}
const srcSize = (im) => [im.naturalWidth ?? im.videoWidth ?? im.width, im.naturalHeight ?? im.videoHeight ?? im.height];

// RGBA → CHW float32. order: 'rgb' | 'bgr'; mean/std — в порядке каналов выходного тензора.
function toCHW(rgba, n, out, order, mean, std) {
  const [i0, i2] = order === 'bgr' ? [2, 0] : [0, 2];
  const p1 = n, p2 = 2 * n;
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    out[i] = (rgba[j + i0] - mean[0]) / std[0];
    out[p1 + i] = (rgba[j + 1] - mean[1]) / std[1];
    out[p2 + i] = (rgba[j + i2] - mean[2]) / std[2];
  }
}

// letterbox: изображение вписывается в size×size с сохранением пропорций, фон PAD_VAL.
// center=true — по центру (ultralytics), false — в левый верхний угол (rtmlib/mmdet).
function letterbox(cv, im, w, h, size, center) {
  const r = Math.min(size / h, size / w);
  const nw = Math.round(w * r), nh = Math.round(h * r);
  let dx = 0, dy = 0;
  if (center) { dx = Math.round((size - nw) / 2 - 0.1); dy = Math.round((size - nh) / 2 - 0.1); }
  cv.ctx.fillStyle = `rgb(${PAD_VAL},${PAD_VAL},${PAD_VAL})`;
  cv.ctx.fillRect(0, 0, size, size);
  cv.ctx.drawImage(im, 0, 0, w, h, dx, dy, nw, nh);
  return { r, dx, dy };
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]), x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return u > 0 ? inter / u : 0;
}
// Жадный NMS по убыванию score. boxes: [{box:[x1,y1,x2,y2], score}], возвращает отобранные.
function nms(items, thr) {
  items.sort((p, q) => q.score - p.score);
  const keep = [];
  for (const it of items) if (keep.every((k) => iou(k.box, it.box) <= thr)) keep.push(it);
  return keep;
}

// ---------- RTMPose: SimCC ----------
const POSE_W = 192, POSE_H = 256, SIMCC_RATIO = 2.0, PAD_FACTOR = 1.25;
const RGB_MEAN = [123.675, 116.28, 103.53], RGB_STD = [58.395, 57.12, 57.375];

// bbox → центр и масштаб (TopDownGetBboxCenterScale: padding 1.25, фиксированное отношение сторон 192:256)
function bboxToCS(b) {
  const cx = (b[0] + b[2]) / 2, cy = (b[1] + b[3]) / 2;
  let w = (b[2] - b[0]) * PAD_FACTOR, h = (b[3] - b[1]) * PAD_FACTOR;
  const ar = POSE_W / POSE_H;
  if (w > h * ar) h = w / ar; else w = h * ar;
  return { cx, cy, w, h };
}

// SimCCLabelDecode: argmax по x/y, /2.0; score = min(max_x, max_y) (get_simcc_maximum в mmpose)
function decodeSimCC(sx, sy, K, Wx, Wy) {
  const pts = [];
  for (let k = 0; k < K; k++) {
    let bx = 0, mx = -Infinity, by = 0, my = -Infinity;
    for (let i = 0; i < Wx; i++) { const v = sx[k * Wx + i]; if (v > mx) { mx = v; bx = i; } }
    for (let i = 0; i < Wy; i++) { const v = sy[k * Wy + i]; if (v > my) { my = v; by = i; } }
    pts.push([bx / SIMCC_RATIO, by / SIMCC_RATIO, Math.min(mx, my)]);
  }
  return pts;
}

export async function createAdapter({ backend = 'wasm', variant = 'rtmpose-t', onStage = () => {}, numThreads, rgb = true, rtmoCenter = false } = {}) {
  const spec = VARIANTS[variant];
  if (!spec) throw new Error(`ort: неизвестный вариант ${variant}`);
  if (backend !== 'wasm' && backend !== 'webgpu') throw new Error(`ort: бэкенд ${backend} не поддерживается`);
  const cold = {};

  // --- импорт рантайма ---
  onStage('import');
  let t0 = performance.now();
  const bundle = backend === 'webgpu' ? 'ort.webgpu.min.mjs' : 'ort.wasm.min.mjs';
  const ort = await import(/* @vite-ignore */ DIST + bundle);
  ort.env.wasm.wasmPaths = DIST;
  if (numThreads) ort.env.wasm.numThreads = numThreads;
  cold.cold_import_ms = performance.now() - t0;

  // --- загрузка моделей ---
  onStage('model-start');
  t0 = performance.now();
  const modelFiles = spec.files.map(ONNX);
  const buffers = [];
  let total = 0;
  for (const u of modelFiles) {
    const resp = await fetch(u, { cache: 'no-store' });
    if (!resp.ok) throw new Error(`модель ${u}: HTTP ${resp.status}`);
    const buf = new Uint8Array(await resp.arrayBuffer());
    total += buf.byteLength;
    buffers.push(buf);
    onStage('model', { loaded: total, total: 0 });
  }
  cold.cold_load_ms = performance.now() - t0;

  // --- создание сессий ---
  onStage('init');
  t0 = performance.now();
  const sessions = [];
  try {
    for (const buf of buffers) {
      sessions.push(await ort.InferenceSession.create(buf, {
        executionProviders: [backend],
        graphOptimizationLevel: 'all',
      }));
    }
  } catch (e) {
    throw new Error(`ort/${backend}: не удалось создать сессию: ${e?.message ?? e}`);
  }
  cold.cold_init_ms = performance.now() - t0;

  const backendInfo = {
    executionProviders: [backend],
    numThreads: ort.env.wasm.numThreads,
    crossOriginIsolated: self.crossOriginIsolated ?? false,
    wasmSimd: ort.env.wasm.simd ?? null,
    graphOptimizationLevel: 'all',
    ortVersion: ort.env.versions?.common ?? null,
    bundle,
  };

  const T = (data, dims) => new ort.Tensor('float32', data, dims);
  let predict;

  // ================= RTMPose (RTMDet-nano + SimCC) =================
  if (spec.kind === 'rtmpose') {
    const [detS, poseS] = sessions;
    const detCv = makeCanvas(320, 320), poseCv = makeCanvas(POSE_W, POSE_H);
    const detBuf = new Float32Array(3 * 320 * 320), poseBuf = new Float32Array(3 * POSE_W * POSE_H);
    const order = rgb ? 'rgb' : 'bgr';
    predict = async (image) => {
      const [w, h] = srcSize(image);
      const tAll = performance.now();
      // 1) детектор: resize keep_ratio + pad 114 (слева сверху), BGR, mean/std OpenMMLab (порядок B,G,R)
      const { r } = letterbox(detCv, image, w, h, 320, false);
      toCHW(detCv.ctx.getImageData(0, 0, 320, 320).data, 320 * 320, detBuf, 'bgr',
        [103.53, 116.28, 123.675], [57.375, 57.12, 58.395]);
      const dOut = await detS.run({ input: T(detBuf, [1, 3, 320, 320]) });
      const dets = dOut.dets.data, nd = dOut.dets.dims[1];
      let bi = -1, bs = -Infinity;
      for (let i = 0; i < nd; i++) if (dets[i * 5 + 4] > bs) { bs = dets[i * 5 + 4]; bi = i; }
      const detMs = performance.now() - tAll;
      if (bi < 0 || bs < DET_SCORE_THR) {
        return { keypoints17: null, instance_score: null, width: w, height: h, timings: { infer_ms: performance.now() - tAll, det_ms: detMs } };
      }
      const box = [0, 1, 2, 3].map((k) => dets[bi * 5 + k] / r);
      // 2) поза: аффинное кадрирование (центр/масштаб, padding 1.25) в 192×256
      const tPose = performance.now();
      const { cx, cy, w: sw, h: sh } = bboxToCS(box);
      const s = POSE_W / sw; // = POSE_H / sh
      const ctx = poseCv.ctx;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, POSE_W, POSE_H);
      ctx.setTransform(s, 0, 0, s, POSE_W / 2 - cx * s, POSE_H / 2 - cy * s);
      ctx.drawImage(image, 0, 0, w, h);
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      toCHW(ctx.getImageData(0, 0, POSE_W, POSE_H).data, POSE_W * POSE_H, poseBuf, order, RGB_MEAN, RGB_STD);
      const pOut = await poseS.run({ input: T(poseBuf, [1, 3, POSE_H, POSE_W]) });
      const sx = pOut.simcc_x.data, sy = pOut.simcc_y.data;
      const K = pOut.simcc_x.dims[1], Wx = pOut.simcc_x.dims[2], Wy = pOut.simcc_y.dims[2];
      const pts = decodeSimCC(sx, sy, K, Wx, Wy);
      // обратное преобразование: точка в кропе → пиксели исходника
      const keypoints17 = pts.map(([x, y, sc]) => [(x - POSE_W / 2) / s + cx, (y - POSE_H / 2) / s + cy, sc]);
      const inferMs = performance.now() - tAll;
      return { keypoints17, instance_score: bs, width: w, height: h,
        timings: { infer_ms: inferMs, det_ms: detMs, pose_ms: performance.now() - tPose }, raw: { box } };
    };

  // ================= YOLO-pose (ultralytics) =================
  } else if (spec.kind === 'yolo') {
    const [S] = sessions;
    const cv = makeCanvas(640, 640), buf = new Float32Array(3 * 640 * 640);
    const inName = S.inputNames[0], outName = S.outputNames[0];
    predict = async (image) => {
      const [w, h] = srcSize(image);
      const tAll = performance.now();
      const { r, dx, dy } = letterbox(cv, image, w, h, 640, true);
      toCHW(cv.ctx.getImageData(0, 0, 640, 640).data, 640 * 640, buf, 'rgb', [0, 0, 0], [255, 255, 255]);
      const out = (await S.run({ [inName]: T(buf, [1, 3, 640, 640]) }))[outName];
      const d = out.data, N = out.dims[2]; // [1,56,N], канал-мажорная раскладка
      const cand = [];
      for (let i = 0; i < N; i++) {
        const sc = d[4 * N + i];
        if (sc < YOLO_CONF_THR) continue;
        const cx = d[i], cy = d[N + i], bw = d[2 * N + i], bh = d[3 * N + i];
        cand.push({ i, score: sc, box: [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2] });
      }
      if (!cand.length) return { keypoints17: null, instance_score: null, width: w, height: h, timings: { infer_ms: performance.now() - tAll } };
      const best = nms(cand, YOLO_NMS_IOU)[0]; // для top-1 NMS не меняет результат, но выполняется как в ultralytics
      const keypoints17 = [];
      for (let k = 0; k < 17; k++) {
        const o = 5 + 3 * k;
        keypoints17.push([(d[o * N + best.i] - dx) / r, (d[(o + 1) * N + best.i] - dy) / r, d[(o + 2) * N + best.i]]);
      }
      return { keypoints17, instance_score: best.score, width: w, height: h,
        timings: { infer_ms: performance.now() - tAll }, raw: { n_candidates: cand.length } };
    };

  // ================= RTMO (bottom-up, NMS внутри ONNX) =================
  } else {
    const [S] = sessions;
    const cv = makeCanvas(640, 640), buf = new Float32Array(3 * 640 * 640);
    predict = async (image) => {
      const [w, h] = srcSize(image);
      const tAll = performance.now();
      const { r, dx, dy } = letterbox(cv, image, w, h, 640, rtmoCenter);
      // BottomupResize + Normalize(mean 0, std 1, to_rgb=false): BGR-порядок, значения 0..255
      toCHW(cv.ctx.getImageData(0, 0, 640, 640).data, 640 * 640, buf, 'bgr', [0, 0, 0], [1, 1, 1]);
      const o = await S.run({ input: T(buf, [1, 3, 640, 640]) });
      const dets = o.dets.data, nd = o.dets.dims[1], kp = o.keypoints.data;
      let bi = -1, bs = -Infinity;
      for (let i = 0; i < nd; i++) if (dets[i * 5 + 4] > bs) { bs = dets[i * 5 + 4]; bi = i; }
      if (bi < 0 || bs < RTMO_SCORE_THR) return { keypoints17: null, instance_score: null, width: w, height: h, timings: { infer_ms: performance.now() - tAll } };
      const keypoints17 = [];
      for (let k = 0; k < 17; k++) {
        const q = (bi * 17 + k) * 3;
        keypoints17.push([(kp[q] - dx) / r, (kp[q + 1] - dy) / r, kp[q + 2]]);
      }
      return { keypoints17, instance_score: bs, width: w, height: h, timings: { infer_ms: performance.now() - tAll }, raw: { n_dets: nd } };
    };
  }

  return {
    model: variant,
    variant,
    backend,
    runtime: RUNTIME,
    runtimeVersion: RUNTIME_VERSION,
    inputSize: spec.inputSize,
    modelBytes: total,
    modelFiles,
    scoreSource: spec.scoreSource,
    coldTimings: cold,
    backendInfo,
    predict,
    async close() { for (const s of sessions) await s.release?.(); },
  };
}

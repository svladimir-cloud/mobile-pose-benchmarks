// Адаптер BlazePose (MediaPipe Tasks Vision, Pose Landmarker), режим IMAGE.
// Интерфейс (REQUIREMENTS.md R1.3): init(backend) → predict(image) → {keypoints17, raw, timings}.
export const RUNTIME = '@mediapipe/tasks-vision';
export const RUNTIME_VERSION = '1.1.1';

// Таблица 4 статьи: индекс COCO → индекс BlazePose.
// Глаз — центральная точка (2, 5); «левый/правый» — анатомические стороны человека.
export const COCO_FROM_BLAZEPOSE = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];

const local = (f) => new URL(`../../models/cache/${f}`, import.meta.url).href;
export const MODEL_URLS = {
  lite: local('pose_landmarker_lite.task'),
  full: local('pose_landmarker_full.task'),
  heavy: local('pose_landmarker_heavy.task'),
};

// visionUrl / wasmBase / modelUrl по умолчанию — из репозитория (стенд); страница для телефона
// передаёт адреса CDN (cdn.jsdelivr.net, storage.googleapis.com) явно.
// onStage(stage, info) — необязательный обратный вызов для индикации: 'import' | 'model-start' | 'model' {loaded, total} | 'init'.
export async function createAdapter({ backend = 'wasm', variant = 'lite', modelUrl, visionUrl, wasmBase, options = {}, onStage = () => {} } = {}) {
  const delegate = backend === 'webgl' ? 'GPU' : 'CPU';
  const timings = {};
  onStage('import');
  let t0 = performance.now();
  const { FilesetResolver, PoseLandmarker } = await import(
    visionUrl ?? new URL('../../node_modules/@mediapipe/tasks-vision/vision_bundle.mjs', import.meta.url).href);
  timings.cold_import_ms = performance.now() - t0; // загрузка JS-модуля рантайма (из кеша — если уже загружен)
  onStage('model-start');
  t0 = performance.now();
  const modelHref = modelUrl ?? MODEL_URLS[variant];
  const resp = await fetch(modelHref, { cache: 'no-store' });
  if (!resp.ok) throw new Error(`модель: HTTP ${resp.status}`);
  const total = Number(resp.headers.get('content-length')) || 0;
  const chunks = [];
  let loaded = 0;
  const reader = resp.body.getReader();
  onStage('model', { loaded, total });
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onStage('model', { loaded, total });
  }
  const modelBuffer = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { modelBuffer.set(c, off); off += c.byteLength; }
  timings.cold_load_ms = performance.now() - t0;

  onStage('init');
  t0 = performance.now();
  const fileset = await FilesetResolver.forVisionTasks(
    wasmBase ?? new URL('../../node_modules/@mediapipe/tasks-vision/wasm', import.meta.url).href);
  const landmarker = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetBuffer: modelBuffer, delegate },
    runningMode: 'IMAGE',
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    outputSegmentationMasks: false,
    ...options,
  });
  timings.cold_init_ms = performance.now() - t0;

  return {
    model: `blazepose-${variant}`,
    variant,
    backend,
    delegate,
    runtime: RUNTIME,
    runtimeVersion: RUNTIME_VERSION,
    inputSize: '256x256',
    modelBytes: modelBuffer.byteLength,
    modelFiles: [modelHref],
    scoreSource: 'mean_visibility',
    coldTimings: timings,
    // image: HTMLImageElement | ImageBitmap | HTMLCanvasElement | HTMLVideoElement
    predict(image) {
      const w = image.naturalWidth ?? image.videoWidth ?? image.width;
      const h = image.naturalHeight ?? image.videoHeight ?? image.height;
      const t = performance.now();
      const res = landmarker.detect(image); // синхронно: включает пред-/постобработку и чтение с GPU
      const inferMs = performance.now() - t;
      const lm = res.landmarks?.[0];
      if (!lm) return { keypoints17: null, instance_score: null, raw: null, width: w, height: h, timings: { infer_ms: inferMs } };
      const raw = lm.map((p) => [p.x * w, p.y * h, p.z, p.visibility ?? null, p.presence ?? null]);
      const keypoints17 = COCO_FROM_BLAZEPOSE.map((i) => [raw[i][0], raw[i][1], raw[i][3]]);
      // У BlazePose нет оценки человека: оценка экземпляра = среднее visibility по 17 точкам.
      const instance_score = keypoints17.reduce((a, p) => a + p[2], 0) / 17;
      return { keypoints17, instance_score, raw, width: w, height: h, timings: { infer_ms: inferMs } };
    },
    close() { landmarker.close(); },
  };
}

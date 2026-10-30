// Адаптер BlazePose (MediaPipe Tasks Vision, Pose Landmarker), режим IMAGE.
// Интерфейс (REQUIREMENTS.md R1.3): init(backend) → predict(image) → {keypoints17, raw, timings}.
export const RUNTIME = '@mediapipe/tasks-vision';
export const RUNTIME_VERSION = '1.1.1';

// Таблица 4 статьи: индекс COCO → индекс BlazePose.
// Глаз — центральная точка (2, 5); «левый/правый» — анатомические стороны человека.
export const COCO_FROM_BLAZEPOSE = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];

export const MODEL_URLS = {
  lite: 'models/cache/pose_landmarker_lite.task',
  full: 'models/cache/pose_landmarker_full.task',
  heavy: 'models/cache/pose_landmarker_heavy.task',
};

// visionUrl / wasmBase / modelUrl по умолчанию — из репозитория (стенд); страница для телефона
// передаёт адреса CDN (cdn.jsdelivr.net, storage.googleapis.com) явно.
export async function createAdapter({ backend = 'wasm', variant = 'lite', modelUrl, visionUrl, wasmBase, options = {} } = {}) {
  const delegate = backend === 'webgl' ? 'GPU' : 'CPU';
  const timings = {};
  let t0 = performance.now();
  const { FilesetResolver, PoseLandmarker } = await import(
    visionUrl ?? new URL('../../node_modules/@mediapipe/tasks-vision/vision_bundle.mjs', import.meta.url).href);
  timings.cold_import_ms = performance.now() - t0; // загрузка JS-модуля рантайма (из кеша — если уже загружен)
  t0 = performance.now();
  const resp = await fetch(modelUrl ?? MODEL_URLS[variant], { cache: 'no-store' });
  if (!resp.ok) throw new Error(`модель: HTTP ${resp.status}`);
  const modelBuffer = new Uint8Array(await resp.arrayBuffer());
  timings.cold_load_ms = performance.now() - t0;

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
    coldTimings: timings,
    // image: HTMLImageElement | ImageBitmap | HTMLCanvasElement | HTMLVideoElement
    predict(image) {
      const w = image.naturalWidth ?? image.videoWidth ?? image.width;
      const h = image.naturalHeight ?? image.videoHeight ?? image.height;
      const t = performance.now();
      const res = landmarker.detect(image); // синхронно: включает пред-/постобработку и чтение с GPU
      const inferMs = performance.now() - t;
      const lm = res.landmarks?.[0];
      if (!lm) return { keypoints17: null, raw: null, width: w, height: h, timings: { infer_ms: inferMs } };
      const raw = lm.map((p) => [p.x * w, p.y * h, p.z, p.visibility ?? null, p.presence ?? null]);
      const keypoints17 = COCO_FROM_BLAZEPOSE.map((i) => [raw[i][0], raw[i][1], raw[i][3]]);
      return { keypoints17, raw, width: w, height: h, timings: { infer_ms: inferMs } };
    },
    close() { landmarker.close(); },
  };
}

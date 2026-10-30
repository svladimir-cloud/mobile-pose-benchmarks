// Адаптер MoveNet SinglePose (Lightning/Thunder) на TensorFlow.js (graph-модели TF Hub v4 / Kaggle).
// Пред-/постобработка своя (пакет pose-detection не используется): центрированный pad до квадрата →
// resizeBilinear до 192/256 → int32 [1,H,W,3]; выход [1,1,17,3] = (y, x, score), нормировано к квадрату.
// Режим отдельных изображений: без кропа по предыдущему кадру. Порядок 17 точек MoveNet = порядок COCO.
// MoveNet SinglePose всегда возвращает одного человека → keypoints17 никогда не null.
export const RUNTIME = '@tensorflow/tfjs';
export const RUNTIME_VERSION = '4.22.0';

const nm = (f) => new URL(`../../node_modules/@tensorflow/${f}`, import.meta.url).href;
const SCRIPTS = {
  core: nm('tfjs/dist/tf.min.js'), // core + converter + cpu + webgl + layers + data
  wasm: nm('tfjs-backend-wasm/dist/tf-backend-wasm.min.js'),
  webgpu: nm('tfjs-backend-webgpu/dist/tf-backend-webgpu.min.js'),
};
const WASM_DIR = nm('tfjs-backend-wasm/dist/');
const MODEL_DIR = (variant) => new URL(`../../models/cache/tfjs/movenet-${variant}/`, import.meta.url).href;
const SIZES = { lightning: 192, thunder: 256 };

const loaded = new Map();
function loadScript(src) {
  if (!loaded.has(src)) {
    loaded.set(src, new Promise((ok, fail) => {
      const s = document.createElement('script');
      s.src = src;
      s.onload = ok;
      s.onerror = () => fail(new Error(`не загрузился скрипт ${src}`));
      document.head.append(s);
    }));
  }
  return loaded.get(src);
}

// onStage(stage, info): 'import' | 'model-start' | 'init'.
export async function createAdapter({ backend = 'wasm', variant = 'lightning', onStage = () => {} } = {}) {
  if (!SIZES[variant]) throw new Error(`tfjs: неизвестный вариант ${variant}`);
  if (!['wasm', 'webgl', 'webgpu'].includes(backend)) throw new Error(`tfjs: бэкенд ${backend} не поддерживается`);
  const S = SIZES[variant];
  const cold = {};

  onStage('import');
  let t0 = performance.now();
  await loadScript(SCRIPTS.core);
  if (backend === 'wasm' || backend === 'webgpu') await loadScript(SCRIPTS[backend]);
  cold.cold_import_ms = performance.now() - t0; // загрузка и разбор JS рантайма и бэкенда
  const tf = window.tf;

  onStage('init');
  t0 = performance.now();
  if (backend === 'wasm') tf.wasm.setWasmPaths(WASM_DIR);
  let ok = false;
  try { ok = await tf.setBackend(backend); await tf.ready(); } catch (e) {
    throw new Error(`tfjs: бэкенд ${backend} не поднялся: ${e.message ?? e}`);
  }
  if (!ok || tf.getBackend() !== backend) throw new Error(`tfjs: бэкенд ${backend} не поднялся (активен ${tf.getBackend()})`);
  cold.cold_init_ms = performance.now() - t0; // инициализация бэкенда (WASM-модуль / контекст WebGL / адаптер и устройство WebGPU); компиляция шейдеров — в первом инференсе

  onStage('model-start');
  t0 = performance.now();
  const dir = MODEL_DIR(variant);
  const modelUrl = dir + 'model.json';
  const manifest = await (await fetch(modelUrl, { cache: 'no-store' })).json();
  const shardNames = manifest.weightsManifest.flatMap((g) => g.paths);
  const modelFiles = [modelUrl, ...shardNames.map((p) => dir + p)];
  const model = await tf.loadGraphModel(modelUrl);
  cold.cold_load_ms = performance.now() - t0;
  let modelBytes = 0;
  for (const u of modelFiles) {
    const r = await fetch(u, { method: 'HEAD', cache: 'no-store' });
    modelBytes += Number(r.headers.get('content-length')) || 0;
  }

  const flag = (k) => { try { return tf.env().get(k); } catch { return undefined; } };
  const backendInfo = { backend: tf.getBackend(), flags: {} };
  const names = { wasm: ['WASM_HAS_SIMD_SUPPORT', 'WASM_HAS_MULTITHREAD_SUPPORT'],
    webgl: ['WEBGL_VERSION', 'WEBGL_RENDER_FLOAT32_CAPABLE', 'WEBGL_FORCE_F16_TEXTURES'], webgpu: ['WEBGPU_CPU_FORWARD'] }[backend];
  for (const k of names) backendInfo.flags[k] = flag(k);

  return {
    model: `movenet-${variant}`,
    variant,
    backend,
    runtime: RUNTIME,
    runtimeVersion: RUNTIME_VERSION,
    inputSize: `${S}x${S}`,
    modelBytes,
    modelFiles,
    scoreSource: 'mean_keypoint_score',
    coldTimings: cold,
    backendInfo,
    // image: HTMLImageElement | ImageBitmap | HTMLCanvasElement | HTMLVideoElement
    async predict(image) {
      const w = image.naturalWidth ?? image.videoWidth ?? image.width;
      const h = image.naturalHeight ?? image.videoHeight ?? image.height;
      const t = performance.now();
      const side = Math.max(w, h);
      const padX = (side - w) / 2, padY = (side - h) / 2;
      const input = tf.tidy(() => {
        const px = tf.browser.fromPixels(image, 3).expandDims(0).toFloat(); // [1,h,w,3]
        const top = Math.floor(padY), bottom = side - h - top, left = Math.floor(padX), right = side - w - left;
        const padded = tf.pad(px, [[0, 0], [top, bottom], [left, right], [0, 0]]);
        return tf.image.resizeBilinear(padded, [S, S]).toInt();
      });
      let out;
      try { out = model.execute(input); } finally { input.dispose(); }
      const data = await out.data(); // [1,1,17,3] = y, x, score
      out.dispose();
      const left = Math.floor(padX), top = Math.floor(padY);
      const kp = [];
      let sum = 0;
      for (let i = 0; i < 17; i++) {
        const y = data[i * 3], x = data[i * 3 + 1], sc = data[i * 3 + 2];
        kp.push([x * side - left, y * side - top, sc]);
        sum += sc;
      }
      return { keypoints17: kp, instance_score: sum / 17, width: w, height: h, timings: { infer_ms: performance.now() - t } };
    },
    memory() { return tf.memory(); },
    close() { model.dispose(); },
  };
}

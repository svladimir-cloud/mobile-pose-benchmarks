// Реестр моделей стенда: id модели → модуль адаптера, параметры, поддерживаемые бэкенды.
// Контракт адаптера (REQUIREMENTS.md R1.3):
//   createAdapter({ backend, onStage, ...options }) → {
//     model, variant, backend, runtime, runtimeVersion, inputSize, modelBytes, modelFiles: [url],
//     scoreSource, coldTimings: { cold_import_ms?, cold_load_ms, cold_init_ms },
//     async predict(image) → { keypoints17: [[x, y, score]] | null, instance_score, width, height,
//                               timings: { infer_ms, ... }, raw?, instances? },
//     close() }
// Координаты — пиксели исходного изображения; keypoints17 — один (лучший) человек в порядке COCO.
// infer_ms — от входа predict() до готовых чисел на CPU (пред-/постобработка и чтение с GPU включены).
export const REGISTRY = {
  'blazepose-lite': { module: './mediapipe.js', options: { variant: 'lite' }, backends: ['wasm', 'webgl'] },
  'blazepose-full': { module: './mediapipe.js', options: { variant: 'full' }, backends: ['wasm', 'webgl'] },
  'blazepose-heavy': { module: './mediapipe.js', options: { variant: 'heavy' }, backends: ['wasm', 'webgl'] },
  'movenet-lightning': { module: './tfjs.js', options: { variant: 'lightning' }, backends: ['wasm', 'webgl', 'webgpu'] },
  'movenet-thunder': { module: './tfjs.js', options: { variant: 'thunder' }, backends: ['wasm', 'webgl', 'webgpu'] },
  // MoveNet, два прохода: второй — по кропу вокруг точек первого. В статье на COCO — рядом с однопроходным
  // (сопоставимость с публикацией RTMPose, Табл. 5); решение автора 31.10.2026.
  'movenet-lightning-refine': { module: './tfjs.js', options: { variant: 'lightning', refine: true }, backends: ['wasm'] },
  'movenet-thunder-refine': { module: './tfjs.js', options: { variant: 'thunder', refine: true }, backends: ['wasm'] },
  'rtmpose-t': { module: './ort.js', options: { variant: 'rtmpose-t' }, backends: ['wasm', 'webgpu'] },
  'rtmpose-s': { module: './ort.js', options: { variant: 'rtmpose-s' }, backends: ['wasm', 'webgpu'] },
  'rtmo-s': { module: './ort.js', options: { variant: 'rtmo-s' }, backends: ['wasm', 'webgpu'] },
  'yolov8n-pose': { module: './ort.js', options: { variant: 'yolov8n-pose' }, backends: ['wasm', 'webgpu'] },
  'yolo11n-pose': { module: './ort.js', options: { variant: 'yolo11n-pose' }, backends: ['wasm', 'webgpu'] },
};

export async function loadAdapter(model, backend, extra = {}) {
  const entry = REGISTRY[model];
  if (!entry) throw new Error(`неизвестная модель: ${model}`);
  if (!entry.backends.includes(backend)) throw new Error(`${model}: бэкенд ${backend} не поддерживается адаптером`);
  const mod = await import(entry.module);
  return mod.createAdapter({ backend, ...entry.options, ...extra });
}

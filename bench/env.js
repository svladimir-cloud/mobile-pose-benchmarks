// Сбор сведений об окружении браузера (R4.3, R5.1).
// Вызовы, которые на части устройств могут зависнуть, ограничены по времени.
const withTimeout = (p, ms) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), ms))]);

export async function probeEnv() {
  const env = {
    user_agent: navigator.userAgent,
    hardware_concurrency: navigator.hardwareConcurrency ?? null,
    device_memory_gb: navigator.deviceMemory ?? null,
    cross_origin_isolated: self.crossOriginIsolated ?? false,
    screen: `${screen.width}x${screen.height}@${devicePixelRatio}`,
  };
  try {
    const uad = await withTimeout(navigator.userAgentData?.getHighEntropyValues?.(['platform', 'platformVersion', 'model', 'fullVersionList']) ?? Promise.resolve(null), 3000);
    if (uad) env.ua_data = uad;
  } catch (e) { /* нет в Safari/Firefox */ }
  // WASM SIMD: минимальный модуль с инструкцией v128 (из wasm-feature-detect).
  const simd = new Uint8Array([0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,10,10,1,8,0,65,0,253,15,253,98,11]);
  env.wasm_simd = WebAssembly.validate(simd);
  env.wasm_threads = typeof SharedArrayBuffer !== 'undefined' && env.cross_origin_isolated;
  try {
    const c = document.createElement('canvas');
    const gl = c.getContext('webgl2');
    env.webgl2 = !!gl;
    if (gl) {
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      env.webgl_vendor = ext ? gl.getParameter(ext.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR);
      env.webgl_renderer = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    }
  } catch (e) { env.webgl2 = false; }
  env.webgpu = false;
  try {
    if (navigator.gpu) {
      const a = await withTimeout(navigator.gpu.requestAdapter(), 3000);
      env.webgpu = !!a;
      if (a?.info) env.webgpu_adapter = `${a.info.vendor} ${a.info.architecture} ${a.info.description}`.trim();
    }
  } catch (e) { env.webgpu_error = String(e?.message ?? e); }
  return env;
}

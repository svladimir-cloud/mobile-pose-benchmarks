// Возможности браузера и сведения об устройстве (R4.3, R5.1).

const SIMD_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);

export async function detectCapabilities() {
  const caps = {
    cross_origin_isolated: !!self.crossOriginIsolated,
    wasm_simd: false,
    wasm_threads: typeof SharedArrayBuffer !== "undefined" && !!self.crossOriginIsolated,
    webgl2: false, webgpu: false, webgpu_adapter: "no-api", webgpu_fallback: false,
    request_video_frame_callback: "requestVideoFrameCallback" in HTMLVideoElement.prototype,
  };
  try { caps.wasm_simd = WebAssembly.validate(SIMD_PROBE); } catch { /* нет WASM */ }
  try { caps.webgl2 = !!document.createElement("canvas").getContext("webgl2"); } catch { /* нет */ }
  if (navigator.gpu) {
    try {
      const ad = await navigator.gpu.requestAdapter();
      if (ad) {
        caps.webgpu = true;
        const info = ad.info || (ad.requestAdapterInfo ? await ad.requestAdapterInfo() : {});
        caps.webgpu_adapter = [info.vendor, info.architecture].filter(Boolean).join("/") || "adapter";
        // программный адаптер (SwiftShader) — WebGPU формально есть, но без GPU
        caps.webgpu_fallback = !!(info.isFallbackAdapter || ad.isFallbackAdapter || /swiftshader/i.test(caps.webgpu_adapter));
      } else caps.webgpu_adapter = "no-adapter";
    } catch (e) { caps.webgpu_adapter = "error: " + e.message; }
  }
  return caps;
}

function parseBrowser(ua) {
  const rules = [
    [/YaBrowser\/([\d.]+)/, "Yandex"], [/SamsungBrowser\/([\d.]+)/, "Samsung Internet"],
    [/EdgA?\/([\d.]+)/, "Edge"], [/OPR\/([\d.]+)/, "Opera"], [/Firefox\/([\d.]+)/, "Firefox"],
    [/CriOS\/([\d.]+)/, "Chrome iOS"], [/Chrome\/([\d.]+)/, "Chrome"], [/Version\/([\d.]+).*Safari/, "Safari"],
  ];
  for (const [re, name] of rules) { const m = ua.match(re); if (m) return { browser: name, browser_version: m[1] }; }
  return { browser: "unknown", browser_version: "" };
}

export async function detectDevice() {
  const ua = navigator.userAgent;
  const dev = {
    ua, ua_model: "", platform: "", platform_version: "", ...parseBrowser(ua),
    mobile: /Android|iPhone|iPad|Mobile/i.test(ua),
    hardware_concurrency: navigator.hardwareConcurrency || null,
    device_memory_gb: navigator.deviceMemory || null,
    screen: [screen.width, screen.height], dpr: devicePixelRatio,
    battery: null, user_label: "", temp_c_start: null, soc: "",
  };
  const m = ua.match(/Android ([\d.]+)/) || ua.match(/OS (\d+[_\d]*) like Mac/);
  if (m) { dev.platform = /Android/.test(ua) ? "Android" : "iOS"; dev.platform_version = m[1].replace(/_/g, "."); }
  if (navigator.userAgentData && navigator.userAgentData.getHighEntropyValues) {
    try {
      const h = await navigator.userAgentData.getHighEntropyValues(["model", "platformVersion", "fullVersionList"]);
      dev.ua_model = h.model || "";
      dev.platform = navigator.userAgentData.platform || dev.platform;
      if (h.platformVersion) dev.platform_version = h.platformVersion;
      const full = (h.fullVersionList || []).find((b) => /Chrome|Yandex|Edge|Samsung/i.test(b.brand) && !/Chromium/.test(b.brand));
      if (full && dev.browser === "Chrome") dev.browser_version = full.version;
    } catch { /* нет UA-CH */ }
  }
  dev.battery = await readBattery();
  return dev;
}

export async function readBattery() {
  try {
    if (!navigator.getBattery) return null;
    const b = await navigator.getBattery();
    return { level: Math.round(b.level * 100) / 100, charging: b.charging };
  } catch { return null; }
}

export function heapMb() {
  return performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1e5) / 10 : null;
}

// Общие функции: статистика, загрузка моделей с замером, ключевые точки.

export const COCO17 = [
  "nose", "left_eye", "right_eye", "left_ear", "right_ear",
  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_hip", "right_hip",
  "left_knee", "right_knee", "left_ankle", "right_ankle",
];
// BlazePose 33 -> COCO-17 (Таблица 4 статьи; глаз = центральная точка 2 и 5)
export const MP33_TO_COCO17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
export const SKELETON = [
  [5, 7], [7, 9], [6, 8], [8, 10], [5, 6], [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16], [0, 1], [0, 2], [1, 3], [2, 4],
];

export const now = () => performance.now();
// Пути относительно страницы: работает и на своём сервере (/), и на GitHub Pages (/<репозиторий>/)
export const abs = (p) => new URL(p.replace(/^\//, ""), document.baseURI).href;
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export function quantile(sorted, q) {
  if (!sorted.length) return null;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export function summarize(values) {
  const v = values.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return { mean: null, p50: null, p95: null, p99: null, n: 0 };
  const r = (x) => Math.round(x * 100) / 100;
  return {
    mean: r(v.reduce((a, b) => a + b, 0) / v.length),
    p50: r(quantile(v, 0.5)), p95: r(quantile(v, 0.95)), p99: r(quantile(v, 0.99)), n: v.length,
  };
}

export const round = (x, d = 2) => (Number.isFinite(x) ? Math.round(x * 10 ** d) / 10 ** d : null);

// Детерминированный генератор (mulberry32) — для воспроизводимой рандомизации порядка моделей
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function shuffle(arr, seed) {
  const r = rng(seed), a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

// Загрузка файла модели с замером времени и признаком «из кэша» (Resource Timing)
export async function fetchTimed(url) {
  const t0 = now();
  const resp = await fetch(url, { cache: "default" });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${url}`);
  const buffer = await resp.arrayBuffer();
  const ms = now() - t0;
  const abs = new URL(url, location.href).href;
  const entries = performance.getEntriesByName(abs);
  const e = entries[entries.length - 1];
  // через service worker (GitHub Pages) Resource Timing не различает кэш и сеть -> null
  const viaSw = !!(navigator.serviceWorker && navigator.serviceWorker.controller);
  const fromCache = e && !viaSw ? e.transferSize === 0 && e.decodedBodySize > 0 : null;
  return { buffer, ms, bytes: buffer.byteLength, fromCache, url };
}

export class Loader {
  constructor() { this.files = []; }
  async get(url) {
    const r = await fetchTimed(url);
    this.files.push(r);
    return r.buffer;
  }
  summary() {
    return {
      download_ms: round(this.files.reduce((a, f) => a + f.ms, 0), 1),
      download_bytes: this.files.reduce((a, f) => a + f.bytes, 0),
      from_cache: this.files.length ? this.files.every((f) => f.fromCache === true) : null,
      files: this.files.map((f) => ({ url: f.url, bytes: f.bytes })),
    };
  }
}

export function srcSize(src) {
  return {
    w: src.videoWidth || src.naturalWidth || src.width,
    h: src.videoHeight || src.naturalHeight || src.height,
  };
}

export function meanScore(kps) {
  if (!kps) return 0;
  return kps.reduce((a, p) => a + (p[2] || 0), 0) / kps.length;
}

export function slug(s) {
  return String(s || "").replace(/[^A-Za-z0-9.-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40);
}

export function newId(prefix = "") {
  const d = new Date();
  const p = (n, w = 2) => String(n).padStart(w, "0");
  return `${prefix}${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}-${Math.random().toString(36).slice(2, 6)}`;
}

export function loadScript(src) {
  return new Promise((res, rej) => {
    if ([...document.scripts].some((s) => s.src.endsWith(src))) return res();
    const s = document.createElement("script");
    s.src = src; s.onload = res; s.onerror = () => rej(new Error("не загрузился " + src));
    document.head.appendChild(s);
  });
}

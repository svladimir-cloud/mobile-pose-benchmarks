// Проверка стенда в headless Chromium (env=desktop-headless: числа скорости — НЕ результаты).
//   python bench/server.py --http --port 8080 &   затем   node tools/smoke.mjs [query]
// query по умолчанию — короткий прогон всех моделей на WASM по тестовому ролику.
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
let chromium;
try { ({ chromium } = require("playwright")); } catch { ({ chromium } = require("/opt/node22/lib/node_modules/playwright")); }

const base = process.env.MPB_URL || "http://localhost:8080/";
const query = process.argv[2] || "auto=1&mode=custom&backend=wasm&source=video:/media/openpose-demo.mp4&warm=5/5&meas=30/10&pause=0&env=desktop-headless&label=cloud-headless";
const timeoutMin = Number(process.env.MPB_TIMEOUT_MIN || 20);

const browser = await chromium.launch({ args: ["--autoplay-policy=no-user-gesture-required", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist"] });
const page = await browser.newPage();
const errors = [];
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
page.on("pageerror", (e) => errors.push(String(e)));
await page.goto(base + "?" + query);
await page.waitForFunction(() => window.__mpbDone === true, null, { timeout: timeoutMin * 60000 });
const rows = await page.evaluate(() => window.__mpb.results.map((d) => ({
  model: d.model.id, backend: d.model.backend_actual || d.model.backend_requested, threads: d.model.threads, status: d.status,
  error: d.error, n: d.timing && d.timing.n_measure, p50: d.timing && d.timing.latency_ms && d.timing.latency_ms.p50,
  fps: d.timing && d.timing.model_fps, dropped: d.timing && d.timing.dropped_frames_pct,
  found: (d.frames || d.predictions || []).filter((f) => f.kps).length + "/" + (d.frames || d.predictions || []).length,
  coi: d.capabilities.cross_origin_isolated,
})));
console.table(rows);
if (errors.length) console.log("ошибки консоли:\n" + [...new Set(errors)].slice(0, 20).join("\n"));
await browser.close();

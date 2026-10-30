// Раннер Playwright: headless Chromium, прогоны BlazePose на изображениях COCO.
// Каждый бэкенд — в новом процессе браузера (холодный старт). Результат → results/raw/*.json,
// строка → logs/runs.md.
//
//   node bench/runner.mjs --ids eval/ids_sub300_seed20261030.txt --backends wasm,webgl \
//       [--variant lite] [--warmup 50] [--measure 200] [--latency-images 20]
import { createRequire } from 'node:module';
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { startServer } from './server.mjs';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');
const pwVersion = require('playwright/package.json').version;

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => {
  if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1]]);
  return acc;
}, []));
const idsFile = args.ids ?? 'eval/ids_sub300_seed20261030.txt';
const backends = (args.backends ?? 'wasm,webgl').split(',');
const variant = args.variant ?? 'lite';
const nWarmup = Number(args.warmup ?? 50);
const nMeasure = Number(args.measure ?? 200);
const nLatImages = Number(args['latency-images'] ?? 20);
const device = args.device ?? 'cloud-desktop';
const env = 'desktop-headless';

const ids = readFileSync(resolve(ROOT, idsFile), 'utf8').trim().split('\n').map(Number);
const urls = ids.map((id) => `/datasets/coco/val2017/${String(id).padStart(12, '0')}.jpg`);
const modelPath = `models/cache/pose_landmarker_${variant}.task`;
const modelSha = createHash('sha256').update(readFileSync(resolve(ROOT, modelPath))).digest('hex');

const LAUNCH = {
  wasm: [],
  // Программный WebGL (SwiftShader) — в облаке нет GPU.
  webgl: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
};

const server = await startServer(0);
const base = `http://127.0.0.1:${server.address().port}`;
mkdirSync(resolve(ROOT, 'results/raw'), { recursive: true });

for (const backend of backends) {
  const date = new Date().toISOString();
  const runId = `${date.slice(0, 10).replaceAll('-', '')}-${date.slice(11, 19).replaceAll(':', '')}_${device}_blazepose-${variant}_${backend}`;
  const browser = await chromium.launch({ args: LAUNCH[backend] ?? [] });
  const page = await browser.newPage();
  const consoleLog = [];
  page.on('console', (m) => consoleLog.push(`${m.type()}: ${m.text()}`.slice(0, 500)));
  page.on('pageerror', (e) => consoleLog.push(`pageerror: ${e.message}`.slice(0, 500)));
  let status = 'ok';
  const record = {
    run_id: runId, date, env, device, model: `blazepose-${variant}`, variant, backend,
    model_file: modelPath, model_sha256: modelSha,
    model_url: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_${variant}/float16/latest/pose_landmarker_${variant}.task`,
    dataset: 'coco-single', ids_file: idsFile, n_images: ids.length,
    n_warmup: nWarmup, n_measure: nMeasure, latency_images: nLatImages,
    harness: { playwright: pwVersion, browser_version: browser.version(), launch_args: LAUNCH[backend] ?? [], node: process.version },
  };
  try {
    await page.goto(`${base}/bench/index.html`);
    await page.waitForFunction(() => window.bench && document.getElementById('status').dataset.ready === '1');
    record.browser_env = await page.evaluate(() => window.bench.probeEnv());
    const t0 = Date.now();
    const info = await page.evaluate((o) => window.bench.init(o), { backend, variant, modelUrl: `/${modelPath}` });
    record.runtime = info.runtime; record.runtime_version = info.runtimeVersion;
    record.delegate = info.delegate; record.input_size = info.inputSize; record.model_bytes = info.modelBytes;
    record.cold = { ...info.cold };
    record.cold.first_infer_ms = await page.evaluate((u) => window.bench.firstInfer(u), urls[0]);
    // Изображения — пачками, чтобы не держать один долгий evaluate.
    record.predictions = [];
    for (let i = 0; i < urls.length; i += 50) {
      const part = await page.evaluate(([u, d]) => window.bench.runImages(u, d), [urls.slice(i, i + 50), ids.slice(i, i + 50)]);
      record.predictions.push(...part);
      process.stdout.write(`\r${backend}: ${record.predictions.length}/${urls.length}`);
    }
    process.stdout.write('\n');
    record.latency_ms = await page.evaluate(([u, w, m]) => window.bench.runLatency(u, w, m), [urls.slice(0, nLatImages), nWarmup, nMeasure]);
    record.wall_s = (Date.now() - t0) / 1000;
  } catch (e) {
    status = `failed: ${String(e.message ?? e).split('\n')[0].slice(0, 200)}`;
  }
  record.status = status;
  record.console = consoleLog.slice(0, 200);
  await browser.close();
  const file = `results/raw/${runId}.json`;
  writeFileSync(resolve(ROOT, file), JSON.stringify(record));
  const n = record.predictions?.length ?? 0;
  appendFileSync(resolve(ROOT, 'logs/runs.md'),
    `| ${date.slice(0, 10)} | ${runId} | blazepose-${variant} | ${backend} | ${env} | ${n} изобр. + ${nWarmup}/${nMeasure} замеров | \`${file}\` | ${status} |\n`);
  console.log(runId, status);
}
server.close();

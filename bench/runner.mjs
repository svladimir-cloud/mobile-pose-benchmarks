// Раннер Playwright: headless Chromium, матрица «модель × бэкенд» на изображениях COCO.
// Каждая пара — в новом процессе браузера (холодный старт). Порядок пар перемешивается (seed).
// Результат → <out-dir>/<run_id>.json, строка → logs/runs.md (кроме --no-log).
//
//   node bench/runner.mjs --models blazepose-lite,movenet-lightning --backends wasm,webgl \
//       [--ids eval/ids_sub300_seed20261030.txt] [--limit N] [--warmup 50] [--measure 200] \
//       [--latency-images 20] [--seed 1] [--timeout-min 30] [--out-dir results/raw] [--no-log] [--device cloud-desktop]
// Бэкенд, не поддерживаемый адаптером, пропускается (пишется в лог консоли).
import { createRequire } from 'node:module';
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { startServer } from './server.mjs';
import { REGISTRY } from './adapters/registry.js';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');
const pwVersion = require('playwright/package.json').version;

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const argv = process.argv.slice(2);
const flag = (k) => argv.includes(`--${k}`);
const arg = (k, d) => { const i = argv.indexOf(`--${k}`); return i >= 0 && argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[i + 1] : d; };
const idsFile = arg('ids', 'eval/ids_sub300_seed20261030.txt');
const models = arg('models', arg('model', 'blazepose-lite')).split(',');
const backends = arg('backends', 'wasm,webgl').split(',');
const nWarmup = Number(arg('warmup', 50));
const nMeasure = Number(arg('measure', 200));
const nLatImages = Number(arg('latency-images', 20));
const device = arg('device', 'cloud-desktop');
const seed = Number(arg('seed', 1));
const timeoutMs = Number(arg('timeout-min', 30)) * 60000;
const outDir = resolve(ROOT, arg('out-dir', 'results/raw'));
const noLog = flag('no-log');
const env = 'desktop-headless';

let ids = readFileSync(resolve(ROOT, idsFile), 'utf8').trim().split('\n').map(Number);
if (arg('limit')) ids = ids.slice(0, Number(arg('limit')));
const urls = ids.map((id) => `/datasets/coco/val2017/${String(id).padStart(12, '0')}.jpg`);

const LAUNCH = {
  wasm: [],
  // Программный GPU (SwiftShader) — в облаке нет GPU.
  webgl: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
  webgpu: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=swiftshader', '--use-webgpu-adapter=swiftshader',
    '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
};

// Перемешивание пар (Фишер — Йейтс с LCG), чтобы порядок не влиял систематически.
function shuffle(a, s) {
  let x = s >>> 0;
  const rnd = () => ((x = (x * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}
const sha256File = (p) => createHash('sha256').update(readFileSync(p)).digest('hex');
const withTimeout = (p, ms, what) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(`timeout ${ms / 60000} мин: ${what}`)), ms))]);

const server = await startServer(0);
const base = `http://127.0.0.1:${server.address().port}`;
mkdirSync(outDir, { recursive: true });
for (const m of models) if (!REGISTRY[m]) throw new Error(`модели ${m} нет в bench/adapters/registry.js`);
const all = models.flatMap((m) => backends.map((b) => [m, b]));
for (const [m, b] of all) if (!REGISTRY[m].backends.includes(b)) console.log(`${m}/${b}: не поддерживается адаптером — пропуск`);
const pairs = shuffle(all.filter(([m, b]) => REGISTRY[m].backends.includes(b)), seed);
console.log('порядок:', pairs.map((p) => p.join('/')).join(', '));

for (const [model, backend] of pairs) {
  const date = new Date().toISOString();
  const runId = `${date.slice(0, 10).replaceAll('-', '')}-${date.slice(11, 19).replaceAll(':', '')}_${device}_${model}_${backend}`;
  const browser = await chromium.launch({ args: LAUNCH[backend] ?? [] });
  const page = await browser.newPage();
  const consoleLog = [];
  page.on('console', (m) => consoleLog.push(`${m.type()}: ${m.text()}`.slice(0, 500)));
  page.on('pageerror', (e) => consoleLog.push(`pageerror: ${e.message}`.slice(0, 500)));
  let status = 'ok';
  const record = {
    run_id: runId, date, env, device, model, backend,
    dataset: 'coco-single', ids_file: idsFile, n_images: ids.length,
    n_warmup: nWarmup, n_measure: nMeasure, latency_images: nLatImages, order_seed: seed,
    harness: { playwright: pwVersion, browser_version: browser.version(), launch_args: LAUNCH[backend] ?? [], node: process.version },
  };
  const t0 = Date.now();
  try {
    await page.goto(`${base}/bench/index.html`);
    await page.waitForFunction(() => window.bench && document.getElementById('status').dataset.ready === '1');
    record.browser_env = await page.evaluate(() => window.bench.probeEnv());
    const work = (async () => {
      const info = await page.evaluate((o) => window.bench.init(o), { model, backend });
      Object.assign(record, { variant: info.variant, runtime: info.runtime, runtime_version: info.runtimeVersion,
        delegate: info.delegate, input_size: info.inputSize, model_bytes: info.modelBytes,
        score_source: info.scoreSource, backend_info: info.backendInfo, cold: { ...info.cold } });
      record.model_files = info.modelFiles.map((u) => {
        const p = resolve(ROOT, '.' + new URL(u, base).pathname);
        return existsSync(p) ? { file: relative(ROOT, p), sha256: sha256File(p) } : { url: u };
      });
      record.cold.first_infer_ms = await page.evaluate((u) => window.bench.firstInfer(u), urls[0]);
      record.predictions = [];
      for (let i = 0; i < urls.length; i += 50) {
        const part = await page.evaluate(([u, d]) => window.bench.runImages(u, d), [urls.slice(i, i + 50), ids.slice(i, i + 50)]);
        record.predictions.push(...part);
        process.stdout.write(`\r${model}/${backend}: ${record.predictions.length}/${urls.length}`);
      }
      process.stdout.write('\n');
      record.latency_ms = await page.evaluate(([u, w, m]) => window.bench.runLatency(u, w, m), [urls.slice(0, nLatImages), nWarmup, nMeasure]);
    })();
    await withTimeout(work, timeoutMs, `${model}/${backend}`);
  } catch (e) {
    status = `failed: ${String(e.message ?? e).split('\n')[0].slice(0, 200)}`;
  }
  record.wall_s = (Date.now() - t0) / 1000;
  record.status = status;
  record.console = consoleLog.slice(0, 200);
  // close() у браузера с флагами WebGPU иногда не возвращается — ограничиваем по времени.
  await withTimeout(browser.close(), 30000, 'browser.close').catch(() => { try { browser.process()?.kill('SIGKILL'); } catch {} });
  const file = resolve(outDir, `${runId}.json`);
  writeFileSync(file, JSON.stringify(record));
  const n = record.predictions?.length ?? 0;
  if (!noLog) {
    appendFileSync(resolve(ROOT, 'logs/runs.md'),
      `| ${date.slice(0, 10)} | ${runId} | ${model} | ${backend} | ${env} | ${n} изобр. + ${nWarmup}/${nMeasure} замеров | \`${relative(ROOT, file)}\` | ${status} |\n`);
  }
  console.log(runId, status);
}
server.close();

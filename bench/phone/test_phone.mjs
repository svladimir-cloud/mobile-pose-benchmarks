// Проверка страницы для телефона в headless Chromium (эмуляция Pixel 7). Не замер телефона!
// Запросы к CDN перехватываются и отдаются из node_modules/ и models/cache/ (в облаке Chromium не ходит в сеть).
//   node bench/phone/test_phone.mjs [--backend wasm|webgl] [--measure 20]
import { createRequire } from 'node:module';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { startServer } from '../server.mjs';

const require = createRequire(import.meta.url);
const { chromium, devices } = require('playwright');
const ROOT = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const backend = arg('backend', 'wasm');
const server = await startServer(0);
const base = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({ args: backend === 'webgl' ? ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] : [] });
const ctx = await browser.newContext({ ...devices['Pixel 7'], acceptDownloads: true });
const page = await ctx.newPage();
const served = [];
await page.route(/cdn\.jsdelivr\.net\/npm\/@mediapipe\/tasks-vision@[\d.]+\/(.*)$/, (route) => {
  const rel = route.request().url().match(/tasks-vision@[\d.]+\/(.*)$/)[1];
  served.push(rel);
  const type = rel.endsWith('.wasm') ? 'application/wasm' : 'text/javascript';
  route.fulfill({ body: readFileSync(resolve(ROOT, 'node_modules/@mediapipe/tasks-vision', rel)), contentType: type,
    headers: { 'Access-Control-Allow-Origin': '*', 'Cross-Origin-Resource-Policy': 'cross-origin' } });
});
await page.route(/storage\.googleapis\.com\/mediapipe-models\/.*\/(pose_landmarker_\w+\.task)$/, (route) => {
  served.push('model');
  route.fulfill({ body: readFileSync(resolve(ROOT, 'models/cache/pose_landmarker_lite.task')),
    headers: { 'Access-Control-Allow-Origin': '*', 'Cross-Origin-Resource-Policy': 'cross-origin' } });
});
page.on('pageerror', (e) => console.log('pageerror', e.message));
await page.goto(`${base}/bench/phone/dist/phone.html?measure=${arg('measure', 20)}&warmup=5`);
await page.fill('#device', 'test-headless');
await page.selectOption('#backend', backend);
await page.click('#run');
await page.waitForFunction(() => !document.getElementById('run').disabled && window.__phone.last, null, { timeout: 300000 });
const status = await page.textContent('#status');
const [dl] = await Promise.all([page.waitForEvent('download'), page.click('#csv')]);
const out = resolve(arg('out', '/tmp'), dl.suggestedFilename());
await dl.saveAs(out);
const csv = readFileSync(out, 'utf8').trim().split('\n');
console.log(status, '| CDN-запросы:', [...new Set(served)].join(' '), '| CSV:', out, csv.length - 1, 'строк');
console.log(csv[0]); console.log(csv[1]); console.log(csv.at(-1));
await browser.close(); server.close();

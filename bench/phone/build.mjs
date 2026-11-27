// Сборка одностраничного файла для телефона: адаптер + env.js + 20 изображений (base64) → bench/phone/dist/phone.html.
//   node bench/phone/build.mjs   (нужны изображения в datasets/coco/val2017/)
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const strip = (src) => src.replace(/^export /gm, '');
const adapter = strip(readFileSync(resolve(ROOT, 'bench/adapters/mediapipe.js'), 'utf8'));
const env = strip(readFileSync(resolve(ROOT, 'bench/env.js'), 'utf8'));
const meta = JSON.parse(readFileSync(resolve(ROOT, 'bench/phone/images.json'), 'utf8'));
const images = meta.map((m) => {
  const buf = readFileSync(resolve(ROOT, 'datasets/coco/val2017', m.file));
  return { image_id: m.image_id, license: m.license, flickr_url: m.flickr_url,
    sha256: createHash('sha256').update(buf).digest('hex'), data: `data:image/jpeg;base64,${buf.toString('base64')}` };
});
const version = createHash('sha256').update(adapter + env + readFileSync(resolve(ROOT, 'bench/phone/template.html'))).digest('hex').slice(0, 10);
const html = readFileSync(resolve(ROOT, 'bench/phone/template.html'), 'utf8')
  .replace('/*__ENV__*/', () => env)
  .replace('/*__ADAPTER__*/', () => adapter)
  .replace('/*__IMAGES__*/', () => JSON.stringify(images))
  .replace('/*__VERSION__*/', () => version);
mkdirSync(resolve(ROOT, 'bench/phone/dist'), { recursive: true });
writeFileSync(resolve(ROOT, 'bench/phone/dist/phone.html'), html);
console.log(`bench/phone/dist/phone.html: ${(html.length / 1e6).toFixed(2)} МБ, версия ${version}`);

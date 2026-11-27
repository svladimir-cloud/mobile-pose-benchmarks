// Копирует браузерные рантаймы из node_modules в bench/web/vendor/ (раздаются своим сервером,
// без CDN: фиксированные версии из package-lock.json и честный замер холодного старта).
//   npm ci && npm run vendor
import { cpSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const nm = join(root, "node_modules");
const out = join(root, "bench", "web", "vendor");
rmSync(out, { recursive: true, force: true });

const versions = {};
function copy(pkg, files, dest) {
  const base = join(nm, pkg);
  versions[pkg] = JSON.parse(readFileSync(join(base, "package.json"), "utf8")).version;
  mkdirSync(join(out, dest), { recursive: true });
  for (const f of files) {
    const name = f.split("/").pop();
    cpSync(join(base, f), join(out, dest, name));
  }
}
function copyMatching(pkg, dir, re, dest) {
  const files = readdirSync(join(nm, pkg, dir)).filter((f) => re.test(f)).map((f) => `${dir}/${f}`);
  copy(pkg, files, dest);
}

// ONNX Runtime Web: сборка "all" (WASM + WebGL + WebGPU) и все варианты wasm
copyMatching("onnxruntime-web", "dist", /^(ort\.all\.min\.mjs|ort-wasm-simd-threaded.*\.(mjs|wasm))$/, "ort");
// MediaPipe Tasks Vision
copy("@mediapipe/tasks-vision", ["vision_bundle.mjs"], "mediapipe");
copyMatching("@mediapipe/tasks-vision", "wasm", /\.(js|wasm)$/, "mediapipe/wasm");
// TF.js (UMD, глобальный tf) + бэкенды + pose-detection
copy("@tensorflow/tfjs-core", ["dist/tf-core.min.js"], "tfjs");
copy("@tensorflow/tfjs-converter", ["dist/tf-converter.min.js"], "tfjs");
copy("@tensorflow/tfjs-backend-webgl", ["dist/tf-backend-webgl.min.js"], "tfjs");
copy("@tensorflow/tfjs-backend-webgpu", ["dist/tf-backend-webgpu.min.js"], "tfjs");
copy("@tensorflow/tfjs-backend-wasm", ["dist/tf-backend-wasm.min.js"], "tfjs");
copyMatching("@tensorflow/tfjs-backend-wasm", "dist", /\.wasm$/, "tfjs/wasm");
copy("@tensorflow-models/pose-detection", ["dist/pose-detection.min.js"], "tfjs");

writeFileSync(join(out, "versions.json"), JSON.stringify(versions, null, 2));
console.log("vendor ->", out, versions);

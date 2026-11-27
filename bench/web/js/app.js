// Управление страницей: выбор режима, очередь конфигураций, отправка результатов.
import { detectCapabilities, detectDevice, readBattery } from "./caps.js";
import { BACKENDS, preferredBackends } from "./models.js";
import { DEFAULT_PARAMS, VideoSource, runFrames, runImages, runRealtime } from "./runner.js";
import { newId, round, shuffle, sleep } from "./util.js";

const $ = (id) => document.getElementById(id);
const FPS_TARGET = 10;   // ориентир цели статьи; окончательный порог — eval/fmin.py
const QS = new URLSearchParams(location.search);

const state = { cfg: null, caps: null, device: null, versions: {}, results: [], running: false, stop: false, wakeLock: null };

async function init() {
  const [cfg, caps, device, versions] = await Promise.all([
    fetch("/api/config").then((r) => r.json()),
    detectCapabilities(), detectDevice(),
    fetch("/vendor/versions.json").then((r) => (r.ok ? r.json() : {})).catch(() => ({})),
  ]);
  Object.assign(state, { cfg, caps, device, versions });
  $("devLabel").value = localStorage.getItem("mpb.label") || device.ua_model || "";
  $("pSeed").value = String(Math.floor(Math.random() * 1e6));
  renderCaps();
  renderSources();
  renderModels();
  for (const id of ["mode", "source"]) $(id).addEventListener("change", updateMode);
  $("start").onclick = start;
  $("stop").onclick = () => { state.stop = true; $("stop").disabled = true; };
  $("download").onclick = download;
  $("share").onclick = share;
  updateMode();
  if (QS.get("auto") === "1") autostart();
}

function renderCaps() {
  const c = state.caps, d = state.device;
  const yes = (v) => (v ? '<b class="ok">да</b>' : '<b class="bad">нет</b>');
  $("caps").innerHTML = [
    `${d.platform || "?"} ${d.platform_version || ""} · ${d.browser} ${d.browser_version}`,
    `ядер: ${d.hardware_concurrency ?? "?"}`, `ОЗУ: ${d.device_memory_gb ?? "?"} ГБ`,
    `изоляция (многопоточность): ${yes(c.cross_origin_isolated)}`, `WASM SIMD: ${yes(c.wasm_simd)}`,
    `WebGL2: ${yes(c.webgl2)}`, `WebGPU: ${yes(c.webgpu)}${c.webgpu ? " (" + c.webgpu_adapter + ")" : ""}`,
    d.battery ? `заряд: ${Math.round(d.battery.level * 100)}%${d.battery.charging ? " (заряжается!)" : ""}` : "",
  ].filter(Boolean).map((s) => `<span>${s}</span>`).join("");
}

function renderSources() {
  const sel = $("source");
  sel.innerHTML = "";
  for (const m of state.cfg.media.filter((m) => m.kind === "video")) sel.add(new Option(`видео: ${m.name}`, "video:" + m.url));
  sel.add(new Option("камера (фронтальная)", "camera:user"));
  sel.add(new Option("камера (основная)", "camera:environment"));
  for (const m of state.cfg.media.filter((m) => m.kind === "frames")) sel.add(new Option(`кадры: ${m.name} (${m.count})`, "frames:" + m.url + ":" + m.name));
}

function browserModels() { return state.cfg.models.filter((m) => m.role === "browser"); }

function renderModels() {
  const box = $("models");
  box.innerHTML = "";
  for (const m of browserModels()) {
    const l = document.createElement("label");
    l.innerHTML = `<input type="checkbox" value="${m.id}" ${m.ready ? "checked" : "disabled"}> ${m.id}${m.ready ? ` <span class="muted">${(m.bytes / 1e6).toFixed(0)} МБ</span>` : ' <span class="muted">нет весов</span>'}`;
    box.appendChild(l);
  }
}

function updateMode() {
  const mode = $("mode").value, src = $("source").value;
  const manual = mode === "custom" || mode === "long";
  $("modelsBox").style.display = manual || mode === "frames" || mode === "images" ? "" : "none";
  $("backendBox").style.display = manual ? "" : "none";
  const bsel = $("backend");
  bsel.innerHTML = '<option value="auto">лучший доступный</option>';
  for (const [v, t] of [["all-cpu", "CPU для всех (MediaPipe cpu, остальные WASM)"], ["all-gpu", "GPU/WebGL для всех"], ["all-webgpu", "WebGPU (MediaPipe — gpu)"], ["gpu", "gpu"], ["cpu", "cpu"], ["webgl", "webgl"], ["wasm", "wasm"], ["webgpu", "webgpu"]]) bsel.add(new Option(t, v));
  const hints = {
    quick: "Каждая модель запускается на лучшем доступном бэкенде (WebGPU/WebGL/GPU, иначе WASM). Если бэкенд не сработал — пробуется следующий. Телефон без зарядки, экран не гасить, вкладку не сворачивать.",
    full: "Все доступные бэкенды для каждой модели. Долго: запускайте на охлаждённом телефоне.",
    long: "Одна модель 10 минут подряд — проверка троттлинга. С камерой: телефон на подставке, человек выполняет упражнения в кадре.",
    frames: "Все кадры тестового ролика по порядку — для оценки точности контроля упражнения (повторы, амплитуда).",
    images: "Изображения COCO по одному — для оценки точности на открытом наборе. Лучше запускать на ПК.",
    custom: "Отметьте модели и бэкенд.",
  };
  $("hint").textContent = hints[mode] || "";
  if (mode === "frames" && !src.startsWith("frames:")) {
    const opt = [...$("source").options].find((o) => o.value.startsWith("frames:"));
    if (opt) $("source").value = opt.value;
  }
}

function readParams() {
  const pair = (s) => s.split("/").map((x) => parseFloat(x.trim()));
  const [wf, ws] = pair($("pWarm").value), [mf, ms] = pair($("pMeas").value);
  return {
    ...DEFAULT_PARAMS,
    warmup_frames: wf || 50, warmup_max_s: ws || 15, measure_frames: mf || 200, measure_max_s: ms || 40,
    det_interval: parseInt($("pDet").value, 10) || 1, long_minutes: parseFloat($("pLong").value) || 10,
    seed: parseInt($("pSeed").value, 10) || 1, repeats: parseInt($("pRep").value, 10) || 1,
    pause_s: parseFloat($("pPause").value) || 0, coco_limit: parseInt($("pCoco").value, 10) || 0,
  };
}

function selectedModels() {
  return [...$("models").querySelectorAll("input:checked")].map((i) => browserModels().find((m) => m.id === i.value));
}

// очередь конфигураций для режима
function buildPlan(mode, params) {
  const ready = browserModels().filter((m) => m.ready);
  const caps = state.caps;
  const plan = [];
  const forced = $("backend").value;
  if (mode === "quick") for (const m of ready) plan.push({ model: m, backends: preferredBackends(m, caps), fallback: true });
  else if (mode === "full") for (const m of ready) for (const b of BACKENDS[m.runtime]) plan.push({ model: m, backends: [b] });
  else {
    const ms = selectedModels();
    for (const m of ms) {
      const pref = preferredBackends(m, caps);
      // «все-cpu» / «все-webgl» — общий выбор для разных рантаймов
      const alias = { "all-cpu": { "mediapipe-tasks-vision": "cpu", tfjs: "wasm", "onnxruntime-web": "wasm" },
        "all-gpu": { "mediapipe-tasks-vision": "gpu", tfjs: "webgl", "onnxruntime-web": "webgl" },
        "all-webgpu": { "mediapipe-tasks-vision": "gpu", tfjs: "webgpu", "onnxruntime-web": "webgpu" } };
      const f = alias[forced] ? alias[forced][m.runtime] : forced;
      const bs = forced === "auto" ? pref : (BACKENDS[m.runtime].includes(f) ? [f] : null);
      if (bs) plan.push({ model: m, backends: bs, fallback: forced === "auto" });
    }
  }
  let out = [];
  for (let r = 0; r < params.repeats; r++) out = out.concat(shuffle(plan, params.seed + r).map((p, i) => ({ ...p, repeat: r + 1, order_index: i })));
  return out;
}

async function acquireWakeLock() {
  try { if ("wakeLock" in navigator) state.wakeLock = await navigator.wakeLock.request("screen"); } catch { state.wakeLock = null; }
}
document.addEventListener("visibilitychange", () => { if (!document.hidden && state.running) acquireWakeLock(); });

async function start() {
  if (state.running) return;
  const mode = $("mode").value, params = readParams();
  const plan = buildPlan(mode, params);
  if (!plan.length) { $("status").textContent = "Нет моделей для запуска."; return; }
  localStorage.setItem("mpb.label", $("devLabel").value.trim());
  state.running = true; state.stop = false;
  $("start").disabled = true; $("stop").disabled = false;
  await acquireWakeLock();
  const device = { ...state.device, battery: await readBattery(), user_label: $("devLabel").value.trim(),
    year: parseInt($("devYear").value, 10) || null, temp_c_start: parseFloat($("devTemp").value) || null, note: $("devNote").value.trim() };
  const ctx = {
    batchId: newId("b"), newId: () => newId(), env: QS.get("env") || (device.mobile ? "phone" : (navigator.webdriver ? "desktop-headless" : "desktop")),
    device, caps: state.caps, versions: state.versions, models: state.cfg.models, wakeLock: state.wakeLock,
  };
  const ui = {
    overlay: $("overlay"),
    phase: (t) => { $("phase").textContent = t; },
    progress: (p) => {
      $("status").textContent = p.total
        ? `${p.processed} / ${p.total} · ${p.lat.toFixed(0)} мс`
        : `${p.processed} кадров · ${p.elapsed.toFixed(0)} с · ${p.fps.toFixed(1)} кадр/с · ${p.lat.toFixed(0)} мс`;
    },
  };
  const [kind, ...rest] = $("source").value.split(":");
  const source = new VideoSource($("video"));
  try {
    if (mode === "images") {
      let items = state.cfg.coco.items;
      if (params.coco_limit) items = items.slice(0, params.coco_limit);
      if (!items.length) throw new Error("на сервере нет изображений COCO (python eval/coco_select.py && python eval/coco_download.py)");
      for (const [i, p] of plan.entries()) {
        if (state.stop) break;
        await runOne(p, i, plan.length, (b) => runImages(ctx, p.model, b, items, params, ui));
      }
    } else if (mode === "frames" || kind === "frames") {
      const [url, name] = [rest.slice(0, -1).join(":"), rest[rest.length - 1]];
      if (kind !== "frames") throw new Error("выберите источник «кадры» (tools/extract_frames.py)");
      for (const [i, p] of plan.entries()) {
        if (state.stop) break;
        await runOne(p, i, plan.length, (b) => runFrames(ctx, p.model, b, { url, name }, params, ui));
      }
    } else {
      await source.open(kind === "camera" ? { kind, facing: rest[0] } : { kind, url: rest.join(":") });
      const m = mode === "long" ? "long" : "speed";
      for (const [i, p] of plan.entries()) {
        if (state.stop) break;
        await runOne(p, i, plan.length, (b) => runRealtime(ctx, p.model, b, source, { ...params, order_index: p.order_index, repeat: p.repeat }, m, ui));
        if (params.pause_s && i < plan.length - 1) await sleep(params.pause_s * 1000);
      }
    }
    $("phase").textContent = state.stop ? "остановлено" : "готово";
  } catch (e) {
    $("status").textContent = "Ошибка: " + e.message;
  } finally {
    source.close();
    state.running = false;
    $("start").disabled = false; $("stop").disabled = true;
    try { if (state.wakeLock) await state.wakeLock.release(); } catch { /* нет */ }
    window.__mpbDone = true;
  }
}

async function runOne(p, i, n, fn) {
  for (const b of p.backends) {
    if (state.stop) return;
    $("status").textContent = `${i + 1}/${n}: ${p.model.id} · ${b} · загрузка…`;
    const doc = await fn(b);
    doc.params.order_index = p.order_index; doc.params.repeat = p.repeat; doc.params.seed = readParams().seed;
    addResult(doc);
    await send(doc);
    if (doc.status === "ok" || !p.fallback) return;
  }
}

function addResult(doc) {
  state.results.push(doc);
  const t = doc.timing || {}, c = doc.cold_start || {};
  const fps = t.e2e_fps ?? (t.latency_ms && t.latency_ms.mean ? 1000 / t.latency_ms.mean : null);
  let verdict;
  if (doc.status !== "ok") verdict = `<span class="bad">ошибка</span> <span class="muted">${escapeHtml(doc.error || "").slice(0, 80)}</span>`;
  else if (fps == null) verdict = "—";
  else verdict = fps >= FPS_TARGET ? `<span class="ok">≥ ${FPS_TARGET} кадр/с</span>` : `<span class="warn">&lt; ${FPS_TARGET} кадр/с</span>`;
  const startS = c.download_ms != null ? (c.download_ms + c.init_ms + c.first_infer_ms) / 1000 : null;
  const tr = document.createElement("tr");
  tr.innerHTML = `<td>${doc.model.id}</td><td>${doc.model.backend_actual || doc.model.backend_requested}${doc.model.threads > 1 ? " ×" + doc.model.threads : ""}</td>
    <td class="num">${fmt(fps, 1)}</td><td class="num">${fmt(t.latency_ms && t.latency_ms.p50, 0)}</td><td class="num">${fmt(t.latency_ms && t.latency_ms.p95, 0)}</td>
    <td class="num">${fmt(t.dropped_frames_pct, 0)}</td><td class="num">${fmt(startS, 1)}</td><td>${verdict}</td>`;
  $("results").querySelector("tbody").appendChild(tr);
}

const fmt = (v, d) => (v == null || !Number.isFinite(v) ? "—" : v.toFixed(d));
const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let sent = 0, unsent = 0;
async function send(doc) {
  try {
    const r = await fetch("/api/result", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(doc) });
    if (!r.ok) throw new Error("HTTP " + r.status);
    sent++;
  } catch { unsent++; }
  $("sendState").textContent = `Отправлено на сервер: ${sent}` + (unsent ? ` · не отправлено: ${unsent} — скачайте файл` : "");
}

function bundle() {
  const name = `mpb_${(state.results[0] && state.results[0].batch_id) || newId()}.json`;
  return new File([JSON.stringify(state.results)], name, { type: "application/json" });
}
function download() {
  const f = bundle(), a = document.createElement("a");
  a.href = URL.createObjectURL(f); a.download = f.name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
async function share() {
  const f = bundle();
  if (navigator.canShare && navigator.canShare({ files: [f] })) {
    try { await navigator.share({ files: [f], title: "Результаты замеров" }); } catch { /* отмена */ }
  } else download();
}

// автозапуск по ссылке: ?auto=1&mode=quick&source=video:/media/x.mp4&models=a,b&backend=wasm&env=desktop-headless
function autostart() {
  if (QS.get("mode")) $("mode").value = QS.get("mode");
  updateMode();
  if (QS.get("backend")) $("backend").value = QS.get("backend");
  if (QS.get("source")) $("source").value = QS.get("source");
  if (QS.get("models")) {
    const want = QS.get("models").split(",");
    for (const i of $("models").querySelectorAll("input")) i.checked = want.includes(i.value) && !i.disabled;
  }
  for (const [k, id] of [["warm", "pWarm"], ["meas", "pMeas"], ["det", "pDet"], ["long", "pLong"], ["coco", "pCoco"], ["seed", "pSeed"], ["pause", "pPause"], ["label", "devLabel"]]) {
    if (QS.get(k)) $(id).value = QS.get(k);
  }
  start();
}

window.__mpb = state;
init().catch((e) => { $("caps").textContent = "Ошибка инициализации: " + e.message; });

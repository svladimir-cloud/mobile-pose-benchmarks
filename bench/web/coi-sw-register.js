// На статическом хостинге (GitHub Pages) нельзя задать заголовки COOP/COEP, а без них нет
// crossOriginIsolated -> нет SharedArrayBuffer -> нет многопоточного WASM. Service worker
// (coi-sw.js) добавляет эти заголовки к ответам; после первой регистрации страница один раз
// перезагружается. На своём сервере (bench/server.py) заголовки уже есть, и скрипт ничего не делает.
(() => {
  if (window.crossOriginIsolated || !window.isSecureContext || !("serviceWorker" in navigator)) return;
  if (sessionStorage.getItem("mpb.coiReload")) return;   // не зацикливаться, если браузер не поддерживает
  navigator.serviceWorker.register("coi-sw.js").then((reg) => {
    const reload = () => { sessionStorage.setItem("mpb.coiReload", "1"); location.reload(); };
    if (reg.active && !navigator.serviceWorker.controller) reload();
    else navigator.serviceWorker.addEventListener("controllerchange", reload, { once: true });
  }).catch((e) => console.warn("service worker не зарегистрирован:", e));
})();

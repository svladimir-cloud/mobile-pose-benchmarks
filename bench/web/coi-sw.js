// Добавляет заголовки изоляции к ответам (см. coi-sw-register.js). Все ресурсы стенда — с того же сайта.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.cache === "only-if-cached" && req.mode !== "same-origin") return;
  // страница стенда v1 (bench/phone/) грузит модель с другого домена — изоляция ей помешает
  if (new URL(req.url).pathname.includes("/bench/phone/")) return;
  e.respondWith(fetch(req).then((res) => {
    if (res.status === 0) return res;   // opaque-ответ не трогаем
    const h = new Headers(res.headers);
    h.set("Cross-Origin-Embedder-Policy", "require-corp");
    h.set("Cross-Origin-Opener-Policy", "same-origin");
    h.set("Cross-Origin-Resource-Policy", "cross-origin");
    return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
  }));
});

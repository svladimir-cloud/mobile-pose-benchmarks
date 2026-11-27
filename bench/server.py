"""Сервер стенда: раздаёт страницу, рантаймы, модели, видео и принимает результаты.

  python bench/server.py                  # HTTPS на 8443 (самоподписанный сертификат с IP локальной сети)
  python bench/server.py --http --port 8080   # без TLS (для туннеля cloudflared/ngrok или localhost)

Заголовки Cross-Origin-Opener-Policy / Cross-Origin-Embedder-Policy включают
crossOriginIsolated -> SharedArrayBuffer -> многопоточный WASM (ONNX Runtime, TF.js).
Только стандартная библиотека + cryptography (для сертификата).
"""
from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import json
import mimetypes
import os
import re
import socket
import ssl
import sys
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "bench" / "web"
MODELS = ROOT / "models" / "cache"
MANIFEST = ROOT / "models" / "sources.json"
MEDIA = ROOT / "datasets" / "media"
COCO_IMG = ROOT / "datasets" / "coco" / "val2017"
COCO_LIST = ROOT / "datasets" / "coco" / "single_person_list.json"
RAW = ROOT / "results" / "raw"
RUNS_LOG = ROOT / "logs" / "runs.md"
CERT_DIR = ROOT / "bench" / ".cert"
MAX_POST = 64 * 1024 * 1024
_log_lock = threading.Lock()

for ext, mt in {".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm",
                ".onnx": "application/octet-stream", ".task": "application/octet-stream",
                ".bin": "application/octet-stream", ".json": "application/json", ".mp4": "video/mp4",
                ".webm": "video/webm", ".mov": "video/quicktime", ".jpg": "image/jpeg"}.items():
    mimetypes.add_type(mt, ext)

VIDEO_EXT = {".mp4", ".webm", ".mov", ".m4v"}


def lan_ips() -> list[str]:
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def ensure_cert(ips: list[str]) -> tuple[Path, Path]:
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    crt, key = CERT_DIR / "cert.pem", CERT_DIR / "key.pem"
    stamp = CERT_DIR / "ips.txt"
    want = ",".join(sorted(ips))
    if crt.exists() and key.exists() and stamp.exists() and stamp.read_text() == want:
        return crt, key
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    pk = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "mobile-pose-bench")])
    san = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    san += [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips]
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(pk.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(days=1))
            .not_valid_after(now + dt.timedelta(days=825))
            .add_extension(x509.SubjectAlternativeName(san), critical=False).sign(pk, hashes.SHA256()))
    key.write_bytes(pk.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                     serialization.NoEncryption()))
    crt.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    stamp.write_text(want)
    return crt, key


def list_media(media: Path | None = None) -> list[dict]:
    """Видео в datasets/media + наборы кадров datasets/media/<клип>/frames (после tools/extract_frames.py)."""
    media = media or MEDIA
    out = []
    if not media.exists():
        return out
    for p in sorted(media.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXT:
            out.append({"kind": "video", "name": p.stem, "url": f"media/{p.name}", "bytes": p.stat().st_size})
        elif p.is_dir() and (p / "frames.json").exists():
            meta = json.loads((p / "frames.json").read_text(encoding="utf-8"))
            out.append({"kind": "frames", "name": p.name, "url": f"media/{p.name}/frames.json",
                        "count": meta.get("count"), "fps": meta.get("fps"),
                        "width": meta.get("width"), "height": meta.get("height")})
    return out


def available_models() -> list[dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out = []
    for m in manifest["models"]:
        if m["role"] not in ("browser", "detector"):
            continue
        d = MODELS / m["id"]
        files = sorted(f.name for f in d.iterdir()) if d.exists() else []
        files = [f for f in files if not f.endswith(".pt")]
        ready = bool(files) and (("export" not in m) or m["export"]["out"] in files)
        out.append({k: m.get(k) for k in ("id", "family", "variant", "role", "runtime", "input_size",
                                           "keypoints", "detector", "license", "weights_note")}
                   | {"ready": ready, "files": files,
                      "bytes": sum((d / f).stat().st_size for f in files) if ready else 0})
    return out


def safe_child(base: Path, rel: str) -> Path | None:
    p = (base / rel).resolve()
    try:
        p.relative_to(base.resolve())
    except ValueError:
        return None
    return p


class Handler(SimpleHTTPRequestHandler):
    server_version = "mpb/1"

    def log_message(self, fmt, *args):  # короче стандартного
        if self.command == "POST" or "/api/" in self.path:
            sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    # ---------- маршрутизация ----------
    def route(self) -> Path | None:
        path = unquote(urlparse(self.path).path)
        if path in ("/", ""):
            return WEB / "index.html"
        for prefix, base in (("/models/", MODELS), ("/media/", MEDIA), ("/coco/", COCO_IMG)):
            if path.startswith(prefix):
                return safe_child(base, path[len(prefix):])
        return safe_child(WEB, path.lstrip("/"))

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/config":
            return self.send_json(self.config())
        if path == "/api/health":
            return self.send_json({"ok": True})
        target = self.route()
        if target is None or not target.is_file():
            return self.send_error(HTTPStatus.NOT_FOUND)
        self.send_file(target)

    def do_HEAD(self):
        target = self.route()
        if target is None or not target.is_file():
            return self.send_error(HTTPStatus.NOT_FOUND)
        self.send_file(target, head=True)

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/result":
            return self.send_error(HTTPStatus.NOT_FOUND)
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_POST:
            return self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        try:
            doc = json.loads(self.rfile.read(n))
            assert isinstance(doc, dict) and str(doc.get("schema", "")).startswith("mpb-result/")
        except Exception:  # noqa: BLE001
            return self.send_error(HTTPStatus.BAD_REQUEST, "ожидается JSON mpb-result")
        name = save_result(doc)
        self.send_json({"ok": True, "file": name})

    # ---------- ответы ----------
    def config(self) -> dict:
        coco = []
        if COCO_LIST.exists():
            items = json.loads(COCO_LIST.read_text(encoding="utf-8"))
            coco = [it for it in items if (COCO_IMG / it["file_name"]).exists()]
        return {"models": available_models(), "media": list_media(),
                "coco": {"count": len(coco), "items": coco}, "server_time": dt.datetime.now().isoformat()}

    def send_json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, target: Path, head: bool = False):
        size = target.stat().st_size
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        status = HTTPStatus.OK
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip())
            if not m:
                return self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            if m.group(1):
                start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
            elif m.group(2):
                start = max(0, size - int(m.group(2)))
            if start > end:
                return self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            status = HTTPStatus.PARTIAL_CONTENT
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        rel = target.as_posix()
        # модели/рантаймы кэшируются (повторный заход меряет «тёплый» старт), страница — нет
        if "/models/cache/" in rel or "/vendor/" in rel:
            self.send_header("Cache-Control", "public, max-age=604800")
        else:
            self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if head:
            return
        with target.open("rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", s or "unknown").strip("-")[:40] or "unknown"


def save_result(doc: dict) -> str:
    RAW.mkdir(parents=True, exist_ok=True)
    dev = doc.get("device", {})
    device = dev.get("user_label") or dev.get("ua_model") or dev.get("platform") or "device"
    model = doc.get("model", {})
    stamp = (doc.get("created_at") or dt.datetime.now(dt.timezone.utc).isoformat())[:19].replace("-", "").replace(":", "").replace("T", "-")
    name = "_".join([stamp, slug(device), slug(model.get("id", "model")), slug(model.get("backend_requested", "")),
                     slug(doc.get("mode", ""))]) + ".json"
    path = RAW / name
    i = 1
    while path.exists():
        path = RAW / name.replace(".json", f"_{i}.json")
        i += 1
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    t = doc.get("timing", {})
    line = (f"| {doc.get('created_at', '')[:19]} | {doc.get('run_id', '')} | {model.get('id', '')} | "
            f"{model.get('backend_actual') or model.get('backend_requested', '')} | {doc.get('env', '')} / {slug(device)} | "
            f"{doc.get('mode', '')} {doc.get('source', {}).get('name', '')} | {t.get('n_measure', len(doc.get('predictions') or doc.get('frames') or []))} | "
            f"results/raw/{path.name} | {doc.get('status', '')} |\n")
    with _log_lock:
        if not RUNS_LOG.exists() or "| дата |" not in RUNS_LOG.read_text(encoding="utf-8"):
            with RUNS_LOG.open("a", encoding="utf-8") as f:
                f.write("\n| дата | run_id | модель | бэкенд | окружение / устройство | режим, источник | замеров | файл | статус |\n"
                        "|---|---|---|---|---|---|---|---|---|\n")
        with RUNS_LOG.open("a", encoding="utf-8") as f:
            f.write(line)
    return path.name


def print_qr(url: str) -> None:
    try:
        import qrcode  # необязательная зависимость
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.print_ascii(invert=True)
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--http", action="store_true", help="без TLS (localhost или за туннелем HTTPS)")
    a = ap.parse_args()
    if not (WEB / "vendor").exists():
        print("нет bench/web/vendor — выполните: npm ci && npm run vendor", file=sys.stderr)
    ips = lan_ips()
    httpd = ThreadingHTTPServer((a.host, a.port), Handler)
    scheme = "http" if a.http else "https"
    if not a.http:
        crt, key = ensure_cert(ips)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(crt, key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    print(f"\nСтенд замеров: {scheme}://localhost:{a.port}/")
    for ip in ips:
        url = f"{scheme}://{ip}:{a.port}/"
        print(f"  с телефона (та же Wi-Fi сеть): {url}")
    if ips and not a.http:
        print_qr(f"{scheme}://{ips[0]}:{a.port}/")
    print("Результаты: results/raw/, журнал: logs/runs.md. Остановить: Ctrl+C\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

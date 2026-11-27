# Стенд на VPS: постоянная ссылка для телефонов

Модели работают на телефонах, сервер только раздаёт файлы и принимает результаты (JSON).

## Требования

| Ресурс | Сколько |
|---|---|
| ОЗУ | 512 МБ – 1 ГБ (сервер стенда ≈ 50–100 МБ); 4 ГБ — если на этом же сервере считать эталон и `eval/` |
| CPU | 1 vCPU (2 — для эталона) |
| Диск | 2–3 ГБ |
| Трафик | ≈ 200 МБ на телефон при первом заходе (веса всех моделей и рантаймы), повторно — из кэша браузера |
| Канал | от 50–100 Мбит/с, иначе холодный старт меряет сервер, а не телефон |
| Домен | обязателен: HTTPS с настоящим сертификатом (камера и многопоточный WASM работают только в защищённом контексте) |

## Установка (Ubuntu/Debian)

```bash
sudo apt install -y python3-venv nodejs npm git caddy
git clone -b claude/sleepy-hopper-ycca7l https://github.com/svladimir-cloud/mobile-pose-benchmarks /opt/mpb
cd /opt/mpb
python3 -m venv .venv
.venv/bin/pip install cryptography
npm ci && npm run vendor
```

Веса: либо `.venv/bin/pip install -r requirements.txt && .venv/bin/python models/fetch_models.py --role browser --skip-yolo`, либо скопировать готовую папку `models/cache/` с ПК (с YOLO — только так, если на сервере нет torch). Достаточно браузерных моделей и `rtmdet-nano`; `ref-*` на сервере не нужны.

Тестовое видео — в `datasets/media/` (mp4, H.264).

## Автозапуск (systemd)

`/etc/systemd/system/mpb.service`:

```ini
[Unit]
Description=mobile-pose-benchmarks bench server
After=network.target

[Service]
WorkingDirectory=/opt/mpb
ExecStart=/opt/mpb/.venv/bin/python bench/server.py --http --host 127.0.0.1 --port 8080
Restart=always
User=www-data

[Install]
WantedBy=multi-user.target
```

```bash
sudo chown -R www-data /opt/mpb/results /opt/mpb/logs
sudo systemctl enable --now mpb
```

## HTTPS (Caddy, сертификат Let's Encrypt автоматически)

`/etc/caddy/Caddyfile`:

```
bench.example.ru {
    encode gzip
    request_body {
        max_size 64MB
    }
    reverse_proxy 127.0.0.1:8080
}
```

```bash
sudo systemctl reload caddy
```

Заголовки COOP/COEP ставит сам `bench/server.py`, Caddy их передаёт. Проверка: на странице в блоке «Устройство» должно быть «изоляция (многопоточность): да».

Ссылка для участников: `https://bench.example.ru/?auto=1&mode=quick&source=video:/media/squat01.mp4`.

## Результаты

Сохраняются в `/opt/mpb/results/raw/`, журнал — `/opt/mpb/logs/runs.md`. Забрать на ПК:

```bash
rsync -av user@server:/opt/mpb/results/raw/ results/raw/
```

## Безопасность

- Сервер принимает только JSON в формате `mpb-result/*`, не больше 64 МБ. Пути к файлам не выходят за `bench/web`, `models/cache`, `datasets/media`, `datasets/coco/val2017`.
- Видео с камеры участников на сервер не передаётся. В режиме видео передаются координаты точек тестового ролика.
- Если ссылка станет публичной, стоит закрыть страницу паролем (в Caddy — `basicauth`).

# Вариант без сервера: GitHub Pages

Адрес: **https://svladimir-cloud.github.io/mobile-pose-benchmarks/** (репозиторий публичный, Pages бесплатен).

Сайт собирает GitHub Actions (`.github/workflows/pages.yml`): `npm ci && npm run vendor`, скачивание весов и экспорт YOLO (`models/fetch_models.py --skip-reference`), сборка `tools/build_site.py` → `_site/`. Веса в git не попадают.

Отличия от своего сервера:
- **Заголовки изоляции** (многопоточный WASM) добавляет service worker (`bench/web/coi-sw.js`); при первом открытии страница один раз перезагружается.
- **Результаты не уходят на сервер автоматически**: они сохраняются в браузере телефона; участник нажимает «Поделиться» (мессенджер, почта) или «Скачать» и присылает JSON. Полученные файлы раскладываются по `results/raw/` командой `python tools/import_results.py <файлы>`.
- **Видео для режима «видео»** берутся из `bench/media/` (коммитятся в репозиторий; только свои записи с согласием, mp4 H.264, до ~20 МБ). Без них доступна только камера.
- `from_cache` в холодном старте не определяется (через service worker Resource Timing не отличает кэш от сети).

Включение (один раз, владелец репозитория):
1. Settings → Pages → Build and deployment → Source: **GitHub Actions**.
2. Settings → Environments → `github-pages` → Deployment branches: добавить `claude/sleepy-hopper-ycca7l` (или влить ветку в `main`).
3. Actions → `pages` → Run workflow (или любой push в `bench/`). Сборка ≈ 5–10 мин.

Ссылка с автозапуском для участников: `https://svladimir-cloud.github.io/mobile-pose-benchmarks/?auto=1&mode=quick&source=video:media/squat01.mp4`.

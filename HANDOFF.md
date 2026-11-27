# Как продолжить в новом чате

Подключить к сессии оба репозитория: `svladimir-cloud/mobile-pose-benchmarks` (эксперименты) и `svladimir-cloud/mobile-pose-article` (статья). Ветка с работой — `claude/sleepy-hopper-ycca7l`.

Текст для нового чата:

> Продолжаем экспериментальную часть статьи: можно ли на большинстве телефонов 2022+ извлекать ключевые точки локально в браузере достаточно быстро и точно для контроля упражнений. Репозиторий — mobile-pose-benchmarks, ветка claude/sleepy-hopper-ycca7l. Прочитай STATUS.md, AGENTS.md, docs/phone_protocol.md, docs/result_format.md и продолжи со «Следующих шагов». Протокол — раздел 3 статьи (mobile-pose-article/sections/03_methods.md); его нужно привести в соответствие с новой постановкой (STATUS.md, раздел «Постановка»).
> Правила: агенты пишут только в свои файлы и записывают результат после каждого шага; вспомогательных агентов запускай на модели sonnet, не больше 2–3 одновременно; после каждого шага — коммит и push.

Чтобы воспроизвести окружение в облаке:

```bash
pip install -r requirements.txt -r requirements-yolo.txt
npm ci && npm run vendor
python models/fetch_models.py
python eval/coco_select.py && python eval/coco_download.py
python bench/server.py --http --port 8080 &
node tools/smoke.mjs
```

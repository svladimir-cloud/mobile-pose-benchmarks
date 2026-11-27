"""Разложить присланные файлы «Скачать/Поделиться» (JSON-массив результатов) по results/raw/.

  python tools/import_results.py ~/Downloads/mpb_*.json

Каждый элемент сохраняется так же, как его сохранил бы сервер стенда (имя по R5.2, строка в logs/runs.md).
Повторный импорт того же run_id пропускается.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import server  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    known = set()
    for f in (ROOT / "results" / "raw").glob("*.json"):
        try:
            known.add(json.loads(f.read_text(encoding="utf-8")).get("run_id"))
        except Exception:  # noqa: BLE001
            pass
    added = skipped = 0
    for path in sys.argv[1:]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for doc in data if isinstance(data, list) else [data]:
            if not str(doc.get("schema", "")).startswith("mpb-result/"):
                continue
            if doc.get("run_id") in known:
                skipped += 1
                continue
            print("  ->", server.save_result(doc))
            known.add(doc.get("run_id"))
            added += 1
    print(f"добавлено {added}, пропущено (уже есть) {skipped}")


if __name__ == "__main__":
    main()

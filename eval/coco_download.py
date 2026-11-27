#!/usr/bin/env python3
"""Скачивание изображений COCO val2017 из списка (single_person_list.json)."""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = "http://images.cocodataset.org/val2017/"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, out_dir, base=BASE, retries=4, timeout=30, opener=None):
    """Скачивает один файл (если нет), с повторами. Возвращает (name, статус)."""
    dst = os.path.join(out_dir, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return name, "skip"
    opener = opener or urllib.request.urlopen
    err = None
    for k in range(retries):
        try:
            with opener(base + name, timeout=timeout) as r:
                data = r.read()
            if not data:
                raise IOError("пустой ответ")
            tmp = dst + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dst)
            return name, "ok"
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(1.5 * (k + 1))
    return name, f"fail: {err}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", default="datasets/coco/single_person_list.json")
    ap.add_argument("--out-dir", default="datasets/coco/val2017")
    ap.add_argument("--sha", default="datasets/coco/val2017_sha256.txt")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    with open(a.list) as f:
        names = [it["file_name"] for it in json.load(f)]
    os.makedirs(a.out_dir, exist_ok=True)
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda n: fetch(n, a.out_dir), names))
    bad = [r for r in res if r[1].startswith("fail")]
    print(f"всего {len(names)}; скачано {sum(r[1]=='ok' for r in res)}; "
          f"уже было {sum(r[1]=='skip' for r in res)}; ошибок {len(bad)}")
    for n, s in bad[:20]:
        print("  ", n, s)
    with open(a.sha, "w") as f:
        for n in sorted(names):
            p = os.path.join(a.out_dir, n)
            if os.path.exists(p):
                f.write(f"{sha256_file(p)}  {n}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

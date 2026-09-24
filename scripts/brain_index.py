#!/usr/bin/env python3
"""Индексация *_sum.md из ~/gdrive/Brain в векторную базу LanceDB.

1 структурный блок = 1 запись: «Главная мысль», каждый тезис, «Вывод».
Инкрементально: переиндексирует только новые/изменённые файлы (по хэшу),
удаляет записи исчезнувших файлов.

Usage: brain_index.py [--full]
"""
import hashlib
import pathlib
import re
import sys

import lancedb
import mlx.core as mx
import numpy as np
from mlx_embeddings.utils import load

HOME = pathlib.Path.home()
ROOT = HOME / "gdrive/Brain"
DB = HOME / "brain-vector/db"
MODEL_NAME = "mlx-community/Qwen3-Embedding-4B-4bit-DWQ"
BATCH = 16


def embed(model, tok, texts):
    enc = tok(texts, padding=True, truncation=True, max_length=2048, return_tensors="np")
    out = model(mx.array(enc["input_ids"]), mx.array(enc["attention_mask"]))
    return np.array(out.text_embeds.astype(mx.float32))


def parse(f: pathlib.Path):
    """→ (meta dict, chunks list[(section, text)]) или None."""
    t = f.read_text(encoding="utf-8")
    m = re.match(r"^- \*\*Название:\*\* ?(.*)\n- \*\*Автор:\*\* ?(.*)\n"
                 r"- \*\*Дата публикации:\*\* ?(.*)\n- \*\*Ссылка:\*\* ?(.*)\n", t)
    if not m:
        return None
    meta = {"title": m.group(1), "author": m.group(2), "date": m.group(3), "link": m.group(4)}
    body = t[m.end():]

    def section(name):
        sm = re.search(rf"^# {name}\n(.*?)(?=^# |\Z)", body, re.M | re.S)
        return sm.group(1).strip() if sm else ""

    chunks = []
    main = section("Главная мысль")
    if main:
        chunks.append(("мысль", main))
    theses = section("Ключевые тезисы")
    for b in re.split(r"\n\s*\n", theses):
        b = b.strip()
        if b.startswith("- "):
            chunks.append(("тезис", b[2:].strip()))
    out = section("Вывод")
    if out:
        chunks.append(("вывод", out))
    return meta, [c for c in chunks if len(c[1]) > 20]


def main():
    if "--optimize" in sys.argv:
        # склейка фрагментов + удаление старых версий; модель не нужна
        tbl = lancedb.connect(DB).open_table("chunks")
        tbl.optimize()
        nfiles = sum(1 for p in DB.rglob("*") if p.is_file())
        print(f"optimize OK: записей {tbl.count_rows()}, файлов в db: {nfiles}", flush=True)
        return
    full = "--full" in sys.argv
    db = lancedb.connect(DB)
    files = {f.name: f for f in ROOT.rglob("*_sum.md")}
    existing = {}
    tbl = None
    if "chunks" in db.table_names() and not full:
        tbl = db.open_table("chunks")
        for r in tbl.to_arrow().select(["filename", "fhash"]).to_pylist():
            existing[r["filename"]] = r["fhash"]

    changed, gone = [], set(existing) - set(files)
    for name, f in files.items():
        h = hashlib.sha1(f.read_bytes()).hexdigest()
        if full or existing.get(name) != h:
            changed.append((f, h))
    if not full and not changed and not gone:
        print("без изменений", flush=True)  # ранний выход: модель не грузим
        return
    print("загружаю модель…", flush=True)
    model, tok = load(MODEL_NAME)
    print(f"файлов: {len(files)} | новых/изменённых: {len(changed)} | удалённых: {len(gone)}", flush=True)

    records = []
    for f, h in changed:
        parsed = parse(f)
        if not parsed:
            print(f"  !! не распарсился: {f.name}", flush=True)
            continue
        meta, chunks = parsed
        texts = [c[1] for c in chunks]
        embs = []
        for i in range(0, len(texts), BATCH):
            embs.extend(embed(model, tok, texts[i:i + BATCH]).tolist())
        for (section, text), vec in zip(chunks, embs):
            records.append({"id": f"{f.name}#{len(records)}", "text": text, "vector": vec,
                            "filename": f.name, "folder": f.parent.name, "section": section,
                            "fhash": h, **meta})

    if tbl is not None and (changed or gone):
        drop = {f.name for f, _ in changed} | gone
        tbl.delete(" OR ".join(f"filename = '{n.replace(chr(39), chr(39)*2)}'" for n in drop))
    if records:
        if tbl is None:
            tbl = db.create_table("chunks", records)
        else:
            tbl.add(records)
    print(f"OK: записей в базе {tbl.count_rows() if tbl else 0} (+{len(records)})", flush=True)


if __name__ == "__main__":
    main()

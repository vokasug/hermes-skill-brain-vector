#!/usr/bin/env python3
"""Семантический поиск по векторной базе Brain.

Usage: brain_search.py "<запрос>" [--folder Finances] [--author habr]
       [--after 2026-01-01] [--before 2026-09-24] [--top 5] [--chunks 30] [--verbose]

Вывод: топ-N файлов (дедуп по лучшему чанку), затем для --verbose все чанки.
"""
import argparse
import pathlib

import lancedb
import mlx.core as mx
import numpy as np
from mlx_embeddings.utils import load

HOME = pathlib.Path.home()
ROOT = HOME / "gdrive/Brain"
DB = HOME / "brain-vector/db"
MODEL_NAME = "mlx-community/Qwen3-Embedding-4B-4bit-DWQ"
INSTRUCT = ("Instruct: Given a user question, retrieve relevant passages "
            "that answer the question\nQuery: ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--folder")
    ap.add_argument("--author")
    ap.add_argument("--after")
    ap.add_argument("--before")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--chunks", type=int, default=30)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    model, tok = load(MODEL_NAME)
    enc = tok([INSTRUCT + a.query], padding=True, truncation=True, max_length=2048, return_tensors="np")
    out = model(mx.array(enc["input_ids"]), mx.array(enc["attention_mask"]))
    qv = np.array(out.text_embeds.astype(mx.float32))[0].tolist()

    tbl = lancedb.connect(DB).open_table("chunks")
    where = []
    if a.folder:
        where.append(f"folder = '{a.folder}'")
    if a.author:
        where.append(f"lower(author) LIKE '%{a.author.lower()}%'")
    if a.after:
        where.append(f"date >= '{a.after}'")
    if a.before:
        where.append(f"date <= '{a.before}'")
    q = tbl.search(qv).limit(a.chunks)
    if where:
        q = q.where(" AND ".join(where))
    hits = q.to_list()

    best = {}
    for h in hits:
        fn = h["filename"]
        if fn not in best or h["_distance"] < best[fn]["_distance"]:
            best[fn] = h
    top = sorted(best.values(), key=lambda h: h["_distance"])[:a.top]

    print(f"запрос: {a.query} | фильтры: {' AND '.join(where) or 'нет'} | чанков проверено: {len(hits)}\n")
    for i, h in enumerate(top, 1):
        print(f"{i}. [{1 - h['_distance']:.3f}] {h['folder']}/{h['filename']}")
        print(f"   {h['title']} | {h['author']} | {h['date']}")
        print(f"   {h['section']}: {h['text'][:200]}{'…' if len(h['text']) > 200 else ''}\n")
    if a.verbose:
        print("=== все чанки ===")
        for h in hits:
            print(f"[{1 - h['_distance']:.3f}] {h['filename']} :: {h['section']} :: {h['text'][:120]}")


if __name__ == "__main__":
    main()

"""OpenWebText source: download parquet shards and yield documents in a seeded order.

The corpus is 80 parquet files. Shard order is a seeded shuffle and rows inside a shard are a seeded
permutation, so the document order is a pure function of the seed and we only download as many
shards as the token target needs. No Hugging Face token is required (the dataset is public).
"""

import os
import random
import shutil
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tiktoken

from dlm.shards import Encoder

N_SHARDS = 80
BASE_URL = (
    "https://huggingface.co/datasets/Skylion007/openwebtext/resolve/main/plain_text/"
    "train-{i:05d}-of-{n:05d}.parquet"
)
SOURCE = "Skylion007/openwebtext (plain_text, 80 parquet shards)"
EOS_ID = 50256


def shard_order(seed: int) -> list[int]:
    order = list(range(N_SHARDS))
    random.Random(seed).shuffle(order)
    return order


def download(i: int, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"train-{i:05d}.parquet"
    if dest.exists():
        return dest
    part = dest.with_suffix(".part")
    url = BASE_URL.format(i=i, n=N_SHARDS)
    with urllib.request.urlopen(url, timeout=60) as r, part.open("wb") as f:
        shutil.copyfileobj(r, f, 1 << 22)
    part.replace(dest)
    return dest


def owt_documents(seed: int, raw_dir: Path, keep_raw: bool = False) -> Iterator[str]:
    for i in shard_order(seed):
        path = download(i, raw_dir)
        texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
        for j in np.random.default_rng([seed, i]).permutation(len(texts)):
            yield texts[int(j)]
        if not keep_raw:
            path.unlink()


def gpt2_encoder() -> Encoder:
    enc = tiktoken.get_encoding("gpt2")
    threads = os.cpu_count() or 1

    def encode(texts: list[str]) -> list[list[int]]:
        return enc.encode_ordinary_batch(texts, num_threads=threads)

    return encode

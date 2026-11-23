"""Packed uint16 token shards and the deterministic chunk loader (ADR-0006).

Layout: documents are tokenized, each followed by EOS, and concatenated into one stream. The
stream is cut into shards whose length is a multiple of seq_len, so a chunk never straddles a
shard. Chunks are read back at fixed offsets, so every chunk is exactly seq_len tokens with no
padding and no cross-document masking.
"""

import hashlib
import itertools
import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

Encoder = Callable[[list[str]], list[list[int]]]
MAX_ID = 65535  # uint16


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def _batched(items: Iterable[str], n: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for x in items:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def _encode_docs(
    docs: Iterable[str], encode: Encoder, eos: int, batch_docs: int
) -> Iterator[tuple[np.ndarray, int]]:
    """Yield (uint16 tokens with EOS after each document, number of documents) per batch."""
    for batch in _batched(docs, batch_docs):
        parts: list[np.ndarray] = []
        for ids in encode(batch):
            arr = np.asarray([*ids, eos], dtype=np.int64)
            if arr.max() > MAX_ID:
                raise ValueError(f"token id {int(arr.max())} does not fit in uint16")
            parts.append(arr.astype(np.uint16))
        yield np.concatenate(parts), len(batch)


def build_shards(
    docs: Iterable[str],
    encode: Encoder,
    out_dir: Path,
    *,
    train_tokens: int,
    val_tokens: int,
    shard_tokens: int,
    seq_len: int,
    eos: int,
    seed: int,
    tokenizer: str,
    source: str,
    batch_docs: int = 2000,
) -> dict[str, Any]:
    """Write validation then train shards from an ordered document stream; return the manifest.

    Validation takes the first documents until it holds val_tokens, so the splits are disjoint by
    document. Both splits are truncated to a multiple of seq_len. Train is cut at exactly
    floor(train_tokens / seq_len) * seq_len tokens.
    """
    if shard_tokens % seq_len:
        raise ValueError("shard_tokens must be a multiple of seq_len")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = train_tokens // seq_len * seq_len
    stream = _encode_docs(docs, encode, eos, batch_docs)

    # Validation: the first whole batches of documents until we have enough tokens.
    val_parts: list[np.ndarray] = []
    val_docs, n = 0, 0
    spill: list[np.ndarray] = []
    for arr, nd in stream:
        if n >= val_tokens:
            spill.append(arr)
            break
        val_parts.append(arr)
        val_docs += nd
        n += len(arr)
    val = np.concatenate(val_parts)
    # Keep the split exact: documents beyond val_tokens in the last batch go to train, so cut at a
    # document boundary (an EOS), not mid-document.
    if len(val) > val_tokens:
        ends = np.flatnonzero(val == eos)
        keep = int(ends[np.searchsorted(ends, val_tokens - 1)]) + 1
        spill.insert(0, val[keep:])
        val_docs -= int(np.count_nonzero(val[keep:] == eos))
        val = val[:keep]
    leftover = np.concatenate(spill) if spill else np.empty(0, dtype=np.uint16)
    val_kept = len(val) // seq_len * seq_len
    val_path = out_dir / "val.bin"
    val[:val_kept].tofile(val_path)

    shards: list[dict[str, Any]] = []
    buf: list[np.ndarray] = []
    buffered, written = 0, 0

    def take(n: int) -> np.ndarray:
        nonlocal buf, buffered
        cat = np.concatenate(buf)
        head, rest = cat[:n], cat[n:]
        buf, buffered = ([rest], len(rest)) if len(rest) else ([], 0)
        return head

    def flush(arr: np.ndarray) -> None:
        nonlocal written
        path = out_dir / f"train_{len(shards):03d}.bin"
        arr.tofile(path)
        shards.append({"file": path.name, "tokens": len(arr), "sha256": sha256_file(path)})
        written += len(arr)

    for arr, _ in itertools.chain([(leftover, 0)] if len(leftover) else [], stream):
        buf.append(arr)
        buffered += len(arr)
        while buffered >= shard_tokens and written + shard_tokens < target:
            flush(take(shard_tokens))
        if written + buffered >= target:
            flush(take(target - written))
            break
    if written < target:
        raise RuntimeError(f"documents exhausted at {written + buffered} of {target} train tokens")

    manifest: dict[str, Any] = {
        "version": 1,
        "tokenizer": tokenizer,
        "eos_id": eos,
        "seq_len": seq_len,
        "seed": seed,
        "source": source,
        "dtype": "uint16",
        "val": {
            "file": val_path.name,
            "tokens": val_kept,
            "docs": val_docs,
            "sha256": sha256_file(val_path),
        },
        "train": {"tokens": written, "shards": shards},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class ShardDataset:
    """Chunks of seq_len tokens in a deterministic order that is a pure function of (seed, step)."""

    def __init__(self, data_dir: Path, split: Literal["train", "val"], seed: int = 0) -> None:
        m = json.loads((data_dir / "manifest.json").read_text())
        self.seq_len: int = m["seq_len"]
        files = [m["val"]] if split == "val" else m["train"]["shards"]
        self._maps = [np.memmap(data_dir / f["file"], dtype=np.uint16, mode="r") for f in files]
        per_shard = np.array([len(a) // self.seq_len for a in self._maps])
        self._starts = np.concatenate([[0], np.cumsum(per_shard)])
        self.n_chunks = int(self._starts[-1])
        self._order = (
            np.random.default_rng(seed).permutation(self.n_chunks)
            if split == "train"
            else np.arange(self.n_chunks)
        )

    def chunk(self, i: int) -> Tensor:
        s = int(np.searchsorted(self._starts, i, side="right")) - 1
        off = (i - int(self._starts[s])) * self.seq_len
        return torch.from_numpy(self._maps[s][off : off + self.seq_len].astype(np.int64))

    def batch(self, step: int, batch_size: int) -> Tensor:
        idx = self._order[step * batch_size : (step + 1) * batch_size]
        if len(idx) < batch_size:
            raise IndexError("past the end of the epoch")
        return torch.stack([self.chunk(int(i)) for i in idx])

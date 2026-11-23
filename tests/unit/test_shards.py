import json
from pathlib import Path

import numpy as np
import pytest

from dlm.shards import ShardDataset, build_shards

EOS = 0
SEQ = 4


def encode(texts: list[str]) -> list[list[int]]:
    return [[ord(c) for c in t] for t in texts]  # chars are >= 97, EOS 0 never collides


def docs(n: int) -> list[str]:
    return [f"d{i:03d}" + "x" * (i % 5) for i in range(n)]  # unique, varied lengths


def build(tmp: Path, items: list[str], **kw: int) -> dict:  # type: ignore[type-arg]
    args = {"train_tokens": 64, "val_tokens": 8, "shard_tokens": 16}
    args.update(kw)
    return build_shards(
        items, encode, tmp, seq_len=SEQ, eos=EOS, seed=0, tokenizer="test", source="test",
        batch_docs=3, **args,
    )  # fmt: skip


def read(tmp: Path, name: str) -> np.ndarray:
    return np.fromfile(tmp / name, dtype=np.uint16)


def test_packing_concatenates_documents_with_eos_and_no_padding(tmp_path: Path) -> None:
    build(tmp_path, ["abcd", "ef", "g", "hij", "kl", "mnop"] * 20, val_tokens=5, train_tokens=16)
    # Validation stops at the document boundary after 5 tokens ("abcd" + EOS), cut to one chunk.
    assert read(tmp_path, "val.bin").tolist() == [ord(c) for c in "abcd"]
    expected = [
        *map(ord, "ef"),
        0,
        ord("g"),
        0,
        *map(ord, "hij"),
        0,
        *map(ord, "kl"),
        0,
        *map(ord, "mnop"),
    ]
    assert read(tmp_path, "train_000.bin").tolist() == expected  # no padding, EOS between docs


def test_train_is_exactly_the_target_and_shards_are_chunk_aligned(tmp_path: Path) -> None:
    m = build(tmp_path, docs(200))
    assert m["train"]["tokens"] == 64
    sizes = [s["tokens"] for s in m["train"]["shards"]]
    assert sizes == [16, 16, 16, 16] and all(s % SEQ == 0 for s in sizes)
    assert m["val"]["tokens"] % SEQ == 0 and m["val"]["tokens"] >= 4


def test_validation_and_train_share_no_document(tmp_path: Path) -> None:
    build(tmp_path, docs(200))
    train = np.concatenate(
        [
            read(tmp_path, s["file"])
            for s in json.loads((tmp_path / "manifest.json").read_text())["train"]["shards"]
        ]
    )

    def doc_set(tokens: np.ndarray) -> set[str]:
        out = {
            "".join(map(chr, d))
            for d in np.split(tokens, np.flatnonzero(tokens == EOS) + 1)
            if len(d) > 1
        }
        return {d[:-1] for d in out}

    val_docs = doc_set(read(tmp_path, "val.bin"))
    assert val_docs and not (val_docs & doc_set(train))


def test_rebuilding_with_same_input_gives_identical_hashes(tmp_path: Path) -> None:
    a = build(tmp_path / "a", docs(200))
    b = build(tmp_path / "b", docs(200))
    assert [s["sha256"] for s in a["train"]["shards"]] == [
        s["sha256"] for s in b["train"]["shards"]
    ]
    assert a["val"]["sha256"] == b["val"]["sha256"]


def test_running_out_of_documents_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="exhausted"):
        build(tmp_path, docs(5))


def test_token_ids_must_fit_uint16(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="uint16"):
        build_shards(
            ["a"],
            lambda t: [[70000] for _ in t],
            tmp_path,
            train_tokens=4,
            val_tokens=4,
            shard_tokens=4,
            seq_len=SEQ,
            eos=EOS,
            seed=0,
            tokenizer="t",
            source="t",
        )


def test_loader_order_is_seeded_complete_and_stable(tmp_path: Path) -> None:
    build(tmp_path, docs(200))
    a, a2, b = (ShardDataset(tmp_path, "train", s) for s in (0, 0, 1))
    n = a.n_chunks
    assert n == 16
    first_a = [a.batch(i, 2).tolist() for i in range(n // 2)]
    assert first_a == [a2.batch(i, 2).tolist() for i in range(n // 2)]
    assert first_a != [b.batch(i, 2).tolist() for i in range(n // 2)]
    visited = sorted(tuple(c) for batch in first_a for c in batch)
    assert visited == sorted(tuple(a.chunk(i).tolist()) for i in range(n))  # each chunk once
    with pytest.raises(IndexError):
        a.batch(n // 2, 2)


def test_validation_loader_is_sequential(tmp_path: Path) -> None:
    build(tmp_path, docs(200))
    v = ShardDataset(tmp_path, "val")
    raw = read(tmp_path, "val.bin")
    assert v.batch(0, 1)[0].tolist() == raw[:SEQ].tolist()

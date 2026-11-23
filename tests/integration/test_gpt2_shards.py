"""Real GPT-2 tokenizer through the shard builder. Needs network once for the BPE vocab."""

from pathlib import Path

import numpy as np
import pytest
import tiktoken

from dlm.owt import EOS_ID, gpt2_encoder
from dlm.shards import ShardDataset, build_shards

pytestmark = pytest.mark.integration


def test_gpt2_ids_fit_and_roundtrip(tmp_path: Path) -> None:
    texts = [f"Document {i}: the quick brown fox jumps over the lazy dog. " * 20 for i in range(60)]
    m = build_shards(
        texts, gpt2_encoder(), tmp_path, train_tokens=4096, val_tokens=1024, shard_tokens=2048,
        seq_len=256, eos=EOS_ID, seed=0, tokenizer="gpt2", source="test",
    )  # fmt: skip
    ids = np.concatenate(
        [np.fromfile(tmp_path / s["file"], dtype=np.uint16) for s in m["train"]["shards"]]
    )
    assert ids.max() <= EOS_ID  # the DLM's [MASK] id is added at runtime, never stored
    enc = tiktoken.get_encoding("gpt2")
    chunk = ShardDataset(tmp_path, "train", 0).chunk(0).tolist()
    assert enc.encode(enc.decode(chunk), allowed_special={"<|endoftext|>"}) == chunk

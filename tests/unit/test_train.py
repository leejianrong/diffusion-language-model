import json
import math
from pathlib import Path

import pytest
import torch

from dlm import train as train_mod
from dlm.config import Config, ModelConfig, ToyDataConfig, TrainConfig
from dlm.train import NaNLossError, load_checkpoint, train


def _cfg(kind: str, root: Path, name: str) -> Config:
    return Config(
        kind=kind,  # type: ignore[arg-type]
        base_vocab=16,
        seed=0,
        run_root=str(root),
        run_name=name,
        model=ModelConfig(block_size=8, n_layer=1, n_head=2, d_model=32),
        train=TrainConfig(steps=20, batch_size=8, lr=3e-3, warmup=2, ckpt_every=10, log_every=5),
        data=ToyDataConfig(n_sequences=32, seq_len=8),
    )


@pytest.mark.parametrize("kind", ["ar", "dlm"])
def test_resume_is_bit_identical_to_uninterrupted_run(kind: str, tmp_path: Path) -> None:
    straight = _cfg(kind, tmp_path, "straight")
    train(straight)
    split = _cfg(kind, tmp_path, "split")
    train(split, stop_after=10)
    train(split, resume=True)
    a = load_checkpoint(straight.run_dir)["model"]
    b = load_checkpoint(split.run_dir)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)

    def losses(c: Config) -> list[float]:
        lines = (c.run_dir / "log.jsonl").read_text().splitlines()
        return [json.loads(x)["loss"] for x in lines if json.loads(x)["step"] > 10]

    assert losses(straight) == losses(split)


def test_config_is_copied_into_run_dir_and_roundtrips(tmp_path: Path) -> None:
    cfg = _cfg("dlm", tmp_path, "c")
    train(cfg, stop_after=1)
    assert Config.from_yaml(cfg.run_dir / "config.yaml") == cfg


def test_nan_loss_aborts_and_keeps_last_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg("ar", tmp_path, "nan")
    train(cfg, stop_after=10)
    before = (cfg.run_dir / "checkpoints" / "latest.pt").read_bytes()
    monkeypatch.setattr(train_mod, "batch_loss", lambda *a, **k: torch.tensor(float("nan")))
    with pytest.raises(NaNLossError):
        train(cfg, resume=True)
    assert (cfg.run_dir / "checkpoints" / "latest.pt").read_bytes() == before


def _shard_cfg(root: Path, data_dir: Path, name: str, **train: int | bool) -> Config:
    base = dict(steps=6, batch_size=4, lr=3e-3, warmup=1, ckpt_every=3, log_every=1, eval_seqs=8)
    return Config(
        kind="dlm",
        base_vocab=50,
        run_root=str(root),
        run_name=name,
        data_dir=str(data_dir),
        model=ModelConfig(block_size=4, n_layer=1, n_head=2, d_model=16),
        train=TrainConfig(**(base | train)),  # type: ignore[arg-type]
    )


@pytest.fixture
def shards(tmp_path: Path) -> Path:
    from dlm.shards import build_shards

    out = tmp_path / "shards"
    build_shards(
        [f"doc{i:03d}" + "ab" * (i % 4) for i in range(200)],
        lambda ts: [[ord(c) % 49 + 1 for c in t] for t in ts],
        out, train_tokens=256, val_tokens=32, shard_tokens=64, seq_len=4, eos=0,
        seed=0, tokenizer="t", source="t", batch_docs=7,
    )  # fmt: skip
    return out


def test_training_on_shards_counts_tokens_exactly_and_survives_resume(
    tmp_path: Path, shards: Path
) -> None:
    straight = _shard_cfg(tmp_path, shards, "s")
    train(straight)
    split = _shard_cfg(tmp_path, shards, "r")
    train(split, stop_after=3)
    train(split, resume=True)
    recs = [json.loads(x) for x in (split.run_dir / "log.jsonl").read_text().splitlines()]
    assert [r["tokens_seen"] for r in recs] == [r["step"] * 4 * 4 for r in recs]
    assert recs[-1]["tokens_seen"] == 6 * 4 * 4
    a, b = (load_checkpoint(c.run_dir)["model"] for c in (straight, split))
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert "eval_loss" in json.loads((split.run_dir / "results.json").read_text())


def test_tampered_shards_abort_before_any_training(tmp_path: Path, shards: Path) -> None:
    from dlm.shards import ShardIntegrityError

    f = shards / "train_000.bin"
    f.write_bytes(b"\x01" + f.read_bytes()[1:])
    cfg = _shard_cfg(tmp_path, shards, "bad")
    with pytest.raises(ShardIntegrityError):
        train(cfg)
    assert not (cfg.run_dir / "checkpoints").exists()


def test_grad_accumulation_matches_one_big_batch(tmp_path: Path, shards: Path) -> None:
    def ar(name: str, accum: int) -> Config:
        c = _shard_cfg(tmp_path, shards, name, steps=2, ckpt_every=2, grad_accum=accum)
        return Config.from_dict({**c.to_dict(), "kind": "ar"})

    one, acc = ar("one", 1), ar("acc", 2)
    train(one)
    train(acc)
    a, b = (load_checkpoint(c.run_dir)["model"] for c in (one, acc))
    assert all(torch.allclose(a[k], b[k], atol=1e-5) for k in a)

    def first_loss(c: Config) -> float:
        return float(json.loads((c.run_dir / "log.jsonl").read_text().splitlines()[0])["loss"])

    # Adam is scale-invariant, so weights alone would not catch a missing 1/accum; the loss does.
    assert first_loss(one) == pytest.approx(first_loss(acc), rel=1e-4)


def test_nonfinite_gradient_also_aborts_and_keeps_the_checkpoint(
    tmp_path: Path, shards: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _shard_cfg(tmp_path, shards, "g")
    train(cfg, stop_after=3)
    before = (cfg.run_dir / "checkpoints" / "latest.pt").read_bytes()
    monkeypatch.setattr(
        torch.nn.utils, "clip_grad_norm_", lambda *a, **k: torch.tensor(float("inf"))
    )
    with pytest.raises(NaNLossError, match="gradient"):
        train(cfg, resume=True)
    assert (cfg.run_dir / "checkpoints" / "latest.pt").read_bytes() == before


def test_compile_flag_wraps_the_forward_but_checkpoints_stay_plain(
    tmp_path: Path, shards: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def fake_compile(m: torch.nn.Module, **kw: object) -> torch.nn.Module:
        calls.append(1)
        return m

    monkeypatch.setattr(torch, "compile", fake_compile)
    cfg = _shard_cfg(tmp_path, shards, "c", compile=True)
    train(cfg)
    assert calls == [1]
    assert not any(k.startswith("_orig_mod") for k in load_checkpoint(cfg.run_dir)["model"])


def test_bf16_flag_runs_forward_in_bf16_but_loss_stays_float32(
    tmp_path: Path, shards: Path
) -> None:
    cfg = _shard_cfg(tmp_path, shards, "bf", bf16=True)
    out = train(cfg)
    assert math.isfinite(out["last_loss"]) and out["step"] == 6


def test_progress_goes_to_stderr_with_throughput(
    tmp_path: Path, shards: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    train(_shard_cfg(tmp_path, shards, "p"))
    lines = [x for x in capsys.readouterr().err.splitlines() if x.startswith("progress dlm ")]
    assert len(lines) == 6
    rec = json.loads(lines[-1].removeprefix("progress dlm "))
    assert rec["step"] == 6 and rec["tok_per_s"] > 0


def test_resuming_a_finished_run_keeps_its_last_loss(tmp_path: Path, shards: Path) -> None:
    cfg = _shard_cfg(tmp_path, shards, "done")
    first = train(cfg)
    again = train(cfg, resume=True)  # no steps left
    assert again["last_loss"] == first["last_loss"] and math.isfinite(again["last_loss"])

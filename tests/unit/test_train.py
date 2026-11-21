import json
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

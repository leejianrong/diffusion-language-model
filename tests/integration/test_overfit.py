"""V1 end-to-end: both tiny models memorise a toy set, and the samplers reproduce it.

Slow (minutes on CPU), so it is excluded from the pre-push gate and runs in its own CI job.
"""

import dataclasses
import json
import math
from pathlib import Path

import pytest
import torch

from dlm.config import Config
from dlm.data import memorisation_accuracy, toy_dataset
from dlm.diffusion import corrupt, sample_ar, sample_dlm
from dlm.train import build_model, load_checkpoint, train

pytestmark = pytest.mark.integration
CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _run(name: str, tmp_path: Path) -> tuple[Config, dict[str, float]]:
    cfg = dataclasses.replace(Config.from_yaml(CONFIGS / name), run_root=str(tmp_path))
    return cfg, train(cfg)


def _gap_closed(cfg: Config, loss: float) -> float:
    """Fraction of the distance from ln(V) (untrained) to the memorisation floor covered.

    The floor is ln(N) / predicted_tokens: choosing which of N sequences it is costs ln N nats
    and nothing else is unknown. The DLM's exact ELBO reaches the same bound.
    """
    n, length = cfg.data.n_sequences, cfg.data.seq_len
    floor = math.log(n) / (length if cfg.kind == "dlm" else length - 1)
    return (math.log(cfg.base_vocab) - loss) / (math.log(cfg.base_vocab) - floor)


def _samples(cfg: Config, train_data: torch.Tensor, n: int = 16) -> torch.Tensor:
    """AR: unconditional. DLM: infill 64 steps from a sequence with half its tokens visible.

    Unconditional DLM sampling from all-[MASK] is not gated: at this training length the model
    cannot yet pick one memorised sequence from zero context (about 0.4 match), though it
    recovers masked tokens almost perfectly once a quarter of the sequence is visible.
    """
    model = build_model(cfg)
    model.load_state_dict(load_checkpoint(cfg.run_dir)["model"])
    model.eval()
    gen = torch.Generator().manual_seed(0)
    if cfg.kind == "dlm":
        x0 = train_data[:n]
        start = corrupt(x0, torch.full((n,), 0.5), cfg.mask_id, gen)
        return sample_dlm(model, n, cfg.data.seq_len, cfg.mask_id, 64, gen, x_init=start)
    return sample_ar(model, n, cfg.data.seq_len, 0, gen)


@pytest.mark.parametrize("name", ["tiny_ar.yaml", "tiny_dlm.yaml"])
def test_tiny_model_memorises_toy_set(name: str, tmp_path: Path) -> None:
    cfg, summary = _run(name, tmp_path)
    assert _gap_closed(cfg, summary["eval_loss"]) >= 0.9
    results = json.loads((cfg.run_dir / "results.json").read_text())
    assert results["kind"] == cfg.kind
    train_data = toy_dataset(cfg.data.n_sequences, cfg.data.seq_len, cfg.base_vocab, cfg.seed)
    assert memorisation_accuracy(_samples(cfg, train_data), train_data) >= 0.9

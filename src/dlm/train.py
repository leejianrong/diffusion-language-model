"""Training loop with run directory, checkpoints and exact resume."""

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from dlm.config import Config
from dlm.data import toy_dataset
from dlm.diffusion import ar_loss, corrupt, dlm_loss, sample_t
from dlm.model import GPT


class NaNLossError(RuntimeError):
    pass


def build_model(cfg: Config) -> GPT:
    m = cfg.model
    return GPT(
        cfg.vocab_size,
        m.block_size,
        m.n_layer,
        m.n_head,
        m.d_model,
        cfg.attention,
        abs_pos=m.abs_pos,
    )


def lr_at(step: int, cfg: Config) -> float:
    t = cfg.train
    if step < t.warmup:
        return t.lr * (step + 1) / t.warmup
    frac = (step - t.warmup) / max(1, t.steps - t.warmup)
    return t.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))


def batch_loss(model: GPT, cfg: Config, x0: Tensor, gen: torch.Generator) -> Tensor:
    if cfg.kind == "ar":
        return ar_loss(model(x0), x0)
    t = sample_t(x0.shape[0], gen)
    xt = corrupt(x0, t, cfg.mask_id, gen)
    return dlm_loss(model(xt), x0, xt, t, cfg.mask_id)


@torch.no_grad()
def evaluate(model: GPT, cfg: Config, data: Tensor, seed: int = 1234, reps: int = 4) -> float:
    """Fixed-seed loss in nats per token. Exact NLL for AR, ELBO upper bound for the DLM."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    total, n = 0.0, 0
    for _ in range(reps if cfg.kind == "dlm" else 1):
        for i in range(0, len(data), 256):
            x0 = data[i : i + 256].to(device)
            total += float(batch_loss(model, cfg, x0, gen)) * len(x0)
            n += len(x0)
    model.train()
    return total / n


def _save(
    path: Path, model: GPT, opt: torch.optim.Optimizer, step: int, gen: torch.Generator
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "step": step,
            "gen": gen.get_state(),
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(run_dir: Path) -> dict[str, Any]:
    return torch.load(run_dir / "checkpoints" / "latest.pt", weights_only=True)  # type: ignore[no-any-return]


def train(
    cfg: Config, resume: bool = False, stop_after: int | None = None, device: str = "cpu"
) -> dict[str, Any]:
    """Train to cfg.train.steps (or stop_after, for tests). Returns a summary dict."""
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(run_dir / "config.yaml")
    torch.manual_seed(cfg.seed)
    data = toy_dataset(cfg.data.n_sequences, cfg.data.seq_len, cfg.base_vocab, cfg.seed)
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay, betas=(0.9, 0.95)
    )
    gen = torch.Generator().manual_seed(cfg.seed)
    step = 0
    if resume:
        ck = load_checkpoint(run_dir)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        gen.set_state(ck["gen"])
        step = int(ck["step"])
    end = cfg.train.steps if stop_after is None else min(cfg.train.steps, stop_after)
    log = (run_dir / "log.jsonl").open("a")
    t0, last = time.time(), float("nan")
    while step < end:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        idx = torch.randint(0, len(data), (cfg.train.batch_size,), generator=gen)
        loss = batch_loss(model, cfg, data[idx].to(device), gen)
        if not torch.isfinite(loss):
            raise NaNLossError(f"non-finite loss at step {step}; last good checkpoint kept")
        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        last = float(loss.detach())
        if step % cfg.train.log_every == 0 or step == end:
            rec: dict[str, Any] = {
                "step": step,
                "tokens_seen": step * cfg.train.batch_size * cfg.data.seq_len,
                "loss": last,
                "wall_s": time.time() - t0,
            }
            if cfg.train.eval_every and step % cfg.train.eval_every == 0:
                rec["eval"] = evaluate(model, cfg, data)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % cfg.train.ckpt_every == 0 or step == end:
            _save(run_dir / "checkpoints" / "latest.pt", model, opt, step, gen)
    log.close()
    final = evaluate(model, cfg, data)
    summary = {
        "run_dir": str(run_dir),
        "kind": cfg.kind,
        "step": step,
        "last_loss": last,
        "eval_loss": final,
    }
    (run_dir / "results.json").write_text(json.dumps(summary, indent=2))
    return summary

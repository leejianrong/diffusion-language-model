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
from dlm.shards import ShardDataset, verify_shards


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


def _forward(model: Any, x: Tensor, bf16: bool) -> Tensor:
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=bf16):
        logits: Tensor = model(x)
    return logits.float()


def batch_loss(
    model: Any, cfg: Config, x0: Tensor, gen: torch.Generator, bf16: bool = False
) -> Tensor:
    if cfg.kind == "ar":
        return ar_loss(_forward(model, x0, bf16), x0)
    t = sample_t(x0.shape[0], gen)
    xt = corrupt(x0, t, cfg.mask_id, gen)
    return dlm_loss(_forward(model, xt, bf16), x0, xt, t, cfg.mask_id)


@torch.no_grad()
def evaluate(
    model: GPT, cfg: Config, data: Tensor, seed: int = 1234, reps: int = 4, batch: int = 256
) -> float:
    """Fixed-seed loss in nats per token. Exact NLL for AR, ELBO upper bound for the DLM."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    total, n = 0.0, 0
    for _ in range(reps if cfg.kind == "dlm" else 1):
        for i in range(0, len(data), batch):
            x0 = data[i : i + batch].to(device)
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
    tc = cfg.train
    if tc.batch_size % tc.grad_accum:
        raise ValueError("batch_size must be a multiple of grad_accum")
    run_dir = cfg.run_dir
    seq_len = cfg.model.block_size if cfg.data_dir else cfg.data.seq_len
    if cfg.data_dir:  # a bad upload must cost nothing: check before any paid compute
        verify_shards(Path(cfg.data_dir))
        shard_train = ShardDataset(Path(cfg.data_dir), "train", cfg.seed)
        val = ShardDataset(Path(cfg.data_dir), "val")
        data = torch.stack([val.chunk(i) for i in range(min(tc.eval_seqs, val.n_chunks))])

        def get_batch(step: int, gen: torch.Generator) -> Tensor:
            return shard_train.batch(step, tc.batch_size)
    else:
        data = toy_dataset(cfg.data.n_sequences, cfg.data.seq_len, cfg.base_vocab, cfg.seed)

        def get_batch(step: int, gen: torch.Generator) -> Tensor:
            return data[torch.randint(0, len(data), (tc.batch_size,), generator=gen)]

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(run_dir / "config.yaml")
    torch.manual_seed(cfg.seed)
    model = build_model(cfg).to(device)
    fwd: Any = torch.compile(model) if tc.compile else model  # checkpoints use the plain model
    opt = torch.optim.AdamW(
        model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay, betas=(0.9, 0.95)
    )
    gen = torch.Generator().manual_seed(cfg.seed)
    step = 0
    if resume:
        ck = load_checkpoint(run_dir)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        gen.set_state(ck["gen"])
        step = int(ck["step"])
    end = tc.steps if stop_after is None else min(tc.steps, stop_after)
    log = (run_dir / "log.jsonl").open("a")
    t0, last = time.time(), float("nan")
    while step < end:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for micro in get_batch(step, gen).to(device).chunk(tc.grad_accum):
            loss = batch_loss(fwd, cfg, micro, gen, tc.bf16) / tc.grad_accum
            if not torch.isfinite(loss):
                raise NaNLossError(f"non-finite loss at step {step}; last good checkpoint kept")
            loss.backward()  # type: ignore[no-untyped-call]
            total += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise NaNLossError(f"non-finite gradient at step {step}; last good checkpoint kept")
        opt.step()
        step += 1
        last = total
        if step % tc.log_every == 0 or step == end:
            rec: dict[str, Any] = {
                "step": step,
                "tokens_seen": step * tc.batch_size * seq_len,
                "loss": last,
                "wall_s": time.time() - t0,
            }
            if tc.eval_every and step % tc.eval_every == 0:
                rec["eval"] = evaluate(model, cfg, data, batch=tc.eval_batch_size)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % tc.ckpt_every == 0 or step == end:
            _save(run_dir / "checkpoints" / "latest.pt", model, opt, step, gen)
    log.close()
    final = evaluate(model, cfg, data, batch=tc.eval_batch_size)
    summary = {
        "run_dir": str(run_dir),
        "kind": cfg.kind,
        "step": step,
        "last_loss": last,
        "eval_loss": final,
    }
    (run_dir / "results.json").write_text(json.dumps(summary, indent=2))
    return summary

"""Masked diffusion (MDLM) pieces: corruption, ELBO loss, sampler (ADR-0001, ADR-0004)."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

T_EPS = 1e-3


def sample_t(b: int, gen: torch.Generator, stratified: bool = True) -> Tensor:
    """Mask times in (0, 1]. Stratified: one draw per equal-width bin, shuffled."""
    u = torch.rand(b, generator=gen, dtype=torch.float64)
    if stratified:
        u = (torch.randperm(b, generator=gen).double() + u) / b
    return (T_EPS + (1 - T_EPS) * u).float()


def corrupt(x0: Tensor, t: Tensor, mask_id: int, gen: torch.Generator) -> Tensor:
    """Replace each token by [MASK] independently with probability t (per sequence)."""
    u = torch.rand(x0.shape, generator=gen).to(x0.device)
    return torch.where(u < t.to(x0.device)[:, None], torch.full_like(x0, mask_id), x0)


def dlm_loss_per_seq(logits: Tensor, x0: Tensor, xt: Tensor, t: Tensor, mask_id: int) -> Tensor:
    """Sum over masked positions of CE / t, shape (B,). Nats, not yet divided by length.

    The network may not predict [MASK], and unmasked positions are carried over, so they
    contribute nothing (MDLM's SUBS parameterisation). The 1/t weight is
    -alpha'(t) / (1 - alpha(t)) for the linear schedule alpha(t) = 1 - t.
    """
    logits = logits.float().clone()
    logits[..., mask_id] = float("-inf")
    nll = -torch.gather(F.log_softmax(logits, dim=-1), -1, x0[..., None]).squeeze(-1)
    masked = (xt == mask_id).float()
    return (nll * masked).sum(-1) / t.to(nll.device)


def dlm_loss(logits: Tensor, x0: Tensor, xt: Tensor, t: Tensor, mask_id: int) -> Tensor:
    """ELBO estimate in nats per token."""
    return dlm_loss_per_seq(logits, x0, xt, t, mask_id).sum() / x0.numel()


def ar_loss(logits: Tensor, x: Tensor) -> Tensor:
    """Next-token cross-entropy in nats per token."""
    return F.cross_entropy(logits[:, :-1].float().flatten(0, 1), x[:, 1:].flatten())


def gumbel_noise(shape: torch.Size, gen: torch.Generator, device: torch.device) -> Tensor:
    """float64 Gumbel noise. float32 sampling lowers effective temperature (ADR-0004)."""
    u = torch.rand(shape, generator=gen, dtype=torch.float64, device=device)
    return -torch.log(-torch.log(u.clamp_min(torch.finfo(torch.float64).tiny)))


def sample_categorical(logits: Tensor, gen: torch.Generator) -> Tensor:
    z = logits.double() + gumbel_noise(logits.shape, gen, logits.device)
    return z.argmax(-1)


@torch.no_grad()
def sample_dlm(
    model: nn.Module,
    n: int,
    length: int,
    mask_id: int,
    steps: int,
    gen: torch.Generator,
    x_init: Tensor | None = None,
) -> Tensor:
    """Ancestral reverse process. Unmasked tokens are never changed.

    Starts from all-[MASK], or from `x_init` (n, length) to infill: any non-[MASK] token in
    it is kept as given.
    """
    device = next(model.parameters()).device
    if x_init is None:
        x = torch.full((n, length), mask_id, dtype=torch.long, device=device)
    else:
        x = x_init.to(device).clone()
    ts = torch.linspace(1, 0, steps + 1, dtype=torch.float64)
    for k in range(steps):
        t, s = float(ts[k]), float(ts[k + 1])
        logits = model(x).float()
        logits[..., mask_id] = float("-inf")
        proposal = sample_categorical(logits, gen)
        u = torch.rand(x.shape, generator=gen, dtype=torch.float64, device=device)
        unmask = (x == mask_id) & (u < (t - s) / t)
        x = torch.where(unmask, proposal, x)
    return x


@torch.no_grad()
def sample_ar(
    model: nn.Module,
    n: int,
    length: int,
    bos_id: int,
    gen: torch.Generator,
    temperature: float = 1.0,
) -> Tensor:
    """Unconditional AR sampling from a start token. Output includes the start token."""
    device = next(model.parameters()).device
    x = torch.full((n, 1), bos_id, dtype=torch.long, device=device)
    block = int(model.block_size)  # type: ignore[arg-type]
    for _ in range(length - 1):
        logits = model(x[:, -block:])[:, -1].float() / temperature
        x = torch.cat([x, sample_categorical(logits, gen)[:, None]], dim=1)
    return x

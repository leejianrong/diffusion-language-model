import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dlm.diffusion import (
    ar_loss,
    corrupt,
    dlm_loss,
    dlm_loss_per_seq,
    gumbel_noise,
    sample_ar,
    sample_categorical,
    sample_dlm,
    sample_t,
)
from dlm.model import GPT


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# --- corruption -------------------------------------------------------------------------


def test_corrupt_extremes_and_rate() -> None:
    x0 = torch.randint(0, 10, (4, 50))
    assert torch.equal(corrupt(x0, torch.zeros(4), 10, _gen()), x0)
    assert (corrupt(x0, torch.ones(4), 10, _gen()) == 10).all()
    big = torch.randint(0, 10, (2, 20000))
    out = corrupt(big, torch.tensor([0.3, 0.8]), 10, _gen())
    rate = (out == 10).float().mean(1)
    assert abs(rate[0] - 0.3) < 0.02 and abs(rate[1] - 0.8) < 0.02


def test_stratified_t_has_lower_variance_than_iid() -> None:
    def batch_stats(strat: bool) -> float:
        g = _gen(1)
        return float(torch.stack([sample_t(16, g, strat).mean() for _ in range(200)]).var())

    assert batch_stats(True) < 0.1 * batch_stats(False)


def test_sample_t_range() -> None:
    t = sample_t(1000, _gen())
    assert t.min() > 0 and t.max() <= 1


# --- DLM loss ---------------------------------------------------------------------------


def test_uniform_model_bound_is_ln_vocab() -> None:
    v, mask_id = 50, 50
    x0 = torch.randint(0, v, (512, 64), generator=_gen())
    g = _gen(2)
    t = sample_t(512, g)
    xt = corrupt(x0, t, mask_id, g)
    loss = dlm_loss(torch.zeros(512, 64, v + 1), x0, xt, t, mask_id)
    assert abs(float(loss) - math.log(v)) / math.log(v) < 0.02


def test_unmasked_positions_contribute_nothing() -> None:
    x0 = torch.randint(0, 5, (3, 6))
    logits = torch.randn(3, 6, 6)
    loss = dlm_loss_per_seq(logits, x0, x0, torch.full((3,), 0.5), 5)  # nothing masked
    assert torch.equal(loss, torch.zeros(3))


V, L = 4, 3
MASK = V


def _toy_table() -> Tensor:
    g = torch.Generator().manual_seed(7)
    return torch.randn((V + 1) ** L, L, V + 1, generator=g).double()


def _logits(xt: Tensor, table: Tensor) -> Tensor:
    idx = (xt * (V + 1) ** torch.arange(L)).sum(-1)
    return table[idx]


def _patterns() -> Tensor:
    return torch.tensor([[(m >> i) & 1 for i in range(L)] for m in range(2**L)], dtype=torch.bool)


def _toy_inputs() -> tuple[Tensor, Tensor, Tensor]:
    g = torch.Generator().manual_seed(3)
    x0s = torch.randint(0, V, (5, L), generator=g)
    pats = _patterns()
    x0 = x0s[:, None, :].expand(-1, 2**L, -1)
    pat = pats[None].expand(5, -1, -1)
    xt = torch.where(pat, torch.full_like(x0, MASK), x0)
    return x0, xt, pat.sum(-1)  # (5, 8, L), (5, 8, L), (5, 8) masked counts


def _exact_discrete_time_elbo(table: Tensor, steps: int = 20000) -> float:
    """First-principles oracle, independent of our loss code.

    Absorbing diffusion with alpha(t) = 1 - t on a `steps`-step grid. The KL between the true
    posterior q(x_s | x_t, x_0) and the SUBS model collapses, for each masked token, to
    (alpha_s - alpha_t) / (1 - alpha_t) * (-log p(x_0 | x_t)). We enumerate every mask
    pattern with its exact probability, so there is no sampling noise.
    """
    x0, xt, count = _toy_inputs()
    lg = _logits(xt.reshape(-1, L), table)
    lg[..., MASK] = -np.inf
    nll = -torch.gather(torch.log_softmax(lg, -1), -1, x0.reshape(-1, L)[..., None]).squeeze(-1)
    masked = (xt.reshape(-1, L) == MASK).double()
    ce = (nll * masked).sum(-1).reshape(5, 2**L).numpy()  # (x0, pattern)
    cnt = count.numpy()
    k = np.arange(1, steps + 1)
    t = k / steps
    w = (1 / steps) / t  # (alpha_s - alpha_t) / (1 - alpha_t)
    prob = t[None, :] ** cnt[0][:, None] * (1 - t[None, :]) ** (L - cnt[0][:, None])  # (pattern, k)
    per_pattern = (prob * w[None, :]).sum(1)  # (pattern,)
    return float((ce * per_pattern[None, :]).sum(1).mean() / L)


def _our_elbo(table: Tensor, loss_fn=dlm_loss_per_seq) -> float:  # type: ignore[no-untyped-def]
    """Our loss, integrated over t exactly by Gauss-Legendre and over patterns by enumeration."""
    x0, xt, count = _toy_inputs()
    nodes, weights = np.polynomial.legendre.leggauss(12)
    total = 0.0
    for node, wt in zip(nodes, weights, strict=True):
        t = float((node + 1) / 2)
        per = loss_fn(
            _logits(xt.reshape(-1, L), table),
            x0.reshape(-1, L),
            xt.reshape(-1, L),
            torch.full((5 * 2**L,), t, dtype=torch.float64),
            MASK,
        ).reshape(5, 2**L)
        prob = (t ** count.double()) * ((1 - t) ** (L - count.double()))
        total += float(wt / 2 * (prob * per).sum(1).mean() / L)
    return total


def test_dlm_loss_matches_enumerated_elbo() -> None:
    table = _toy_table()
    assert abs(_our_elbo(table) - _exact_discrete_time_elbo(table)) < 2e-3


def test_wrong_weighting_fails_the_oracle() -> None:
    table = _toy_table()

    def unweighted(logits: Tensor, x0: Tensor, xt: Tensor, t: Tensor, mask_id: int) -> Tensor:
        return dlm_loss_per_seq(logits, x0, xt, t, mask_id) * t  # drops the 1/t weight

    assert abs(_our_elbo(table, unweighted) - _exact_discrete_time_elbo(table)) > 0.05


def _mdlm_reference(logits: Tensor, x0: Tensor, xt: Tensor, t: Tensor, mask_id: int) -> Tensor:
    """MDLM's loss, ported structurally: SUBS carry-over on every position, then
    log_p * dalpha / (1 - alpha). Written separately from dlm_loss_per_seq on purpose."""
    logits = logits.clone()
    logits[..., mask_id] = -torch.inf
    log_p = torch.log_softmax(logits, -1)
    carry = torch.full_like(log_p, -torch.inf)
    carry.scatter_(-1, xt[..., None], 0.0)  # unmasked: probability one on the observed token
    log_p = torch.where((xt != mask_id)[..., None], carry, log_p)
    log_p_x0 = torch.gather(log_p, -1, x0[..., None]).squeeze(-1)
    alpha, dalpha = 1 - t, torch.full_like(t, -1.0)
    return (log_p_x0 * (dalpha / (1 - alpha))[:, None]).sum(-1)


def test_dlm_loss_matches_ported_mdlm_reference() -> None:
    g = _gen(5)
    x0 = torch.randint(0, 20, (32, 16), generator=g)
    t = sample_t(32, g).double()
    xt = corrupt(x0, t, 20, g)
    logits = torch.randn(32, 16, 21, generator=g, dtype=torch.float64)
    ours = dlm_loss_per_seq(logits, x0, xt, t, 20)
    ref = _mdlm_reference(logits, x0, xt, t, 20)
    assert torch.allclose(ours, ref, rtol=1e-5, atol=1e-6)


# --- AR loss ----------------------------------------------------------------------------


def test_ar_loss_on_uniform_logits_is_ln_vocab() -> None:
    x = torch.randint(0, 30, (4, 10))
    assert abs(float(ar_loss(torch.zeros(4, 10, 30), x)) - math.log(30)) < 1e-5


# --- samplers ---------------------------------------------------------------------------


def test_gumbel_noise_is_float64() -> None:
    assert gumbel_noise(torch.Size([4]), _gen(), torch.device("cpu")).dtype == torch.float64


def test_sample_categorical_matches_distribution() -> None:
    logits = torch.log(torch.tensor([0.1, 0.2, 0.7])).expand(100000, 3)
    freq = torch.bincount(sample_categorical(logits, _gen()), minlength=3) / 100000
    assert torch.allclose(freq, torch.tensor([0.1, 0.2, 0.7]), atol=0.01)


class _Recorder(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.seen: list[Tensor] = []

    def forward(self, x: Tensor) -> Tensor:
        self.seen.append(x.clone())
        return self.inner(x)  # type: ignore[no-any-return]


def test_dlm_sampler_never_changes_unmasked_tokens_and_ends_fully_unmasked() -> None:
    torch.manual_seed(0)
    rec = _Recorder(GPT(9, 12, 1, 2, 16, "bidirectional"))
    out = sample_dlm(rec, 4, 12, 8, steps=10, gen=_gen())
    assert (out != 8).all()
    assert len(rec.seen) == 10
    for a, b in zip(rec.seen, [*rec.seen[1:], out], strict=True):
        keep = a != 8
        assert torch.equal(a[keep], b[keep])


def test_dlm_sampler_infills_and_keeps_given_tokens() -> None:
    torch.manual_seed(0)
    model = GPT(9, 12, 1, 2, 16, "bidirectional")
    start = torch.randint(0, 8, (4, 12))
    start[:, ::2] = 8  # mask every other position
    out = sample_dlm(model, 4, 12, 8, steps=6, gen=_gen(), x_init=start)
    keep = start != 8
    assert torch.equal(out[keep], start[keep]) and (out != 8).all()


def test_ar_sampler_starts_with_bos_and_has_requested_length() -> None:
    torch.manual_seed(0)
    out = sample_ar(GPT(9, 12, 1, 2, 16, "causal"), 3, 12, 0, _gen())
    assert out.shape == (3, 12) and (out[:, 0] == 0).all()


def test_nll_helper_agrees_with_torch_cross_entropy() -> None:
    logits, y = torch.randn(5, 7), torch.randint(0, 7, (5,))
    manual = -torch.log_softmax(logits, -1)[torch.arange(5), y].mean()
    assert torch.allclose(manual, F.cross_entropy(logits, y))

import torch
from torch import Tensor


def toy_dataset(n: int, seq_len: int, vocab: int, seed: int) -> Tensor:
    """Random sequences to memorise. Token 0 is the start token and always comes first."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randint(1, vocab, (n, seq_len), generator=gen)
    x[:, 0] = 0
    return x


def memorisation_accuracy(samples: Tensor, train: Tensor) -> float:
    """Mean over samples of the best token-match fraction against any training sequence."""
    eq = (samples[:, None, :] == train[None, :, :]).float().mean(-1)
    return float(eq.max(1).values.mean())

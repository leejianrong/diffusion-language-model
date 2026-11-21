import torch

from dlm.config import Config, ModelConfig
from dlm.model import GPT, attention_mask
from dlm.train import build_model


def test_causal_mask_attends_to_past_only() -> None:
    m = attention_mask("causal", 8)
    assert torch.equal(m, torch.tril(torch.ones(8, 8, dtype=torch.bool)))


def test_bidirectional_mask_attends_everywhere() -> None:
    assert attention_mask("bidirectional", 8).all()


def test_block_causal_mask_full_inside_block_causal_across() -> None:
    m = attention_mask("block_causal", 8, block=4)
    assert m[:4, :4].all() and m[4:, :].all()  # inside a block, and later blocks see earlier ones
    assert not m[:4, 4:].any()  # earlier blocks never see later ones


def _tiny(kind: str) -> GPT:
    torch.manual_seed(0)
    return GPT(20, 8, 2, 2, 16, kind, attn_block=4)


def test_forward_shape() -> None:
    out = _tiny("causal")(torch.randint(0, 20, (3, 8)))
    assert out.shape == (3, 8, 20)


def test_causal_model_ignores_future_tokens_bidirectional_does_not() -> None:
    x = torch.randint(0, 20, (1, 8))
    y = x.clone()
    y[0, -1] = (y[0, -1] + 1) % 20
    causal, bidir = _tiny("causal"), _tiny("bidirectional")
    assert torch.allclose(causal(x)[:, :-1], causal(y)[:, :-1], atol=1e-6)
    assert not torch.allclose(bidir(x)[:, :-1], bidir(y)[:, :-1], atol=1e-6)


def test_block_causal_model_matches_block_structure() -> None:
    m = _tiny("block_causal")
    x = torch.randint(0, 20, (1, 8))
    y = x.clone()
    y[0, 5] = (y[0, 5] + 1) % 20  # change a token in block 1
    assert torch.allclose(m(x)[:, :4], m(y)[:, :4], atol=1e-6)  # block 0 cannot see block 1
    assert not torch.allclose(m(x)[:, 4:], m(y)[:, 4:], atol=1e-6)


def test_dlm_has_exactly_one_extra_embedding_row() -> None:
    model = ModelConfig(block_size=8, n_layer=2, n_head=2, d_model=16)
    ar = build_model(Config(kind="ar", base_vocab=20, model=model))
    dlm = build_model(Config(kind="dlm", base_vocab=20, model=model))
    n_ar = sum(p.numel() for p in ar.parameters())
    n_dlm = sum(p.numel() for p in dlm.parameters())
    assert n_dlm - n_ar == 16  # one row of the tied embedding

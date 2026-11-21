# ADR-0002: One transformer backbone for both models, with the attention mask as a pluggable function

- Status: Accepted
- Date: 2026-11-22
- Deciders: Jian (owner), agent

## Context
Fair AR vs DLM comparison needs the networks to differ only where the objectives require it. The owner also intends a phase 2 hybrid (AR over blocks, diffusion inside a block), which needs a block-causal attention pattern. That pattern should not require rewriting the backbone.

## Decision
- A single decoder-style transformer (GPT-2-small shape: 12 layers, 12 heads, width 768, RoPE, RMSNorm, SwiGLU or GELU MLP, tied input and output embeddings, about 124M parameters including embeddings). Choices are fixed once and shared by both models.
- The attention pattern is an argument: `causal`, `bidirectional`, or later `block_causal(block_size)`. AR uses `causal`, the DLM uses `bidirectional`. Nothing else in the network differs. The DLM vocabulary has one extra `[MASK]` row, which is the only parameter-count difference and is reported.
- Attention goes through PyTorch SDPA (FlashAttention backend on GPU). A mask function that SDPA cannot fuse falls back to an explicit boolean mask on CPU for tests.
- A unit test asserts the three mask patterns on a tiny sequence, so phase 2 has a verified hook.

## Alternatives considered
| Option | Why not |
|--------|---------|
| Separate model files for AR and DLM | Invites accidental differences that undermine the comparison |
| Hard-code causal and bidirectional only | Forces a backbone edit in phase 2 for no saving now |
| Add FlexAttention now | Newer, more moving parts, and block masks are not needed this milestone. Revisit in phase 2 |

## Consequences
- The comparison is clean: any quality gap is attributable to the objective and the mask.
- Architecture choices (RoPE and so on) are tuned for neither model. Acceptable for a comparison, and phase 2 may revisit them.
- Block-causal is tested as a mask but never trained here, so a phase-2 training bug is still possible.

## Amendment (V1)

The backbone has an optional learned absolute-position embedding (`abs_pos`, default off). Only the toy memorisation configs turn it on: without it, a bidirectional RoPE-only model has no anchor for absolute position and the DLM does not learn to memorise. Real runs keep RoPE only, and AR and DLM always share the same setting.

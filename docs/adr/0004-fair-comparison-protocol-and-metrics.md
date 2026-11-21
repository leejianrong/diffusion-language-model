# ADR-0004: Compare AR and DLM at matched parameters and matched tokens seen, with three metric families

- Status: Accepted
- Date: 2026-11-22
- Deciders: Jian (owner), agent

## Context
It is easy to produce a flattering or misleading AR vs DLM comparison: perplexity is not like for like, generative perplexity can be gamed by low-entropy samples, and speed depends on sampler settings. The protocol is fixed before any result exists so it cannot drift toward the answer.

## Decision
**Matching.** Same backbone (ADR-0002), same tokenizer and data order seed, same optimizer and schedule (AdamW, cosine, same peak LR, warmup, weight decay, batch size in tokens), same total tokens seen (about 2B, one epoch of the subset). The only budget difference allowed is wall-clock, which is logged and reported but not matched. Learning rate is tuned once at the tiny scale and shared. No per-model tuning at full scale, and that limitation is stated in the report.

**Metrics.**
1. Validation NLL on a fixed 5M-token held-out set. AR is exact. DLM is the ELBO bound. Both in nats per token and as perplexity, with the bound caveat in every table caption.
2. Generative perplexity of 256 unconditional samples at length 1024 under GPT-2 large, reported with mean per-sample token entropy next to it. A model that lowers gen-ppl by collapsing entropy is flagged in the report, not praised. DLM sampler at 64, 256, 1024 steps. AR at temperature 1, no top-k.
3. Throughput in tokens per second at length 1024, batch size 1 and a batch size at saturation. AR uses a KV cache. Same GPU model for both and recorded in the result file.

**Sampling precision.** DLM categorical sampling uses float64 Gumbel noise (Q33), because float32 sampling reduces effective temperature and makes DLM gen-ppl look better than it is.

**Noise floor.** Three seeds at about 10M parameters for each model show the seed-to-seed spread of metric 1, so differences smaller than that spread are not reported as findings. Full-scale runs use one seed (budget).

## Alternatives considered
| Option | Why not |
|--------|---------|
| Match wall-clock or FLOPs | Hides the DLM's training inefficiency, which is part of what the owner wants to learn (F4) |
| Downstream zero-shot tasks | Near chance at this scale (Q6) |
| Report DLM perplexity without the bound caveat | Misleading |

## Consequences
- The headline reads "at equal tokens seen, the DLM is X behind", and the cost of that gap in GPU-hours is shown separately.
- A single full-scale seed means the headline carries the noise-floor caveat.
- Gen-ppl needs GPT-2 large on the GPU for each eval, a small extra cost already in the budget.

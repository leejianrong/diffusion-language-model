# ADR-0001: Use masked (absorbing-state) discrete diffusion with the MDLM continuous-time ELBO

- Status: Accepted
- Date: 2026-11-22
- Deciders: Jian (owner), agent

## Context
The project trains a diffusion language model from scratch and compares it to an autoregressive (AR) model. The formulation fixes the loss, the sampler, and which likelihood claims can be made. The owner's later direction (AR + DLM block hybrid, chasing state of the art) points at the masked family, which is what LLaDA and block-diffusion work build on.

## Decision
- Forward process: each token is independently replaced by a `[MASK]` token with probability `t`, `t ~ U(0,1)`, linear schedule (`alpha_t = 1 - t`).
- Loss: the MDLM continuous-time ELBO. Cross-entropy on masked positions only, weighted by `1/t`, averaged over tokens. Use low-discrepancy (stratified) `t` sampling within a batch to cut variance.
- No timestep input to the network (Q32). The model reads the noise level from the visible mask fraction.
- Sampler: ancestral reverse process with `T` steps, tokens once unmasked are never remasked in this milestone. Sampling noise in float64 (ADR-0004).
- Validation reports the ELBO (an upper bound on NLL) in nats per token, computed with a fixed set of `t` draws and fixed seed so numbers are comparable between checkpoints.

## Alternatives considered
| Option | Why not |
|--------|---------|
| Uniform-state discrete diffusion | Worse likelihoods in the literature, no clear reason to prefer it at this scale |
| Continuous embedding diffusion (Diffusion-LM, CDCD) | No tight likelihood bound, needs rounding tricks, a different research track from the owner's stated direction |
| Timestep-conditioned network (adaLN) | Extra parameters and a backbone that differs from AR beyond the mask, which complicates the fairness rule. Cheap to add later |
| Remasking samplers (ReMDM, etc.) | Deferred to after the first comparison, Q5 |

## Consequences
- Likelihood is a bound, so DLM perplexity is an upper bound and AR perplexity is exact. Every report must say so beside the number.
- The `1/t` weighting makes the loss high variance at small `t`. Stratified `t` is required, and a variance check is part of V1.
- Fixed-length generation: sequence length is set up front. Variable-length output is a phase-2 concern.

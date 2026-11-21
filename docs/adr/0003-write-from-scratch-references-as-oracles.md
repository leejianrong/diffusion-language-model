# ADR-0003: Write the training code from scratch in plain PyTorch; reference repos are test oracles only

- Status: Accepted
- Date: 2026-11-22
- Deciders: Jian (owner)

## Context
The stated purpose is learning and research. The MDLM and LLaDA reference code are available but pull in heavy frameworks (Hydra, Lightning) and would turn the project into reading someone else's code. The risk of writing it ourselves is a subtly wrong loss that still trains.

## Decision
- A nanoGPT-style codebase, target about 600 lines for the model, losses, sampler and train loop, no Lightning, no HF Trainer, no Hydra. Config is a dataclass loaded from YAML.
- Package management with `uv`. PyTorch 2.x, `tiktoken`, `numpy`, `matplotlib`, `pytest`. Python 3.12 or 3.13, pinned in `pyproject.toml`.
- Correctness of the DLM loss is checked three ways, none of which depend on trusting our own derivation: (1) a closed-form check on a tiny vocabulary where the exact ELBO is computable by enumeration, (2) an untrained (uniform-output) model has a bound within 2 percent of `ln(V)` nats per token, (3) the loss agrees numerically with the MDLM reference formula, ported into the test as a standalone function (not imported from their repo).

## Alternatives considered
| Option | Why not |
|--------|---------|
| Fork the MDLM repo | Faster to a number, but not what the owner wants, and ties us to its stack |
| Use HF Transformers for the backbone | Hides the attention mask logic that phase 2 depends on |

## Consequences
- More build time (about two extra days against forking) and a real chance of bugs. The oracle tests are the mitigation and slice V1 exists to confront this first.
- The codebase stays small enough to reason about, which phase 2 benefits from.

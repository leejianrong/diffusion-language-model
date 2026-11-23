# ADR-0005: Train on a rented RunPod GPU under a hard $60 cap; local machine does CPU tests only

- Status: Accepted
- Date: 2026-11-22
- Deciders: Jian (owner)

## Context
The host has no GPU and 7 GB RAM. `runpodctl` is already installed. Budget is about $60 in total (F1). Rough sizing: training cost is about `6 * N * D` FLOPs, so 125M parameters over 2B tokens is about 1.5e18 FLOPs per model. At a realistic 30 to 40 percent utilisation on an A100 that is about 4 to 6 hours per model, so about $15 to $25 per model at typical community rates. This estimate is checked in V3 before committing to the long runs.

## Decision
- One pod at a time, A100 40/80 GB or H100 on RunPod, launched through the `runpod-jobs` skill pattern (dead-man's-switch so a forgotten pod terminates itself).
- Spend tracking: a `budget.json` records each pod's start, stop and hourly rate. A script refuses to launch a pod if the recorded spend plus the planned run exceeds $60.
- Order of spend: V3 smoke (under $3), AR full run, DLM full run, eval pass, reserve of about $10 for a rerun or debugging. The DLM run is not started until the AR run has produced a sane number (V4 gate).
- Local machine runs the pytest suite on tiny CPU configs and the report generator only.

## Alternatives considered
| Option | Why not |
|--------|---------|
| Colab or free tiers | Sessions die mid-run, no scripted teardown, hard to reproduce |
| Smaller model to save money | Still possible, the fallback if V3 shows the estimate is off by more than 2x: drop to about 60M parameters and keep the protocol |

## Consequences
- No room for a second full-scale seed or for hyperparameter sweeps at full scale.
- Pod spot interruptions are possible, so training must resume from a checkpoint (Q15).
- If the cap is reached the plan degrades by shrinking scale, never by dropping the fairness rule.

## Amendment 2026-11-24: measured on a real pod (smoke run, $0.24 spent)
- Setup: one A100-SXM4-80GB (secure, $2.19/h), GPT-2-small shape, bf16 autocast plus `torch.compile`, 32 sequences of 1024 per step (two micro-batches of 16), torch 2.15.0+cu132 from PyPI on the `runpod/pytorch:2.8.0-py3.11` image with a uv-managed Python 3.12.
- Throughput: AR 138k tokens/s, DLM 132k tokens/s (the DLM has one extra vocabulary row), 17 GB of GPU memory, GPU utilisation about 98%.
- Projection for 2B tokens: AR about 4.0 h, DLM about 4.2 h, so about $8.80 and $9.10 on the secure cloud, roughly $17 for the pair. That is under the $15-25 per model estimate, so the full-scale plan stands and no shrink to 60M parameters is needed.
- Caveats: a 300-step run, so sustained throughput on the full run may differ a little; GPU memory is far from full, so a bigger micro-batch may be faster; community A100 had no stock and secure was used.
- RunPod's console spend limit is an hourly rate cap (`GET /v2/account/limits`, here $80/h), not a total, so the ledger in `budget.json` is the only total-spend cap.

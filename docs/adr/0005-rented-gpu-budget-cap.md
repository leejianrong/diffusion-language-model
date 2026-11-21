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

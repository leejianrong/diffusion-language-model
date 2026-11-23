# Agent brief

From-scratch masked-diffusion LM vs autoregressive baseline on OpenWebText. Read `docs/PLAN.md` first.

## Build status

Trust the code over the docs where they disagree, and fix the docs in the same PR.

| Slice | State |
|---|---|
| V1 CPU overfit + DLM loss oracles | done; `make test-integration` takes ~3.5 min |
| V2 data shards | in progress |
| V3 pod smoke run | ledger + gate done; **pod launcher blocked** (see below) |
| V4 full runs | not started |
| V5 benchmark + report | not started |

## Commands

- `make install` — `uv sync --frozen` (CPU torch). `make check` — lint, strict mypy, unit tests. Run it before every push.
- `dlm train --config configs/tiny_dlm.yaml` / `dlm sample --run runs/<name> [--visible 0.5]` are the CLI entry points (JSON on stdout).
- `dlm prep-data --out data/owt` builds the 2B-token OpenWebText shards (downloads ~5 GB, ~20-30 min; `data/` is git-ignored). `ShardDataset` in `dlm.shards` reads them; batch order is a pure function of (seed, step).
- `make test` runs only `tests/unit` (no infra, CPU, seconds). Tests needing network/GPU/pod go in `tests/integration` with `@pytest.mark.integration`.
- GPU pods install the CUDA torch wheel separately; never change the lockfile for that.

## RunPod: known blocker (2026-11-23)

RunPod REST v1 is gone (`rest.runpod.io` is NXDOMAIN) and v2 lives at `https://api.runpod.io/v2`. The `rp` skill script and `runpodctl` 2.14.0 (installed locally, and pinned on the pod by `rp`) still target v1, so `rp up` cannot work and the pod-side self-terminate would silently fail. `dlm pod launch` is unit-tested against a fake runner only. Do not launch a real pod until the launcher is ported and the dead-man's-switch is proven on a cheap CPU pod. `runpodctl` 3.1.1 is the current release.

## Workflow

- One branch per slice off fresh `main`, PR-only, CI green before merge. No direct pushes to `main`.
- Every bug becomes a failing test first. A guard test is proven by watching it fail.
- Slices and stories are tracked on the Pandan board for this project. Move a card to done only after its PR merges.

## Rules that are easy to get wrong

- The DLM loss is the MDLM ELBO (ADR-0001). Verify against the oracles in V1, never against our own derivation.
- AR and DLM share one backbone that differs only in attention mask and one `[MASK]` row (ADR-0002). Do not add model differences.
- DLM sampling noise is float64 (ADR-0004). Do not "optimise" it to float32.
- Spend on rented GPUs is capped at $60 and tracked in `budget.json` (`dlm pod status`; the gate refuses a launch whose worst case would exceed the cap). Never launch a pod without the dead-man's-switch (ADR-0005).
- Secrets (RunPod key, HF token) come from env vars only. Never commit or print them.

## Map

`src/dlm/` package · `tests/unit`, `tests/integration` · `docs/PLAN.md`, `docs/SLICES.md`, `docs/QUESTIONS.md`, `docs/adr/` · `scripts/git-hooks/`

# Agent brief

From-scratch masked-diffusion LM vs autoregressive baseline on OpenWebText. Read `docs/PLAN.md` first.

## Build status

Trust the code over the docs where they disagree, and fix the docs in the same PR.

| Slice | State |
|---|---|
| V1 CPU overfit + DLM loss oracles | done; `make test-integration` takes ~3.5 min |
| V2 data shards | in progress |
| V3 pod smoke run | ledger, gate, launcher proven on CPU pods; shard verify and GPU trainer (bf16, compile, grad accumulation) done; GPU smoke run passed (AR and DLM, 300 steps each); kill-and-resume check on GPU still open |
| V4 full runs | not started |
| V5 benchmark + report | not started |

## Commands

- `make install` — `uv sync --frozen` (CPU torch). `make check` — lint, strict mypy, unit tests. Run it before every push.
- `dlm train --config configs/tiny_dlm.yaml` / `dlm sample --run runs/<name> [--visible 0.5]` are the CLI entry points (JSON on stdout).
- `dlm prep-data --out data/owt` builds the 2B-token OpenWebText shards (downloads ~5 GB, ~20-30 min; `data/` is git-ignored). `ShardDataset` in `dlm.shards` reads them; batch order is a pure function of (seed, step).
- `dlm verify-data` checks every shard's size and SHA-256 against the committed `configs/owt_manifest.json` (exit 6 on any mismatch); `dlm train` does the same before any compute when `data_dir` is set. Get shards to a pod by rebuilding there (`dlm prep-data`, deterministic) or copying, then verify.
- Config knobs for real runs: `data_dir`, `train.bf16`, `train.compile`, `train.grad_accum` (batch_size is the full batch). A non-finite loss or gradient exits 3 and keeps the last checkpoint.
- `make test` runs only `tests/unit` (no infra, CPU, seconds). Tests needing network/GPU/pod go in `tests/integration` with `@pytest.mark.integration`.
- GPU pods install the CUDA torch wheel separately; never change the lockfile for that.

## RunPod (REST v2)

Client is `dlm.runpod` (stdlib, `https://api.runpod.io/v2`; v1 and the old `rp` script are dead). The key is read by reference (env var or runpodctl's config), never logged, and error bodies are redacted. Pods get an API key and `RUNPOD_POD_ID` injected, which is how the pod-side switch terminates itself. The log endpoint is an SSE stream: use `Client.logs_text`, never a read-until-EOF. All response reads are bounded by a total deadline.

`dlm pod launch|stop|status --plan <yaml>` goes through the budget gate. Proven live (2026-11-23, `scripts/pod_proof.py`, ~$0.02): the idle watchdog and the max-lifetime ceiling each terminated a CPU pod on their own. Re-run it after touching the start script or client. GPU pods: `gpu: {id: <catalog id>, count: 1}` in the plan, validated live. Check stock first (`GET /catalog/gpus?include=AVAILABILITY&product=POD`; a create with no capacity fails with 400 and spends nothing). The pod job is `scripts/pod_job.sh` (stage markers, 30 s heartbeat with WARN lines, per-step timeouts, always publishes an `RP_REPORT` block); drive it with `uv run python scripts/run_pod_job.py configs/pods/smoke.yaml`, which pins the pushed HEAD, watches the log, saves it to `.pod-logs/`, and always terminates and verifies. Measured numbers are in the ADR-0005 amendment.

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

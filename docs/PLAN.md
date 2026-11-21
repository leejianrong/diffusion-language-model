# Diffusion Language Model: Plan

Status: agreed (round 1 closed) · Milestone: M1, exploratory baseline

## Problem

The owner wants to understand diffusion language models by building one, and to know honestly how a small masked-diffusion LM compares with an autoregressive (AR) model of the same size. Published comparisons use compute budgets far beyond a personal project, and reference code is heavy enough that reading it teaches less than writing it.

A later phase aims to build something competitive, likely an AR + DLM hybrid that generates in blocks. That phase needs working, trusted parts: a correct DLM objective, a fair evaluation, and a training workflow that does not burn money. This milestone produces those parts.

## Solution

A small Python codebase that, from the command line, prepares OpenWebText, trains a ~125M-parameter AR model and a ~125M-parameter masked-diffusion model on the same tokens, evaluates both, and writes one report. Training runs on a rented GPU under a $60 cap and tears itself down. The result is a table and plots showing, at equal tokens seen, how far apart the two models are on likelihood, sample quality and speed, with the caveats stated next to each number.

## Users and actors

- **Primary: the owner**, learning and doing research. Readability of the code outranks speed of the code.
- **Agents** write most of the code and run commands. Every command exits non-zero on failure and emits JSON, so an agent can drive the pipeline.
- **RunPod pod** is a transient actor. It must not outlive its job. When a pod's convenience conflicts with the budget cap, the cap wins.

## Scope

**In this milestone.**
- One shared transformer backbone, AR (causal) and masked-diffusion (bidirectional) objectives (ADR-0001, ADR-0002).
- Data pipeline for a 2B-token OpenWebText subset (ADR-0006).
- Resumable training on a rented GPU with budget tracking (ADR-0005).
- Evaluation under the fixed protocol: validation NLL or ELBO, generative perplexity with entropy, throughput (ADR-0004).
- A noise-floor estimate from three small-scale seeds.
- A `RESULTS.md` with plots.

**Out.**
- The hybrid, block diffusion training, remasking samplers, SOTA claims. Deferred by the owner's own sequencing (Q31). Only the pluggable block-causal mask is built, so the door stays open.
- Downstream benchmark suites. Near chance at this scale (Q6).
- A serving layer, UI, or any product surface.
- Hyperparameter sweeps at full scale. The budget does not allow it.
- Publishing weights or data (OpenWebText licensing, ADR-0006).

## Requirements

| ID | Requirement | Status |
|----|-------------|--------|
| R0 | State, with numbers and caveats, how a from-scratch masked-diffusion LM compares with an AR model at equal parameters and equal tokens seen on OpenWebText | Core goal |
| R1 | Both models come from one shared backbone and one codebase, differing only in attention mask, objective and one `[MASK]` embedding row | Must-have |
| R2 | The DLM loss is demonstrably the correct ELBO, checked against an oracle independent of our derivation | Must-have |
| R3 | The whole pipeline (data, train, eval, report) is reproducible from config and seed, and training resumes from checkpoints | Must-have |
| R4 | Total rented-GPU spend is at most $60, and no pod can run unattended past its job | Must-have |
| R5 | Evaluation follows ADR-0004 exactly, and the report states every caveat that ADR lists | Must-have |
| R6 | A noise floor from three seeds at ~10M parameters is reported, and differences below it are not called findings | Nice-to-have |
| R7 | The attention mask supports causal, bidirectional and block-causal, with block-causal covered by tests for phase 2 | Nice-to-have |

## Shape

| Part | Mechanism | ADR |
|------|-----------|-----|
| S1 | Backbone: one decoder transformer taking an attention-pattern argument. AR uses causal, DLM uses bidirectional | ADR-0002 |
| S2 | AR objective: next-token cross-entropy, shifted by one | ADR-0004 |
| S3 | DLM objective: sample `t` (stratified), mask each token with prob `t`, cross-entropy on masked positions weighted `1/t` | ADR-0001 |
| S4 | DLM sampler: ancestral reverse process in `T` steps, float64 sampling noise, no remasking | ADR-0001, ADR-0004 |
| S5 | Data prep: OWT to GPT-2 BPE to uint16 shards with manifest, run locally and uploaded to the pod | ADR-0006 |
| S6 | Train loop: AdamW, cosine schedule, bf16, `torch.compile`, checkpoint every N steps, NaN guard, logs loss against tokens and wall-clock | ADR-0003 |
| S7 | Pod runner: launches a pod with a dead-man's-switch, checks the budget ledger first, syncs shards and checkpoints | ADR-0005 |
| S8 | Evaluator: validation NLL/ELBO with fixed seed, gen-ppl and entropy under GPT-2 large, throughput benchmark | ADR-0004 |
| S9 | Reporter: reads `results.json` files only, writes tables and PNG plots into `RESULTS.md` | ADR-0004 |

## Affordances

**Non-UI.** There is no UI.

| Affordance | Kind | Wires to |
|------------|------|----------|
| `prep_data` | CLI | S5, writes shards and `manifest.json` |
| `train --config <yaml> [--resume]` | CLI | S1, S2 or S3, S6; writes `runs/<id>/` |
| `eval --run <id>` | CLI | S8; writes `runs/<id>/results.json` |
| `sample --run <id> --steps T` | CLI | S4 (or AR sampling); prints decoded text |
| `pod launch / pod status / pod stop` | CLI | S7, `budget.json` |
| `report` | CLI | S9, reads all `results.json` |
| `runs/<id>/` | Directory store | config, checkpoints (git-ignored), logs, `results.json` |
| `budget.json` | Ledger | pod start, stop, rate, running total |

## Implementation decisions

- Layout: a small package with modules for the model, the two objectives, the sampler, data, the trainer, evaluation, and the pod runner. The model module has no knowledge of objectives, and objectives have no knowledge of the trainer, so each piece is testable on CPU alone.
- Config is a dataclass loaded from YAML and copied into every run directory. A run is identified by `<model>-<yyyymmdd>-<seed>` (Q8).
- Shapes: width 768, 12 layers, 12 heads, context 1024, vocabulary 50258 for the DLM and 50257 for AR (ADR-0002). Global batch about 0.5M tokens per step, so 2B tokens is about 4,000 steps. Final values are set in V3 after measuring memory and throughput.
- Learning rate is picked at the ~10M scale and reused (ADR-0004). Peak LR, warmup and weight decay live in config, not in code.
- Every command prints one JSON object on stdout, human logs go to stderr, exit codes are 0 for success and non-zero otherwise, with distinct codes for budget refusal and NaN abort.
- Secrets: RunPod key and HF token come from environment variables, never from config or logs.

## Testing approach

Test the behaviour at the fewest, highest seams. All tests run on CPU with tiny configs in seconds, so the full suite runs on this machine without a GPU.

- The mathematical seam is the DLM loss and sampler, tested against oracles (ADR-0003). This is where a silent error would hurt most.
- The pipeline seam is "tiny config in, `results.json` out", run end to end.
- Do not test internals such as layer shapes unless a bug has already bitten there.
- GPU-only behaviour (throughput, bf16, compile) is checked by the smoke run in V3 and recorded in its output, not by unit tests.

Per-slice plans are in `SLICES.md`.

## Assumed defaults

| ID | Assumed | Cost if wrong |
|----|---------|---------------|
| Q4 | Masked diffusion, not uniform or continuous | Loss and sampler rewrite, 2 to 3 days |
| Q32 | No timestep conditioning | Small quality loss, add adaLN later. Costs one rerun of the DLM |
| Q20 | GPT-2 large as the gen-ppl judge | Numbers still valid but less comparable to some papers |
| Q27 | One seed at full scale | Headline carries the noise-floor caveat, no extra spend possible |
| Q34 | Tokenize locally and upload | A slower upload, hours not dollars |
| Q35 | Packing without cross-document masking | Small, same for both models |

Row IDs refer to `QUESTIONS.md`.

## Open risks

- **Budget estimate is theoretical.** The 4 to 6 hours per model figure assumes 30 to 40 percent utilisation. V3 measures it before the long runs, and ADR-0005 names the fallback (shrink to ~60M parameters).
- **Silent DLM bugs.** A wrongly weighted loss can still train and give plausible-looking curves. V1 exists to catch this and runs first.
- **DLM may look much worse at this token count.** Published DLMs trail AR at equal tokens, and 2B tokens is a small fraction of theirs. That is a result, not a failure, but R0 must be worded accordingly.
- **Phase 2 expectations.** A SOTA claim cannot rest on $60 and 125M parameters. Phase 2 needs its own budget fork and its own run of this planning process, using M1's code, protocol and ledger as inputs.

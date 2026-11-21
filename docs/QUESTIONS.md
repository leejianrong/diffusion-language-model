# Decision register

Status: FORK = needs the user, ASSUMED = default recorded, DEFERRED = after this milestone, ANSWERED = user decided.
Host facts: WSL2, no local GPU, 7 GB RAM, 16 cores, `uv` and `runpodctl` installed. All training must happen on a rented GPU.

## 1. Primary user and actors
| ID | Question | Status | Answer |
|---|---|---|---|
| Q1 | Who is this for? | ANSWERED | The author, for learning and research. Not a product, no other users. |
| Q2 | Is an agent a writer of the code? | ASSUMED | Yes. Code must be small and readable enough to review line by line, because understanding is the goal. |

## 2. Scope boundary
| ID | Question | Status | Answer |
|---|---|---|---|
| Q3 | What is built? | ANSWERED | A small masked-diffusion LM trained from scratch, plus an autoregressive model at the same scale, plus a benchmark comparing them. |
| Q4 | Which diffusion formulation? | ASSUMED | Masked (absorbing-state) discrete diffusion, MDLM-style continuous-time ELBO. It is the current default and the only one with a clean likelihood bound. Rejected: uniform-state, continuous embedding diffusion (Diffusion-LM). |
| Q5 | Other variants (block diffusion, remasking samplers)? | DEFERRED | After the first comparison. |
| Q6 | Downstream task evals (HellaSwag etc.)? | ASSUMED | Out. At ~100M params and ~1B tokens these sit near chance and add noise. |
| Q7 | Serving, UI, product? | ASSUMED | Out. |

## 3. Data model and identity
| ID | Question | Status | Answer |
|---|---|---|---|
| Q8 | Entities? | ASSUMED | Run (config + seed + git state), Checkpoint, EvalResult. A run is addressed by a directory name `runs/<model>-<yyyymmdd>-<seed>`. |
| Q9 | Tokenizer, shared across both models? | ASSUMED | GPT-2 BPE (50257 + one `[MASK]` token for the DLM). Same tokenizer for both so losses are comparable per token. |

## 4. State and storage
| ID | Question | Status | Answer |
|---|---|---|---|
| Q10 | Where do tokenized data, checkpoints and results live? | ASSUMED | Tokenized shards as `.bin` uint16 memmaps on the pod volume, copied out. Checkpoints and `results.json` copied back to `runs/` locally. Results are plain JSON/CSV so they diff. Checkpoints are git-ignored. |

## 5. Concurrency and conflict
| ID | Question | Status | Answer |
|---|---|---|---|
| Q11 | Two writers? | ASSUMED | Nothing concurrent. One run owns its directory. Resume from the last checkpoint only. |

## 6. Interfaces and contracts
| ID | Question | Status | Answer |
|---|---|---|---|
| Q12 | Surfaces? | ASSUMED | Python CLI scripts: `prep_data`, `train`, `eval`, `sample`, `report`. One YAML/dataclass config per run. |
| Q13 | Agent-friendly output? | ASSUMED | Every command exits non-zero on failure and writes machine-readable JSON. `report` reads only `results.json` files. |
| Q14 | Backbone shared between models? | ANSWERED-by-assumption | See F3. |

## 7. Failure behaviour
| ID | Question | Status | Answer |
|---|---|---|---|
| Q15 | Pod preempted or training diverges? | ASSUMED | Checkpoint every N steps, resumable. NaN loss aborts with non-zero exit and keeps the last good checkpoint. |
| Q16 | Eval metric unavailable (judge model fails to load)? | ASSUMED | Hard error. No partial report presented as complete. |

## 8. External dependencies
| ID | Question | Status | Answer |
|---|---|---|---|
| Q17 | Framework | ASSUMED | PyTorch 2.x (BSD-3), `torch.compile` and SDPA/FlashAttention. No Lightning, no HF Trainer. |
| Q18 | Tokenizer lib | ASSUMED | `tiktoken` (MIT) for GPT-2 BPE. Works offline once cached. |
| Q19 | Data source | FORK | See F2. |
| Q20 | Judge model for generative perplexity | ASSUMED | GPT-2 large (MIT licence, HF weights, runs on the pod). The standard choice in the MDLM paper, so numbers are comparable to the literature. |
| Q21 | Plotting/reporting | ASSUMED | matplotlib, output PNG + a `RESULTS.md`. |

## 9. Runtime and deployment
| ID | Question | Status | Answer |
|---|---|---|---|
| Q22 | Where does training run? | FORK | See F1. Provider assumed RunPod via `runpodctl` (already installed). |
| Q23 | Local role? | ASSUMED | Local machine only runs tests on tiny configs (CPU, a few million params, a few steps) and the report. |
| Q24 | Teardown | ASSUMED | Every pod launch uses the `runpod-jobs` dead-man's-switch pattern so a forgotten pod stops billing. |

## 10. Measurable success
| ID | Question | Status | Answer |
|---|---|---|---|
| Q25 | Pipeline correctness | ASSUMED | A tiny config overfits a 1k-sample set to loss under 0.1 nats/token for both models, on CPU, in under 5 minutes. |
| Q26 | Primary comparison metrics | ASSUMED | (a) validation NLL: exact for AR, ELBO upper bound for DLM, reported as perplexity with the bound caveat stated; (b) generative perplexity under GPT-2 large with sample entropy reported beside it, so low-entropy collapse is visible; (c) throughput in tokens/s at sequence length 1024, DLM at 64/128/256/1024 steps vs AR with KV cache. |
| Q27 | Seeds | ASSUMED | 1 seed per model for the main run (budget), 3 seeds only at the tiny scale to show noise. |
| Q28 | Fairness rule | ASSUMED | Same architecture family, same non-embedding parameter count, same tokens seen, same tokenizer, same optimizer schedule. Differences: causal vs bidirectional mask (see Q32: no timestep conditioning). |

## 11. Security and secrets
| ID | Question | Status | Answer |
|---|---|---|---|
| Q29 | Secrets | ASSUMED | RunPod API key and HF token only, from env vars, never in config, logs or git. Follow the `handling-secrets` skill. |

## 12. Versioning and migration
| ID | Question | Status | Answer |
|---|---|---|---|
| Q30 | Checkpoint/config format changes | DEFERRED | Config is stored beside every checkpoint, which is enough for a learning project. |

## Forks (closed, round 1 of 1)
| ID | Question | Status | Answer | Landed |
|---|---|---|---|---|
| F1 | Compute budget and scale | ANSWERED | About $60 total, ~125M params, rented single A100/H100 | ADR-0005, PLAN §Scope |
| F2 | Training data | ANSWERED | OpenWebText, ~2B-token subset, GPT-2 BPE, seq len 1024 | ADR-0006 |
| F3 | Own code vs reference repo | ANSWERED | Write it ourselves. Reference repos are test oracles only. User adds: after this exploratory milestone, go deeper toward a competitive system, probably an AR + DLM hybrid generating in blocks, chasing state of the art | ADR-0003, ADR-0002 |
| F4 | What "same scale" means | ANSWERED | Matched parameters and matched tokens seen, not wall-clock. Also log loss against wall-clock | ADR-0004 |

## Added after the round
| ID | Question | Status | Answer | Landed |
|---|---|---|---|---|
| Q31 | Phase 2 (hybrid, SOTA chase) | DEFERRED | Not planned here. $60 and 125M params cannot establish a SOTA claim, so phase 2 needs its own budget decision and its own plan run. This milestone only keeps the door open: the attention mask is a pluggable function (causal, bidirectional, block-causal) | ADR-0002 |
| Q32 | Timestep conditioning for the DLM | ASSUMED | None. Model infers noise level from the mask fraction, as LLaDA does and as "time-agnostic" analysis of masked diffusion supports. Keeps the AR and DLM backbones identical except the mask. Supersedes the timestep wording in Q28. Cost if wrong: small quality loss, add adaLN later | ADR-0001 |
| Q33 | Gumbel sampling precision | ASSUMED | float64 Gumbel/categorical sampling in the DLM sampler. Known issue: float32 sampling silently lowers effective temperature and flatters generative perplexity while entropy falls | ADR-0004 |
| Q34 | Data prep location | ASSUMED | Tokenize locally (16 cores, 829 GB free), upload ~4 GB of shards to the pod volume. Avoids paying GPU rates for CPU work | SLICES V2 |
| Q35 | Sequence packing | ASSUMED | Concatenate documents with an EOS separator, cut into 1024-token chunks, no padding, for both models | ADR-0006 |

## Coverage
| Category | Covered by |
|---|---|
| Primary user and actors | Q1, Q2 |
| Scope boundary | Q3-Q7, Q31 |
| Data model and identity | Q8, Q9, Q35 |
| State and storage | Q10 |
| Concurrency and conflict | Q11 |
| Interfaces and contracts | Q12-Q14 |
| Failure behaviour | Q15, Q16 |
| External dependencies | Q17-Q21, F2 |
| Runtime and deployment | Q22-Q24, F1, Q34 |
| Measurable success | Q25-Q28, Q32, Q33, F4 |
| Security and secrets | Q29 |
| Versioning and migration | Q30 |

# Diffusion Language Model: Slices

Vertical increments. Each ends in something you can demonstrate. Slice 1 confronts the riskiest unknown, which is whether our DLM loss and sampler are actually correct.

## V1: Tiny models overfit, and the DLM loss matches an oracle

**Delivers:** R1, R2, R7, part of R3

**Build plan**

1. `uv` project, `pyproject.toml`, pytest, a `tiny` config (about 2M parameters, context 64, vocabulary 256 byte-level for tests only).
2. Backbone with the pluggable attention mask: causal, bidirectional, block-causal (ADR-0002).
3. AR loss and DLM loss with stratified `t` (ADR-0001).
4. A standalone reference ELBO written as a separate, slow, obviously-correct function in the test directory, plus an exact enumeration check on a 3-token, 4-symbol toy.
5. DLM ancestral sampler with float64 noise, and AR sampler.
6. Minimal train loop on CPU that runs a fixed number of steps and writes a run directory.

**Demo:** run `train` for both models on a 1,000-sequence toy set. Both loss curves fall below 0.1 nats per token. Then `sample` prints text for each.

**Rests on assumptions:** Q4 (masked diffusion). If wrong, this slice is the cheapest place to find out. Q32 (no timestep input): the overfit test uses it, and if the DLM cannot overfit without it that is evidence against the default.

### Test plan

#### End-to-end
- `train` on the toy set reaches loss below 0.1 nats per token for AR within 5 minutes on CPU.
- Same for the DLM, with the ELBO evaluated on its own training data below 0.1 nats per token.
- `sample` for the overfit DLM at 64 steps reproduces a memorised training sequence with at least 90 percent token accuracy.

#### Integration
- Our DLM loss equals the exact ELBO from enumeration on the 3-token toy, within 1e-3 averaged over many `t` draws.
- Our DLM loss equals the ported MDLM reference formula on random logits, within 1e-5.
- An untrained model's ELBO is within 2 percent of `ln(V)` per token.
- Stratified `t` has lower variance of the batch loss than iid `t` on the same data, measured over 200 batches.
- Resuming from a checkpoint gives bit-identical loss to an uninterrupted run for 20 steps.

#### Unit
- Causal mask: position `i` attends only to `j <= i`. Bidirectional attends everywhere.
- Block-causal with block size 4: full attention inside a block, causal across blocks.
- Mask corruption: with `t=0` nothing is masked, with `t=1` everything is masked, and the masked fraction over many draws is about `t`.
- Parameter count of the DLM exceeds AR by exactly one embedding row (and its tied output row if tied).
- Sampler never changes a token once it is unmasked, and ends with zero `[MASK]` tokens.
- float64 sampling path is actually float64 (dtype assertion).

## V2: OpenWebText becomes training shards

**Delivers:** R3 (data part)

**Build plan**

1. `prep_data` streams OpenWebText, shuffles documents with a fixed seed, tokenizes with `tiktoken` GPT-2 BPE across all 16 cores, appends EOS per document.
2. Stop at about 2.0B training tokens, split off a 5M-token validation set by document.
3. Write uint16 shards and `manifest.json` (counts, seed, tokenizer, SHA-256).
4. A loader that reads shards as memmaps and yields 1024-token chunks with a deterministic order from a seed.

**Demo:** `prep_data` prints the manifest. Decode three random chunks and read them as recognisable web text with EOS between documents.

**Rests on assumptions:** Q34 (tokenize locally, 829 GB free is plenty), Q35 (packing without boundary masks).

### Test plan

#### End-to-end
- `manifest.json` reports train tokens between 1.95B and 2.05B and validation tokens between 4.9M and 5.1M.
- No document appears in both train and validation (checked by document-id hash sets).
- Re-running `prep_data` with the same seed produces identical shard SHA-256 values.

#### Integration
- The loader with seed 0 yields the same first 100 chunks across two runs, and a different order with seed 1.
- Decoding then re-encoding a sample chunk reproduces the token ids.

#### Unit
- Packing: documents `[a,b]` and `[c]` become `a b EOS c EOS` with no padding, and the chunk boundary falls at exactly 1024 tokens.
- Token ids never exceed 50256 in shards (the `[MASK]` id is added by the DLM at runtime, not stored).

## V3: One pod, one smoke run, verified teardown

**Delivers:** R3, R4

**Build plan**

1. Pod runner using the `runpod-jobs` pattern, with the dead-man's-switch and the `budget.json` ledger and refusal rule.
2. Upload shards to the pod volume (about 4 GB).
3. Run the full 125M configuration for 200 steps for each model, with bf16, `torch.compile`, checkpoint and resume.
4. Measure tokens per second, memory, and compute the projected GPU-hours and cost for 2B tokens per model.
5. Decide batch size and LR schedule for the long runs from what was measured.

**Demo:** `pod status` shows the ledger. After the run, `runpodctl` lists zero running pods and the ledger total is under $3.

**Rests on assumptions:** ADR-0005 sizing (4 to 6 hours per model). If the projection exceeds $22 per model, drop to about 60M parameters before continuing.

### Test plan

#### End-to-end
- The smoke run completes, writes `results.json` with tokens per second and projected cost, and the pod is gone afterwards (checked with `runpodctl`).
- Killing the job halfway and relaunching with `--resume` continues from the saved step with a loss within 1 percent of where it stopped.
- A second `pod launch` with the ledger set artificially to $59 is refused with the budget exit code.

#### Integration
- The dead-man's-switch terminates an idle pod within its configured timeout, tested with a 10-minute timeout and a pod doing nothing.
- Shard SHA-256 values on the pod match the manifest before training starts.

#### Unit
- Ledger arithmetic: summing intervals and rates gives the expected total, and a pod with no recorded stop counts as running up to now.
- Budget gate refuses when `spent + planned > cap`, accepts when equal.

## V4: Both full runs finish with sane curves

**Delivers:** R0 (training half), R1, R3, R4

**Build plan**

1. Launch the AR run for about 2B tokens. Gate: stop and review before continuing if validation loss is not on a sensible curve.
2. If the gate passes, launch the DLM run with the same schedule and data seed.
3. Log training loss and validation NLL or ELBO against tokens and against wall-clock.
4. Copy checkpoints and logs back to `runs/` locally and confirm teardown.

**Demo:** two curves on one plot of validation loss against tokens seen, one per model.

**Rests on assumptions:** the shared learning rate transfers from 10M to 125M (ADR-0004). If either run diverges or plateaus oddly, the reserve in the budget allows one rerun of one model at a lowered LR, and that limitation must be written into the report.

### Test plan

#### End-to-end
- AR final validation loss is at most 3.6 nats per token, or the run is investigated before the DLM starts. This is an expectation from published nanoGPT-scale results at this token count, not a measured fact, so revise it from V3 data if it looks off.
- DLM validation ELBO is finite and decreasing across checkpoints, with final bound perplexity at most 80, or the run is investigated before the results are used.
- Both runs processed within 1 percent of the same token count.
- After each run `runpodctl` shows no pod, and the ledger total is at most $60.

#### Integration
- Checkpoints from each run reload on the local CPU and reproduce the logged validation loss on a 1M-token slice within 1 percent.

#### Unit
- Tokens-seen counter equals steps times batch tokens, and survives a resume.
- NaN loss aborts with the NaN exit code and leaves the previous checkpoint intact.

## V5: Benchmark, noise floor and report

**Delivers:** R0, R5, R6

**Build plan**

1. `eval`: validation NLL or ELBO on the fixed 5M-token set with a fixed seed.
2. Generate 256 samples per setting (AR; DLM at 64, 256 and 1024 steps) and compute gen-ppl under GPT-2 large and mean entropy.
3. Throughput benchmark on one GPU model, batch size 1 and at saturation, AR with KV cache.
4. Three-seed noise-floor runs at about 10M parameters for each model (small pod time).
5. `report` renders `RESULTS.md` with tables, plots and the caveats ADR-0004 requires.

**Demo:** open `RESULTS.md` and read the three tables (likelihood, gen-ppl with entropy, throughput) plus the loss-versus-tokens and loss-versus-wall-clock plots.

**Rests on assumptions:** Q20 (GPT-2 large judge) and Q33 (float64 sampling).

### Test plan

#### End-to-end
- `report` run on its own, with no GPU, regenerates `RESULTS.md` from the `results.json` files and yields byte-identical text on a second run.
- `RESULTS.md` contains the bound caveat in the likelihood table, an entropy column beside every gen-ppl number, and the one-seed caveat for full-scale runs.
- Noise-floor table shows seed-to-seed spread for each model, and any reported gap in the headline is marked "within noise" when it is smaller than that spread.
- Total ledger spend is at most $60 after all evals, and no pod is running.

#### Integration
- Gen-ppl of real validation text (not generated) under GPT-2 large falls in a plausible range (about 20 to 60), which sanity-checks the judge setup before judging models.
- DLM float32 versus float64 sampling shows the expected entropy gap on the same checkpoint, confirming the precision issue is real in our setup.
- Same checkpoint and seed give the same gen-ppl twice.

#### Unit
- Entropy calculation of a uniform-random token string over a vocabulary of 8 equals ln 8.
- Report refuses to render if a required `results.json` field is missing, with a non-zero exit.

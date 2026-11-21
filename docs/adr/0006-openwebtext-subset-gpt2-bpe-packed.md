# ADR-0006: Train on a 2B-token OpenWebText subset, GPT-2 BPE, packed to 1024 tokens

- Status: Accepted
- Date: 2026-11-22
- Deciders: Jian (owner), agent

## Context
OpenWebText is the corpus used in the MDLM paper, so numbers can be sanity-checked against published ones. Our token count is far below theirs, so absolute numbers will be worse, which is expected.

## Decision
- Source: the OpenWebText corpus from Hugging Face datasets. Take documents in a fixed shuffled order (seeded) until about 2.0B tokens, and hold out 5M tokens as validation, split by document.
- Tokenizer: GPT-2 BPE through `tiktoken`, vocabulary 50257, plus `[MASK]` as id 50257 for the DLM. EOS (id 50256) separates documents.
- Packing: concatenate with EOS and cut into 1024-token chunks, no padding, no cross-document attention masking (the same simplification for both models, and the one the MDLM paper uses for OWT).
- Storage: `uint16` memmap shards (`train_000.bin` and so on) plus a `manifest.json` with token counts, tokenizer name, seed and a SHA-256 of each shard.
- Licensing: OpenWebText has no formal licence, it is scraped Reddit-linked web text. Used for private research only. The shards and model weights are not published, and anything published says so.

## Alternatives considered
| Option | Why not |
|--------|---------|
| TinyStories | Cheap, but too narrow for a meaningful gen-ppl and entropy comparison |
| FineWeb-Edu | Better data quality, but loses the published-number sanity check |
| Padded sequences | Wastes compute and gives AR and DLM different effective token counts |

## Consequences
- No document-boundary attention masking means some cross-document leakage in context. Same for both models, so fair, and noted as a limitation.
- Shards are about 4 GB, small enough to upload to a pod volume.

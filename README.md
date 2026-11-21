# diffusion-language-model

A from-scratch masked diffusion language model (MDLM-style) and an autoregressive baseline, trained on the
same OpenWebText tokens at ~125M parameters and compared under a fixed protocol: validation likelihood,
generative perplexity with entropy, and throughput.

**Status: planning done, V1 not started.** Nothing here trains yet. The longer-term aim is an
AR + diffusion block hybrid; this milestone builds the trusted parts that needs.

- [Plan](docs/PLAN.md) · [Slices](docs/SLICES.md) · [Decision register](docs/QUESTIONS.md) · [ADRs](docs/adr/)

## Develop

```bash
make hooks     # install the pre-push gate
make install   # uv sync, CPU-only torch
make check     # lint + strict types + fast tests, the same gate CI runs
```

Training runs on a rented GPU under a $60 budget cap (ADR-0005). The local machine only runs CPU tests.

## Licence

Apache-2.0. OpenWebText is not redistributed here, and trained weights are kept private (ADR-0006).

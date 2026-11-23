"""GPU bisect for the resume offset: which of {compile, bf16} makes a resumed run drift from an
uninterrupted one? Runs in one process on the pod, prints DIAG lines. Needs data_dir to exist."""

import json
import sys

from dlm.config import Config
from dlm.train import train

STEPS, SPLIT = 60, 30


def losses(cfg: Config) -> dict[int, float]:
    recs = [json.loads(x) for x in (cfg.run_dir / "log.jsonl").read_text().splitlines()]
    return {r["step"]: r["loss"] for r in recs}


def make(base: dict, kind: str, compile_: bool, bf16: bool, name: str) -> Config:  # type: ignore[type-arg]
    train_cfg = dict(base["train"], steps=STEPS, ckpt_every=SPLIT, eval_every=0)
    train_cfg.update(compile=compile_, bf16=bf16)
    tag = f"{kind}_c{int(compile_)}_b{int(bf16)}"
    return Config.from_dict(dict(base, kind=kind, run_name=f"{tag}_{name}", train=train_cfg))


def main() -> int:
    base = Config.from_yaml(sys.argv[1]).to_dict()
    combos = [
        ("dlm", True, True),
        ("dlm", True, False),
        ("dlm", False, True),
        ("dlm", False, False),
        ("ar", True, True),
    ]
    for kind, compile_, bf16 in combos:
        ref = make(base, kind, compile_, bf16, "ref")
        split = make(base, kind, compile_, bf16, "split")
        train(ref, device="cuda")
        train(split, stop_after=SPLIT, device="cuda")
        train(split, resume=True, device="cuda")
        a, b = losses(ref), losses(split)
        diffs = {s: f"{(b[s] - a[s]) / a[s]:+.2%}" for s in (40, 50, 60)}
        print(
            f"DIAG {kind} compile={compile_} bf16={bf16} resumed-vs-straight: {diffs}", flush=True
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

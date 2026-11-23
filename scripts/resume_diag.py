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


def main() -> int:
    base = Config.from_yaml(sys.argv[1]).to_dict()
    combos = [("dlm", True, True), ("dlm", True, False), ("dlm", False, True), ("dlm", False, False),
              ("ar", True, True)]  # fmt: skip
    for kind, compile_, bf16 in combos:
        tag = f"{kind}_c{int(compile_)}_b{int(bf16)}"

        def make(name: str) -> Config:
            d = dict(base, kind=kind, run_name=f"{tag}_{name}")
            d["train"] = dict(
                base["train"], steps=STEPS, ckpt_every=SPLIT, eval_every=0,
                compile=compile_, bf16=bf16,
            )  # fmt: skip
            return Config.from_dict(d)

        ref, split = make("ref"), make("split")
        train(ref, device="cuda")
        train(split, stop_after=SPLIT, device="cuda")
        train(split, resume=True, device="cuda")
        a, b = losses(ref), losses(split)
        diffs = {s: f"{(b[s] - a[s]) / a[s]:+.2%}" for s in (40, 50, 60)}
        print(f"DIAG {tag} resumed-vs-straight rel diff by step: {diffs}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

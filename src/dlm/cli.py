"""Command line. Every command prints one JSON object on stdout; logs go to stderr."""

import argparse
import json
import sys
from pathlib import Path

import torch

from dlm.budget import BudgetExceededError, Ledger
from dlm.config import Config
from dlm.data import memorisation_accuracy, toy_dataset
from dlm.diffusion import sample_ar, sample_dlm
from dlm.owt import EOS_ID, SOURCE, gpt2_encoder, owt_documents
from dlm.pod import PodError, PodPlan, launch, stop
from dlm.runpod import ApiError, Client
from dlm.shards import build_shards
from dlm.train import NaNLossError, build_model, load_checkpoint, train

EXIT_NAN = 3
EXIT_BUDGET = 4
EXIT_POD = 5
LEDGER = Path("budget.json")


def _cmd_train(a: argparse.Namespace) -> int:
    cfg = Config.from_yaml(a.config)
    try:
        out = train(cfg, resume=a.resume, device=a.device)
    except NaNLossError as e:
        print(json.dumps({"error": str(e)}))
        return EXIT_NAN
    print(json.dumps(out))
    return 0


def _cmd_sample(a: argparse.Namespace) -> int:
    run = Path(a.run)
    cfg = Config.from_yaml(run / "config.yaml")
    model = build_model(cfg).to(a.device)
    model.load_state_dict(load_checkpoint(run)["model"])
    model.eval()
    gen = torch.Generator(device=a.device).manual_seed(a.seed)
    length = cfg.data.seq_len
    train_data = toy_dataset(cfg.data.n_sequences, length, cfg.base_vocab, cfg.seed)
    if cfg.kind == "dlm":
        start = None
        if a.visible > 0:  # infill from training sequences with this fraction of tokens visible
            keep = (
                torch.rand(train_data[: a.n].shape, generator=torch.Generator().manual_seed(a.seed))
                < a.visible
            )
            start = torch.where(
                keep, train_data[: a.n], torch.full_like(train_data[: a.n], cfg.mask_id)
            )
        x = sample_dlm(model, a.n, length, cfg.mask_id, a.steps, gen, x_init=start)
    else:
        x = sample_ar(model, a.n, length, 0, gen)
    acc = memorisation_accuracy(x.cpu(), train_data)
    print(json.dumps({"samples": x.tolist(), "memorisation_accuracy": acc}))
    return 0


def _cmd_prep_data(a: argparse.Namespace) -> int:
    m = build_shards(
        owt_documents(a.seed, Path(a.raw), keep_raw=a.keep_raw),
        gpt2_encoder(),
        Path(a.out),
        train_tokens=a.train_tokens,
        val_tokens=a.val_tokens,
        shard_tokens=a.shard_tokens,
        seq_len=a.seq_len,
        eos=EOS_ID,
        seed=a.seed,
        tokenizer="gpt2",
        source=SOURCE,
    )
    print(json.dumps({k: m[k] for k in ("train", "val", "seed", "tokenizer")} | {"out": a.out}))
    return 0


def _cmd_pod(a: argparse.Namespace) -> int:
    ledger = Ledger(LEDGER)
    try:
        if a.action == "status":
            out = ledger.summary()
        elif a.action == "launch":
            out = launch(PodPlan.from_yaml(a.plan), ledger, Client())
        else:
            out = stop(PodPlan.from_yaml(a.plan), ledger, Client())
    except BudgetExceededError as e:
        print(json.dumps({"error": str(e)}))
        return EXIT_BUDGET
    except (PodError, ApiError) as e:
        print(json.dumps({"error": str(e)}))
        return EXIT_POD
    print(json.dumps(out))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dlm")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--device", default="cpu")
    t.set_defaults(fn=_cmd_train)
    s = sub.add_parser("sample")
    s.add_argument("--run", required=True)
    s.add_argument("--n", type=int, default=8)
    s.add_argument("--steps", type=int, default=64)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument(
        "--visible", type=float, default=0.0, help="DLM only: infill from this visible fraction"
    )
    s.add_argument("--device", default="cpu")
    s.set_defaults(fn=_cmd_sample)
    d = sub.add_parser("prep-data")
    d.add_argument("--out", default="data/owt")
    d.add_argument("--raw", default="data/raw")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--train-tokens", type=int, default=2_000_000_000)
    d.add_argument("--val-tokens", type=int, default=5_000_000)
    d.add_argument("--shard-tokens", type=int, default=244140 * 1024)
    d.add_argument("--seq-len", type=int, default=1024)
    d.add_argument("--keep-raw", action="store_true")
    d.set_defaults(fn=_cmd_prep_data)
    pd = sub.add_parser("pod")
    pd.add_argument("action", choices=["launch", "stop", "status"])
    pd.add_argument("--plan", help="pod plan YAML (launch and stop)")
    pd.set_defaults(fn=_cmd_pod)
    a = p.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    sys.exit(main())

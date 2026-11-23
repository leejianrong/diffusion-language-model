"""Run configuration. One YAML file per run, copied into the run directory."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

Kind = Literal["ar", "dlm"]


@dataclass(frozen=True)
class ModelConfig:
    block_size: int = 32
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    abs_pos: bool = False


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1000
    batch_size: int = 64
    lr: float = 3e-3
    warmup: int = 50
    weight_decay: float = 0.0
    ckpt_every: int = 100
    log_every: int = 50
    eval_every: int = 0
    grad_accum: int = 1  # micro-batches per optimiser step; batch_size is the full batch
    bf16: bool = False  # autocast forward to bfloat16 (loss and optimiser stay float32)
    compile: bool = False
    eval_seqs: int = 256  # validation sequences used when training on shards
    eval_batch_size: int = 256


@dataclass(frozen=True)
class ToyDataConfig:
    """Random memorisable sequences, each starting with token 0 (the AR start token)."""

    n_sequences: int = 1000
    seq_len: int = 32


@dataclass(frozen=True)
class Config:
    kind: Kind = "ar"
    base_vocab: int = 64  # real tokens; the DLM adds one [MASK] row on top
    seed: int = 0
    run_root: str = "runs"
    run_name: str = ""
    data_dir: str = ""  # shard directory (V2); empty means the toy memorisation set
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: ToyDataConfig = field(default_factory=ToyDataConfig)

    @property
    def vocab_size(self) -> int:
        return self.base_vocab + (1 if self.kind == "dlm" else 0)

    @property
    def mask_id(self) -> int:
        if self.kind != "dlm":
            raise ValueError("only the DLM has a [MASK] token")
        return self.base_vocab

    @property
    def attention(self) -> str:
        return "causal" if self.kind == "ar" else "bidirectional"

    @property
    def run_dir(self) -> Path:
        return Path(self.run_root) / (self.run_name or f"{self.kind}-{self.seed}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Config":
        d = dict(d)
        return Config(
            model=ModelConfig(**d.pop("model", {})),
            train=TrainConfig(**d.pop("train", {})),
            data=ToyDataConfig(**d.pop("data", {})),
            **d,
        )

    @staticmethod
    def from_yaml(path: str | Path) -> "Config":
        return Config.from_dict(yaml.safe_load(Path(path).read_text()))

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=True))

"""配置：一个 yaml 管完，命令行可以覆盖关键项。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelCfg:
    name: str = "Qwen/Qwen2.5-7B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    attn_implementation: str = "sdpa"   # sdpa | flash_attention_2
    trust_remote_code: bool = True


@dataclass
class LoraCfg:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class TrainCfg:
    output_dir: str = "outputs/qwen2.5-7b-nl2sql"
    epochs: float = 3.0
    max_steps: int = -1
    per_device_batch_size: int = 1
    grad_accum: int = 8
    lr: float = 2.0e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 2
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    seed: int = 42
    report_to: list[str] = field(default_factory=list)


@dataclass
class DataCfg:
    train_file: str = "data/demo/train.jsonl"
    dev_file: str = "data/demo/dev.jsonl"
    max_len: int = 2048
    include_samples: bool = True
    eval_ratio: float = 0.0


@dataclass
class Config:
    model: ModelCfg = field(default_factory=ModelCfg)
    lora: LoraCfg = field(default_factory=LoraCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    data: DataCfg = field(default_factory=DataCfg)
    db_root: str = ""   # 若 jsonl 里的 db_path 是相对的，用它拼前缀

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        cfg = cls()
        for section, dataclass_type in (
            ("model", ModelCfg), ("lora", LoraCfg),
            ("train", TrainCfg), ("data", DataCfg),
        ):
            values = raw.get(section) or {}
            obj = dataclass_type()
            for k, v in values.items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
            setattr(cfg, section, obj)
        cfg.db_root = raw.get("db_root", "")
        return cfg

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

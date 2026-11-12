"""数据层：jsonl → token → 带 prompt mask 的训练样本。

重点在 labels 的 mask：只对 assistant 的 SQL 部分算 loss。如果不 mask，
模型会把大量 token 花在复述 schema 上，7B 模型在 24G 卡上收敛明显变慢，
而且学出来的是"抄 schema"而不是"写 SQL"。这是很多人第一次微调踩的坑。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .prompt import build_messages, build_messages_with_answer
from .schema import cached_schema, to_ddl


@dataclass
class Example:
    db_id: str
    db_path: str
    question: str
    sql: str


def read_jsonl(path: str | Path) -> list[Example]:
    """读 jsonl；db_path 为相对路径时按 jsonl 文件所在目录解析，
    这样整个 data/ 目录拷到别的机器上还能直接用。"""
    p = Path(path)
    out: list[Example] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            db_path = Path(r["db_path"])
            if not db_path.is_absolute():
                db_path = (p.parent / db_path).resolve()
            out.append(Example(db_id=r["db_id"], db_path=str(db_path),
                               question=r["question"], sql=r["sql"]))
    return out


def attach_schema(examples: Iterable[Example], include_samples: bool = True) -> list[dict[str, Any]]:
    """给每条样本挂上 DDL 文本（同一个库会命中 lru_cache）。"""
    out = []
    for ex in examples:
        sch = cached_schema(ex.db_path)
        ddl = to_ddl(sch, include_samples=include_samples)
        out.append({"db_id": ex.db_id, "db_path": ex.db_path,
                    "question": ex.question, "sql": ex.sql, "ddl": ddl})
    return out


def encode_example(tok, item: dict[str, Any], max_len: int = 2048) -> dict[str, list[int]] | None:
    """返回 input_ids / attention_mask / labels；超长直接丢弃（返回 None）。"""
    msgs_prompt = build_messages(item["ddl"], item["question"])
    msgs_full = build_messages_with_answer(item["ddl"], item["question"], item["sql"])

    kwargs = dict(tokenize=True, return_dict=False, add_generation_prompt=False)
    ids_full = tok.apply_chat_template(msgs_full, **kwargs)
    ids_prompt = tok.apply_chat_template(
        msgs_prompt, tokenize=True, return_dict=False, add_generation_prompt=True
    )

    # apply_chat_template 的 add_generation_prompt 与完整对话前缀一般严格对齐；
    # 若某些模板在末尾多/少 token，退化成"答案单独编码 + mask 前缀长度"。
    if ids_full[: len(ids_prompt)] == ids_prompt and len(ids_prompt) < len(ids_full):
        n_prompt = len(ids_prompt)
    else:
        ans = tok(item["sql"].strip(), add_special_tokens=False)["input_ids"]
        n_prompt = max(0, len(ids_full) - len(ans) - 1)

    if len(ids_full) > max_len:
        return None

    labels = [-100] * n_prompt + ids_full[n_prompt:]
    if all(l == -100 for l in labels):
        return None
    return {"input_ids": ids_full, "attention_mask": [1] * len(ids_full), "labels": labels}


class Collator:
    """右 padding；labels 用 -100 填充。训练用 pad_side='right'。"""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attn, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_tokenized(samples: list[dict[str, Any]], tok, max_len: int,
                    seed: int = 0) -> tuple[list[dict[str, list[int]]], dict[str, int]]:
    random.Random(seed).shuffle(samples)
    out, dropped = [], 0
    for s in samples:
        enc = encode_example(tok, s, max_len=max_len)
        if enc is None:
            dropped += 1
            continue
        out.append(enc)
    stats = {"kept": len(out), "dropped_too_long": dropped}
    return out, stats

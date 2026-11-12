"""推理：批量生成 SQL，可选"执行结果自一致性"投票。

自一致性（self-consistency）在这里有个天然的正确性信号：SQL 对不对，
跑一遍就知道。采样 k 条，按【执行结果指纹】投票而不是按文本投票——
文本不同的两条 SQL 可能结果相同（都算对），这是比文本投票更靠谱的聚合。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .config import Config
from .data import read_jsonl
from .prompt import build_messages
from .schema import cached_schema, to_ddl
from .sql_utils import extract_sql, result_fingerprint, run_sql, sql_complexity


def _load_model(cfg: Config, adapter: str | None, base_override: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    base = base_override or cfg.model.name
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=cfg.model.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # 生成必须左 padding，否则末尾 token 不是有效续写位置

    kwargs: dict = {
        "trust_remote_code": cfg.model.trust_remote_code,
        "attn_implementation": cfg.model.attn_implementation,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        print(f"[infer] adapter loaded: {adapter}", flush=True)
    model.eval()
    return model, tok


def _select_items(items, complexity: str | None = None, limit: int = 0):
    """按 gold SQL 的难度筛选子集。hard 子集是判断"模型到顶没有"最快的方式。"""
    if complexity:
        items = [it for it in items if sql_complexity(it.sql) == complexity]
    if limit:
        items = items[:limit]
    return items


def _prompts(items: list[dict], include_samples: bool) -> list[str]:
    out = []
    for it in items:
        ddl = to_ddl(cached_schema(it.db_path), include_samples=include_samples)
        msgs = build_messages(ddl, it.question)
        out.append(msgs)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qlora_7b.yaml")
    ap.add_argument("--test-file", default="data/demo/test.jsonl")
    ap.add_argument("--adapter", default=None, help="LoRA 目录；不填=基座模型（做 baseline）")
    ap.add_argument("--base", default=None, help="覆盖基座模型名")
    ap.add_argument("--output", default="outputs/pred.jsonl")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--num-samples", type=int, default=1, help=">1 时启用结果一致性投票")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--filter-complexity", default=None, choices=["easy", "medium", "hard"],
                    help="只测某个难度子集（按 gold SQL 判定）")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    items = _select_items(read_jsonl(args.test_file), args.filter_complexity, args.limit)
    if not items:
        print("no test items", file=sys.stderr)
        return 1

    import torch

    model, tok = _load_model(cfg, args.adapter, args.base)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    if args.cpu:
        model = model.to("cpu")

    messages = _prompts(items, cfg.data.include_samples)
    prompts = [
        tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages
    ]

    k = max(1, args.num_samples)
    do_sample = k > 1 and args.temperature > 0
    records = []

    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i: i + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=cfg.data.max_len).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=0.95 if do_sample else None,
                num_return_sequences=k,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new_tokens, skip_special_tokens=True)

        for j, item in enumerate(items[i: i + args.batch_size]):
            cand = texts[j * k: (j + 1) * k]
            sqls = [extract_sql(t) for t in cand]
            pred = _vote(sqls, item.db_path) if k > 1 else (sqls[0] if sqls else "")
            records.append({
                "db_id": item.db_id,
                "db_path": item.db_path,
                "question": item.question,
                "gold_sql": item.sql,
                "pred_sql": pred,
                "candidates": sqls if k > 1 else None,
                "raw": cand[0] if k == 1 else None,
            })
        print(f"[infer] {min(i + args.batch_size, len(prompts))}/{len(prompts)}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[infer] wrote {len(records)} -> {out}")
    return 0


def _vote(sqls: list[str], db_path: str) -> str:
    """按执行结果投票；全错时退回第一条。"""
    buckets: dict[str, list[int]] = {}
    for idx, sql in enumerate(sqls):
        if not sql:
            continue
        res = run_sql(db_path, sql, timeout_s=10)
        if not res.ok:
            continue
        buckets.setdefault(result_fingerprint(res.rows), []).append(idx)
    if not buckets:
        return sqls[0] if sqls else ""
    best = max(buckets.values(), key=len)
    return sqls[best[0]]


if __name__ == "__main__":
    sys.exit(main())

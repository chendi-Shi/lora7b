"""Qwen2.5 NL2SQL QLoRA 训练入口。

用法（4090 24G）：
    python -m nl2sql.train --config configs/qlora_7b.yaml
    python -m nl2sql.train --config configs/qlora_7b.yaml --max-steps 50   # 冒烟
    python -m nl2sql.train --config configs/qlora_7b.yaml --dry-run        # 只验数据

为什么不用 trl 的 SFTTrainer：它的参数在 0.9→0.12 之间反复改
（max_seq_length→max_length、dataset_text_field 语义变化、packing 默认值），
换台机器就炸。这里直接用 transformers.Trainer + 自己写的 collator，
依赖面小，labels 怎么 mask 完全可控。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from .config import Config
from .data import Collator, attach_schema, build_tokenized, read_jsonl


def _set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _log(msg: str) -> None:
    print(f"[train {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_model(cfg: Config):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=cfg.model.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    kwargs: dict = {
        "trust_remote_code": cfg.model.trust_remote_code,
        "attn_implementation": cfg.model.attn_implementation,
    }
    if torch.cuda.is_available() and cfg.model.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.model.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, cfg.model.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=cfg.model.bnb_4bit_use_double_quant,
        )
        kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, **kwargs)
    model.config.use_cache = False

    if cfg.model.load_in_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.train.gradient_checkpointing
        )

    lora = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    if cfg.train.gradient_checkpointing:
        # gradient checkpointing + LoRA 必须让输入参与求导，否则 adapter 梯度为空
        model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    _log(f"trainable params: {trainable/1e6:.2f}M ({100*trainable/total:.2f}% of {total/1e9:.2f}B)")
    return model, tok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="QLoRA fine-tune Qwen for NL2SQL")
    ap.add_argument("--config", default="configs/qlora_7b.yaml")
    ap.add_argument("--train-file", default=None)
    ap.add_argument("--dev-file", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条训练样本（调试）")
    ap.add_argument("--dry-run", action="store_true", help="只跑数据管线，不加载模型")
    ap.add_argument("--model", default=None, help="覆盖基座，如 Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--cpu", action="store_true", help="CPU 调试（务必配 --model 用小模型）")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    if args.train_file:
        cfg.data.train_file = args.train_file
    if args.dev_file:
        cfg.data.dev_file = args.dev_file
    if args.output_dir:
        cfg.train.output_dir = args.output_dir
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.max_len:
        cfg.data.max_len = args.max_len
    if args.model:
        cfg.model.name = args.model
    if args.cpu or args.no_4bit:
        cfg.model.load_in_4bit = False
        cfg.model.attn_implementation = "sdpa"
    if args.cpu:
        cfg.train.bf16 = False
        cfg.train.optim = "adamw_torch"

    _set_seed(cfg.train.seed)
    _log(f"config: {args.config}")
    _log(f"train: {cfg.data.train_file} | dev: {cfg.data.dev_file} | max_len: {cfg.data.max_len}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=cfg.model.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ex = read_jsonl(cfg.data.train_file)
    if args.limit:
        train_ex = train_ex[: args.limit]
    train_samples = attach_schema(train_ex, include_samples=cfg.data.include_samples)
    train_enc, st = build_tokenized(train_samples, tok, cfg.data.max_len, seed=cfg.train.seed)
    _log(f"tokenized: {st}")
    if not train_enc:
        _log("ERROR: no usable training sample (max_len too small?)")
        return 1
    lens = sorted(len(x["input_ids"]) for x in train_enc)
    _log(f"seq len p50={lens[len(lens)//2]} p95={lens[int(len(lens)*0.95)]} max={lens[-1]}")

    dev_enc = []
    if Path(cfg.data.dev_file).exists():
        dev_samples = attach_schema(read_jsonl(cfg.data.dev_file),
                                    include_samples=cfg.data.include_samples)
        dev_enc, dev_st = build_tokenized(dev_samples, tok, cfg.data.max_len, seed=cfg.train.seed)
        _log(f"dev tokenized: {dev_st}")

    if args.dry_run:
        batch = Collator(tok.pad_token_id)(train_enc[:2])
        _log(f"dry-run ok, batch shapes: "
             f"{ {k: tuple(v.shape) for k, v in batch.items()} }")
        _log(f"masked ratio: {(batch['labels'] == -100).float().mean().item():.2f}")
        _log("data pipeline OK")
        return 0

    import torch
    from transformers import Trainer, TrainingArguments

    model, tok = build_model(cfg)

    ta = TrainingArguments(
        output_dir=cfg.train.output_dir,
        num_train_epochs=cfg.train.epochs,
        max_steps=cfg.train.max_steps,
        per_device_train_batch_size=cfg.train.per_device_batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=cfg.train.lr,
        lr_scheduler_type=cfg.train.lr_scheduler,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        max_grad_norm=cfg.train.max_grad_norm,
        logging_steps=cfg.train.logging_steps,
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        bf16=cfg.train.bf16 and torch.cuda.is_available(),
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=cfg.train.optim,
        report_to=cfg.train.report_to,
        seed=cfg.train.seed,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=ta,
        train_dataset=train_enc,
        eval_dataset=dev_enc or None,
        data_collator=Collator(tok.pad_token_id),
    )

    if torch.cuda.is_available():
        _log(f"GPU: {torch.cuda.get_device_name(0)} | "
             f"before: {torch.cuda.memory_allocated()/2**30:.2f} GiB")

    trainer.train(resume_from_checkpoint=args.resume)

    out = Path(cfg.train.output_dir)
    trainer.save_model(str(out))
    tok.save_pretrained(out)
    (out / "train_config.json").write_text(
        json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if torch.cuda.is_available():
        _log(f"peak reserved: {torch.cuda.max_memory_reserved()/2**30:.2f} GiB")
    _log(f"saved adapter -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

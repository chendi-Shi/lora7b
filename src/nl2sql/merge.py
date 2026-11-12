"""合并 LoRA 权重 -> 完整模型。

注意：4bit 量化的基座不能直接 merge（权重是 nf4 的），所以这里始终用
bf16/fp16 重新加载基座，再 merge_and_unload。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--push-to-hub", default=None, help="可选：推送到 HF 仓库名")
    args = ap.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    print(f"[merge] base={args.base} adapter={args.adapter}", flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, device_map="cpu", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True, max_shard_size="4GB")
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tok.save_pretrained(out)
    print(f"[merge] saved -> {out}")

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub, safe_serialization=True)
        tok.push_to_hub(args.push_to_hub)
        print(f"[merge] pushed -> https://huggingface.co/{args.push_to_hub}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

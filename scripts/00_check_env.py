"""环境自检：跑训练前先过一遍，比跑到一半 OOM 强。"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    ok = True
    print(f"python: {sys.version.split()[0]} ({sys.executable})")

    try:
        import torch
        print(f"torch: {torch.__version__} | cuda build: {torch.version.cuda}")
        print(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            i = 0
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            print(f"gpu: {name} | {props.total_memory/2**30:.1f} GiB | sm_{props.major}{props.minor}")
            if props.total_memory / 2**30 < 20:
                print("  !! 显存 < 20GiB，7B QLoRA 可能吃紧")
        else:
            print("  !! 没有 CUDA，训练会退到 CPU（基本不可用）")
            ok = False
    except ImportError:
        print("torch: MISSING")
        ok = False

    for mod in ("transformers", "peft", "accelerate", "bitsandbytes", "yaml"):
        try:
            m = __import__(mod)
            print(f"{mod}: {getattr(m, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001
            print(f"{mod}: MISSING ({type(e).__name__})")
            ok = False

    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes cuda ops: {bnb.functional is not None}")
    except Exception as e:  # noqa: BLE001
        print(f"bitsandbytes 不可用: {e}（QLoRA 需要它；可先用 --no-4bit 跑 bf16 LoRA）")

    free = shutil.disk_usage(Path.cwd()).free / 2**30
    print(f"disk free: {free:.0f} GiB")
    if free < 30:
        print("  !! 磁盘 < 30GiB，7B 权重 + 输出可能不够")

    print(f"HF_HOME: {os.environ.get('HF_HOME', '~/.cache/huggingface')}")
    print("\nRESULT:", "OK" if ok else "有问题，见上面 !!")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())

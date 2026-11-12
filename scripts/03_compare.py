"""配对比较两个预测文件：谁修好了、谁弄坏了，以及这是不是噪声。

单看两个准确率（比如 47.37% vs 51.88%）分不清是真提升还是波动。
配对比较能回答真正的问题：同一批题，B 修好几条、又弄坏几条。
McNemar 精确检验用二项分布算 p 值，不需要 scipy。

用法：
    python scripts/03_compare.py --a pred_v2.jsonl --b pred_sc.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl2sql.sql_utils import execution_accuracy, sql_complexity  # noqa: E402


def load(path: str) -> dict[tuple[str, str], dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[(r.get("db_id", ""), r.get("question", ""))] = r
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """双侧精确 McNemar：H0 下 b、c 各占一半。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def score(rows: dict[tuple[str, str], dict]) -> dict[tuple[str, str], bool]:
    out = {}
    for k, r in rows.items():
        ok, _ = execution_accuracy(r["db_path"], r.get("pred_sql") or "", r["gold_sql"])
        out[k] = ok
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="基线预测（如 v2）")
    ap.add_argument("--b", required=True, help="新方案预测（如 v2 + 自一致性）")
    ap.add_argument("--title", default="配对比较")
    args = ap.parse_args()

    ra, rb = load(args.a), load(args.b)
    keys = sorted(set(ra) & set(rb))
    if not keys:
        print("两个文件没有共同的 (db_id, question)，无法配对", file=sys.stderr)
        return 1

    sa, sb = score({k: ra[k] for k in keys}), score({k: rb[k] for k in keys})
    fixed = [k for k in keys if not sa[k] and sb[k]]
    broken = [k for k in keys if sa[k] and not sb[k]]
    both_ok = sum(1 for k in keys if sa[k] and sb[k])
    both_bad = sum(1 for k in keys if not sa[k] and not sb[k])

    n = len(keys)
    acc_a = (both_ok + len(broken)) / n
    acc_b = (both_ok + len(fixed)) / n
    p = mcnemar_exact(len(fixed), len(broken))

    print(f"# {args.title}\n")
    print(f"配对样本：{n}")
    print(f"| | A 正确 | A 错误 |")
    print(f"|---|---|---|")
    print(f"| **B 正确** | {both_ok} | {len(fixed)} |")
    print(f"| **B 错误** | {len(broken)} | {both_bad} |")
    print()
    print(f"- A 准确率：{acc_a:.2%}")
    print(f"- B 准确率：{acc_b:.2%}（{acc_b - acc_a:+.2%}）")
    print(f"- 修好 **{len(fixed)}** 条，弄坏 **{len(broken)}** 条")
    print(f"- McNemar 精确检验 p = {p:.4f} → "
          f"{'显著（p<0.05）' if p < 0.05 else '**不显著**，差异可能在噪声范围内'}")

    if fixed:
        stats = Counter(sql_complexity(ra[k]["gold_sql"]) for k in fixed)
        print(f"\n修好的样本按难度：{dict(stats)}")
        print("\n修好的例子（前 3 条）：")
        for k in fixed[:3]:
            print(f"- Q: {k[1][:70]}")
            print(f"  A: {ra[k].get('pred_sql', '')[:90]}")
            print(f"  B: {rb[k].get('pred_sql', '')[:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

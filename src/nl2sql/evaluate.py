"""评测：执行准确率（EA）+ 合法率 + 按难度分层 + 失败归因。

产出 report.json（机器读）和 report.md（贴简历/周报用）。
单独跑 baseline 和 LoRA 两次，README 里有对比命令。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .sql_utils import (
    exact_match,
    execution_accuracy,
    extract_sql,
    run_sql,
    sql_complexity,
)

BUCKET_HINT = {
    "invalid_sql": "生成的 SQL 无法执行（语法/表名/列名错误）",
    "wrong_result": "SQL 能跑但结果集不对（WHERE/JOIN/聚合逻辑错）",
    "empty_output": "模型没输出可解析的 SQL",
    "non_readonly": "生成了写操作，被安全闸拦下",
}


def evaluate_file(path: str | Path, timeout_s: float = 10.0) -> dict:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    total = 0
    hits = 0
    valid = 0
    em = 0
    buckets: Counter[str] = Counter()
    by_complexity: dict[str, list[int]] = {}
    by_db: dict[str, list[int]] = {}
    failures = []
    latencies = []

    for r in rows:
        pred = r.get("pred_sql") or extract_sql(r.get("raw") or "")
        gold = r["gold_sql"]
        ok, reason = execution_accuracy(r["db_path"], pred, gold, timeout_s=timeout_s)
        res = run_sql(r["db_path"], pred, timeout_s=timeout_s) if pred else None
        latencies.append(res.elapsed_ms if res else 0.0)

        total += 1
        valid += 1 if (res is not None and res.ok) else 0
        hits += 1 if ok else 0
        em += 1 if exact_match(pred, gold) else 0
        if not ok:
            buckets[reason.split(":")[0]] += 1
            failures.append({
                "db_id": r.get("db_id"),
                "question": r.get("question"),
                "gold_sql": gold,
                "pred_sql": pred,
                "reason": reason,
                "complexity": sql_complexity(gold),
            })

        c = sql_complexity(gold)
        by_complexity.setdefault(c, [0, 0])
        by_complexity[c][0] += 1 if ok else 0
        by_complexity[c][1] += 1

        db = r.get("db_id") or "unknown"
        by_db.setdefault(db, [0, 0])
        by_db[db][0] += 1 if ok else 0
        by_db[db][1] += 1

    report = {
        "n": total,
        "execution_accuracy": round(hits / total, 4) if total else 0.0,
        "valid_sql_rate": round(valid / total, 4) if total else 0.0,
        "exact_match": round(em / total, 4) if total else 0.0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "by_complexity": {
            k: {"acc": round(v[0] / v[1], 4), "n": v[1]} for k, v in sorted(by_complexity.items())
        },
        "by_db": {
            k: {"acc": round(v[0] / v[1], 4), "n": v[1]}
            for k, v in sorted(by_db.items(), key=lambda kv: -kv[1][1])[:20]
        },
        "error_buckets": dict(buckets.most_common()),
    }
    return {"report": report, "failures": failures}


def to_markdown(report: dict, title: str = "NL2SQL 评测报告") -> str:
    lines = [f"# {title}", ""]
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---|")
    lines.append(f"| 样本数 | {report['n']} |")
    lines.append(f"| **执行准确率 (EA)** | **{report['execution_accuracy']:.2%}** |")
    lines.append(f"| SQL 合法率 | {report['valid_sql_rate']:.2%} |")
    lines.append(f"| 精确匹配率 | {report['exact_match']:.2%} |")
    lines.append(f"| 平均执行耗时 | {report['avg_latency_ms']:.1f} ms |")
    lines.append("")
    lines.append("## 按难度")
    lines.append("| 难度 | 准确率 | 样本数 |")
    lines.append("|---|---|---|")
    for k, v in report["by_complexity"].items():
        lines.append(f"| {k} | {v['acc']:.2%} | {v['n']} |")
    if report["error_buckets"]:
        lines.append("")
        lines.append("## 失败归因")
        lines.append("| 类型 | 数量 | 含义 |")
        lines.append("|---|---|---|")
        for k, v in report["error_buckets"].items():
            lines.append(f"| {k} | {v} | {BUCKET_HINT.get(k, '-')} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="infer.py 产出的预测 jsonl")
    ap.add_argument("--out-dir", default="outputs/report")
    ap.add_argument("--title", default="NL2SQL 评测报告")
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args(argv)

    if not Path(args.pred).exists():
        print(f"pred file not found: {args.pred}", file=sys.stderr)
        return 1

    res = evaluate_file(args.pred, timeout_s=args.timeout)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(res["report"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "failures.json").write_text(
        json.dumps(res["failures"], ensure_ascii=False, indent=2), encoding="utf-8")
    md = to_markdown(res["report"], title=args.title)
    (out / "report.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"[eval] 报告已写入 {out}/report.{{json,md}}, 失败样本 {out}/failures.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

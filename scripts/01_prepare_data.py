"""数据准备：合成 demo 数据集，或把 Spider 转成本项目的 jsonl。

Spider 目录约定（官网下载解压后）：
    spider/
      database/<db_id>/<db_id>.sqlite
      train_spider.json      [{"db_id":..., "question":..., "query":...}, ...]
      dev.json

转换时会用 sqlite 真跑一遍 gold SQL，跑不通的直接丢（脏数据源头），
并报告丢弃率——这个数能反映数据集质量。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl2sql.schema import load_schema, to_ddl  # noqa: E402
from nl2sql.sql_utils import run_sql, sql_complexity  # noqa: E402
from nl2sql.synth import build as build_demo  # noqa: E402


def _load_split(src: Path) -> list[dict]:
    """Spider 的 dev.json 不含 query，金标准在同名 _gold.sql 里，按行序一一对应。"""
    rows = json.loads(src.read_text(encoding="utf-8"))
    if rows and not rows[0].get("query"):
        gold = src.parent / f"{src.stem}_gold.sql"
        if gold.exists():
            golds = [ln.strip() for ln in gold.read_text(encoding="utf-8").splitlines()
                     if ln.strip()]
            for r, g in zip(rows, golds):
                r["query"] = g
            print(f"[info] {src.name} 缺少 query，已从 {gold.name} 补 {min(len(rows), len(golds))} 条")
    return rows


def convert_spider(spider_dir: Path, out_dir: Path, include_others: bool = True) -> dict:
    splits = {"train": "train_spider.json", "dev": "dev.json"}
    extra_train = spider_dir / "train_others.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict] = {}

    for split, fname in splits.items():
        src = spider_dir / fname
        if not src.exists():
            src = spider_dir / "train.json" if split == "train" else src
        if not src.exists():
            print(f"[warn] {src} 不存在，跳过 {split}")
            continue

        rows = _load_split(src)
        if split == "train" and include_others and extra_train.exists():
            extra = _load_split(extra_train)
            rows = rows + extra
            print(f"[info] 合并 {extra_train.name}: +{len(extra)} 条（更多库 = 更好的泛化）")
        kept, dropped, seen = [], Counter(), set()
        for r in rows:
            db_id = r["db_id"]
            db_path = spider_dir / "database" / db_id / f"{db_id}.sqlite"
            if not db_path.exists():
                dropped["missing_db"] += 1
                continue
            sql = (r.get("query") or "").strip()
            q = (r.get("question") or "").strip()
            key = (db_id, q)
            if not sql or not q or key in seen:
                dropped["dup_or_empty"] += 1
                continue
            res = run_sql(db_path, sql)
            if not res.ok:
                dropped["gold_sql_broken"] += 1
                continue
            seen.add(key)
            kept.append({"db_id": db_id, "db_path": str(db_path.resolve()),
                         "question": q, "sql": sql})

        dst = out_dir / f"{split}.jsonl"
        with dst.open("w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        comp = Counter(sql_complexity(r["sql"]) for r in kept)
        stats[split] = {"kept": len(kept), "dropped": dict(dropped),
                        "complexity": dict(comp), "file": str(dst)}
        print(f"[{split}] kept={len(kept)} dropped={dict(dropped)} complexity={dict(comp)}")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="demo", choices=["demo", "spider"])
    ap.add_argument("--spider-dir", default="data/spider")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-train", type=int, default=150)
    ap.add_argument("--include-others", action="store_true", default=True,
                    help="训练集合并 train_others.json（默认开）")
    ap.add_argument("--no-include-others", dest="include_others", action="store_false")
    args = ap.parse_args()

    if args.source == "demo":
        out = args.out_dir or "data/demo"
        paths = build_demo(out, n_train=args.n_train)
        print(json.dumps(paths, ensure_ascii=False, indent=2))
        sch = load_schema(paths["db"], db_id="shop")
        print(f"demo db tables={len(sch.tables)} columns={sch.n_columns}")
        print("\n--- DDL 预览 ---")
        print(to_ddl(sch)[:800])
        return 0

    out = args.out_dir or "data/spider_proc"
    stats = convert_spider(Path(args.spider_dir), Path(out),
                           include_others=args.include_others)
    (Path(out) / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    if not stats:
        print("没有转换出任何数据，检查 --spider-dir 结构", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

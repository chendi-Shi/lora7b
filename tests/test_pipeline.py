"""数据管线测试：合成库 → schema → DDL → prompt。

不依赖 transformers/torch，所以在任何机器上都能跑。
"""

from __future__ import annotations

from nl2sql.data import attach_schema, read_jsonl
from nl2sql.prompt import build_messages, build_messages_with_answer
from nl2sql.schema import load_schema, to_ddl, to_markdown
from nl2sql.sql_utils import run_sql
from nl2sql.synth import build


def test_synth_dataset_and_schema(tmp_path):
    paths = build(tmp_path, n_train=20, n_dev=5, n_test=5)
    train = read_jsonl(paths["train"])
    assert len(train) == 20
    assert all(x.sql.lower().startswith(("select", "with")) for x in train)

    sch = load_schema(paths["db"], db_id="shop")
    names = {t.name for t in sch.tables}
    assert names == {"customers", "products", "orders"}
    assert sch.n_columns >= 13

    ddl = to_ddl(sch)
    assert "CREATE TABLE" in ddl and "FOREIGN KEY" in ddl
    assert '"customers"' in ddl

    md = to_markdown(sch)
    assert "| column | type | note |" in md


def test_gold_sql_all_executable(tmp_path):
    paths = build(tmp_path, n_train=30, n_dev=10, n_test=10)
    for split in ("train", "dev", "test"):
        for ex in read_jsonl(paths[split]):
            res = run_sql(ex.db_path, ex.sql)
            assert res.ok, f"{ex.question} -> {res.error}"


def test_attach_schema_caches_ddl(tmp_path):
    paths = build(tmp_path, n_train=4, n_dev=2, n_test=2)
    items = attach_schema(read_jsonl(paths["train"]))
    assert all("CREATE TABLE" in it["ddl"] for it in items)
    # 同一个库，DDL 文本应当完全一致（lru_cache 命中）
    assert len({it["ddl"] for it in items}) == 1


def test_prompt_contains_schema_and_question():
    ddl = 'CREATE TABLE "t" ("a" INTEGER);'
    msgs = build_messages(ddl, "多少行？")
    assert msgs[0]["role"] == "system" and ddl in msgs[0]["content"]
    assert msgs[1]["role"] == "user" and "多少行？" in msgs[1]["content"]
    assert "ONLY the SQL" in msgs[0]["content"]


def test_prompt_with_answer_ends_with_semicolon():
    msgs = build_messages_with_answer("CREATE TABLE t (a INT);", "q", "SELECT a FROM t")
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-1]["content"].endswith(";")
    assert not msgs[-1]["content"].endswith(";;")


def test_dataset_paths_are_portable(tmp_path):
    """jsonl 里的 db_path 必须是相对路径，否则 rsync 到服务器就失效。"""
    import json
    from pathlib import Path as _P

    paths = build(tmp_path, n_train=4, n_dev=2, n_test=2)
    raw = json.loads(_P(paths["train"]).read_text(encoding="utf-8").splitlines()[0])
    assert "/" not in raw["db_path"] and "\\" not in raw["db_path"]
    ex = read_jsonl(paths["train"])[0]
    assert _P(ex.db_path).exists() and _P(ex.db_path).is_absolute()


def test_generated_ddl_is_valid_sql(tmp_path):
    """DDL 要能真的执行——带注释的 DDL 很容易写成非法语法。"""
    import sqlite3

    paths = build(tmp_path, n_train=2, n_dev=1, n_test=1)
    sch = load_schema(paths["db"], db_id="shop")
    con = sqlite3.connect(":memory:")
    con.executescript(to_ddl(sch, include_samples=True))
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    assert tables == {"customers", "products", "orders"}
    con.close()


def test_infer_prompts_accepts_jsonl_examples(tmp_path):
    """回归测试：read_jsonl 返回的是 Example 对象，infer 里必须用属性访问。

    这个 bug 在 GPU 机器上要等模型加载完才暴露，代价极高，所以在纯 CPU 环境锁死。
    """
    from nl2sql.infer import _prompts

    paths = build(tmp_path, n_train=2, n_dev=1, n_test=1)
    items = read_jsonl(paths["test"])
    msgs = _prompts(items, include_samples=True)
    assert len(msgs) == 1
    assert msgs[0][0]["role"] == "system"
    assert "CREATE TABLE" in msgs[0][0]["content"]


def test_select_items_filters_by_complexity(tmp_path):
    from nl2sql.infer import _select_items
    from nl2sql.sql_utils import sql_complexity

    paths = build(tmp_path, n_train=30, n_dev=10, n_test=20)
    items = read_jsonl(paths["test"])
    hard = _select_items(items, "hard", 0)
    assert all(sql_complexity(i.sql) == "hard" for i in hard)
    assert len(_select_items(items, None, 5)) == 5
    assert len(_select_items(items, None, 0)) == len(items)

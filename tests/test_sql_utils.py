"""SQL 工具单测——评测正确性的最后一道防线。

这些都跑在内存/临时 sqlite 上，不需要 GPU、不需要下载模型。
"""

from __future__ import annotations

import sqlite3

import pytest

from nl2sql.sql_utils import (
    canonical_rows,
    exact_match,
    execution_accuracy,
    extract_sql,
    is_readonly,
    order_sensitive,
    results_equal,
    run_sql,
    sql_complexity,
)


@pytest.fixture()
def db(tmp_path):
    p = tmp_path / "t.sqlite"
    con = sqlite3.connect(p)
    con.executescript(
        """
        CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, v REAL);
        INSERT INTO t VALUES (1, 'Alice', 1.0), (2, 'bob', 2.50), (3, NULL, 3.0);
        """
    )
    con.commit()
    con.close()
    return str(p)


# ---------------------------------------------------------------- extract


@pytest.mark.parametrize("raw,expected", [
    ("SELECT 1;", "SELECT 1"),
    ("```sql\nSELECT 1;\n```", "SELECT 1"),
    ("Here is the query:\nSELECT name FROM t;", "SELECT name FROM t"),
    ("SQL: SELECT a FROM t", "SELECT a FROM t"),
    ("SELECT 'a;b' FROM t", "SELECT 'a;b' FROM t"),
    ("no sql here", ""),
    ("", ""),
])
def test_extract_sql(raw, expected):
    assert extract_sql(raw) == expected


def test_extract_takes_first_statement():
    assert extract_sql("SELECT 1; SELECT 2;") == "SELECT 1"


def test_extract_with_cte():
    sql = "WITH x AS (SELECT 1) SELECT * FROM x"
    assert extract_sql(sql) == sql


# ---------------------------------------------------------------- safety


@pytest.mark.parametrize("sql", [
    "SELECT 1", "WITH a AS (SELECT 1) SELECT * FROM a",
    "select count(*) from t",
])
def test_readonly_allows_select(sql):
    assert is_readonly(sql)


@pytest.mark.parametrize("sql", [
    "DELETE FROM t", "DROP TABLE t", "UPDATE t SET v=1",
    "INSERT INTO t VALUES (9,'x',0)", "PRAGMA table_info(t)",
    "/*x*/ delete from t",
])
def test_readonly_blocks_writes(sql):
    assert not is_readonly(sql)


# ---------------------------------------------------------------- exec


def test_run_sql_ok(db):
    r = run_sql(db, "SELECT name FROM t WHERE id=1")
    assert r.ok and r.rows == [("Alice",)]


def test_run_sql_rejects_write(db):
    r = run_sql(db, "DELETE FROM t")
    assert not r.ok and "readonly" in r.error


def test_run_sql_bad_sql(db):
    r = run_sql(db, "SELECT nope FROM t")
    assert not r.ok


def test_run_sql_missing_db(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_sql(tmp_path / "nope.sqlite", "SELECT 1")


# ---------------------------------------------------------------- compare


def test_results_equal_ignores_row_order():
    assert results_equal([(1,), (2,)], [(2,), (1,)])


def test_results_equal_respects_order_when_asked():
    assert not results_equal([(1,), (2,)], [(2,), (1,)], order_sensitive=True)


def test_numeric_type_coercion():
    # sqlite 里 1 和 1.0 可能来自不同运算路径，应当算相同
    assert results_equal([(1,)], [(1.0,)])


def test_null_handling():
    assert results_equal([(None,)], [(None,)])
    assert not results_equal([(None,)], [("",)])


def test_case_and_space_normalization():
    assert results_equal([("Alice ",)], [("alice",)])


def test_order_sensitive_detection():
    assert order_sensitive("SELECT a FROM t ORDER BY a")
    assert not order_sensitive("SELECT a FROM t")


# ---------------------------------------------------------------- EA


def test_ea_correct(db):
    ok, reason = execution_accuracy(db, "SELECT name FROM t WHERE id = 1",
                                    "SELECT name FROM t WHERE id = 1")
    assert ok and reason == "ok"


def test_ea_equivalent_sql_counts_as_correct(db):
    # 写法不同但结果相同 -> 执行准确率算对（这正是 EA 比 EM 合理的地方）
    ok, _ = execution_accuracy(db, "SELECT name FROM t WHERE 1 = 1 AND id = 1",
                               "SELECT name FROM t WHERE id = 1")
    assert ok


def test_ea_wrong_result(db):
    ok, reason = execution_accuracy(db, "SELECT name FROM t WHERE id = 2",
                                    "SELECT name FROM t WHERE id = 1")
    assert not ok and reason == "wrong_result"


def test_ea_invalid_sql(db):
    ok, reason = execution_accuracy(db, "SELECT bad_col FROM t", "SELECT name FROM t")
    assert not ok and reason == "invalid_sql"


def test_ea_empty_output(db):
    ok, reason = execution_accuracy(db, "", "SELECT name FROM t")
    assert not ok and reason == "empty_output"


def test_ea_order_by_sensitivity(db):
    gold = "SELECT id FROM t ORDER BY id"
    rev = "SELECT id FROM t ORDER BY id DESC"
    ok, reason = execution_accuracy(db, rev, gold, )
    assert not ok and reason == "wrong_result"


# ---------------------------------------------------------------- misc


@pytest.mark.parametrize("sql,level", [
    ("SELECT a FROM t", "easy"),
    ("SELECT a FROM t JOIN u ON t.id=u.id", "medium"),
    ("SELECT a FROM t GROUP BY b", "medium"),
    ("SELECT a FROM t JOIN u ON t.id=u.id JOIN v ON v.id=u.id", "hard"),
    ("SELECT a FROM t UNION SELECT a FROM u", "hard"),
])
def test_complexity(sql, level):
    assert sql_complexity(sql) == level


def test_exact_match_ignores_formatting():
    assert exact_match("SELECT a  FROM t;", "select a from t")


def test_canonical_rows_stable():
    a = canonical_rows([(1, "x"), (2, "y")], order_sensitive=False)
    b = canonical_rows([(2, "y"), (1, "x")], order_sensitive=False)
    assert a == b

"""SQL 工具：抽取、规范化、SQLite 只读执行、结果集等价比较。

这个模块是评测的地基：执行准确率（Execution Accuracy）的错误几乎都出在
"结果集怎么比" 上（列序、行序、浮点、NULL、大小写、空格）。所以这里把
比较规则写死、写全，并用 tests/ 覆盖。
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------- 抽取


_COMMENT_RE = re.compile(r"(--[^\n]*\n)|(/\*.*?\*/)", re.S)
_FENCE_RE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.S | re.I)


def strip_comments(sql: str) -> str:
    return _COMMENT_RE.sub(" ", sql).strip()


def extract_sql(text: str) -> str:
    """从模型输出里抠出第一条完整 SQL 语句。

    处理：```sql 代码块```、前后解释性文字、多语句、未闭合的反引号。
    """
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1)

    text = strip_comments(text).strip()
    # 去掉 Qwen 可能带出来的 "SQL:" / "答案：" 前缀
    text = re.sub(r"^\s*(SQL|sql|答案|查询)\s*[:：]\s*", "", text)

    # 扫描到第一个不在字符串/标识符里的分号
    out: list[str] = []
    quote: str | None = None
    for ch in text:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"', "`"):
            quote = ch
            out.append(ch)
        elif ch == ";":
            break
        else:
            out.append(ch)
    sql = "".join(out).strip()

    # 常见退化：模型只吐了表名/一句自然语言
    if not re.match(r"(?is)^\s*(with|select)\b", sql):
        idx = re.search(r"(?is)\b(with|select)\b", sql)
        if idx:
            sql = sql[idx.start():].strip()
        else:
            return ""
    return " ".join(sql.split())


_READONLY_RE = re.compile(
    r"(?is)\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex)\b"
)


def is_readonly(sql: str) -> bool:
    """拦住写操作 / PRAGMA。评测库是只读打开的，这里再加一道闸。"""
    return not _READONLY_RE.search(strip_comments(sql))


def sql_complexity(sql: str) -> str:
    """粗分难度，用来做分层指标（写进报告，比单一总分有说服力）。"""
    s = strip_comments(sql).lower()
    joins = len(re.findall(r"\bjoin\b", s))
    if joins >= 2 or re.search(r"\bunion\b|\bintersect\b|\bexcept\b", s):
        return "hard"
    if joins == 1 or re.search(r"\bgroup\s+by\b|\bhaving\b", s):
        return "medium"
    return "easy"


# ---------------------------------------------------------------- 执行


@dataclass
class ExecResult:
    ok: bool
    rows: list[tuple[Any, ...]]
    error: str = ""
    elapsed_ms: float = 0.0


class QueryTimeout(Exception):
    pass


def _connect_ro(db_path: str | Path) -> sqlite3.Connection:
    p = Path(db_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"database not found: {p}")
    # mode=ro：物理上杜绝写入，连临时表都不给建
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=5.0)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    return con


def run_sql(
    db_path: str | Path,
    sql: str,
    timeout_s: float = 10.0,
    max_rows: int = 100_000,
) -> ExecResult:
    """只读执行，带超时（sqlite 的 progress_handler 机制）。"""
    t0 = time.perf_counter()
    if not sql or not sql.strip():
        return ExecResult(False, [], "empty sql")
    if not is_readonly(sql):
        return ExecResult(False, [], "non-readonly statement rejected")
    # 库文件缺失是数据问题，必须炸出来而不是算成"SQL 写错"
    if not Path(db_path).exists():
        raise FileNotFoundError(f"database not found: {db_path}")

    con = None
    try:
        con = _connect_ro(db_path)
        deadline = time.monotonic() + timeout_s

        def _progress() -> int:
            return 1 if time.monotonic() > deadline else 0

        con.set_progress_handler(_progress, 20_000)
        cur = con.execute(sql)
        rows = cur.fetchmany(max_rows)
        return ExecResult(True, [tuple(r) for r in rows],
                          elapsed_ms=(time.perf_counter() - t0) * 1000)
    except Exception as e:  # noqa: BLE001 - 任何报错都算 invalid sql
        msg = str(e)
        if time.monotonic() > 0 and "interrupted" in msg.lower():
            msg = f"timeout after {timeout_s}s"
        return ExecResult(False, [], msg, (time.perf_counter() - t0) * 1000)
    finally:
        if con is not None:
            con.close()


# ---------------------------------------------------------------- 结果集比较

_FLOAT_RE = re.compile(r"^-?\d+\.\d+$")


def _norm_cell(v: Any) -> Any:
    if v is None:
        return ("null",)
    if isinstance(v, bool):
        return ("num", float(v))
    if isinstance(v, (int, float)):
        return ("num", round(float(v), 6))
    s = str(v).strip()
    if _FLOAT_RE.match(s):
        try:
            return ("num", round(float(s), 6))
        except ValueError:
            pass
    return ("str", s.lower())


def canonical_rows(rows: Iterable[Sequence[Any]], order_sensitive: bool) -> tuple:
    """把结果集压成可比较的元组。默认忽略行序（SQL 无 ORDER BY 时行序无意义）。"""
    norm = [tuple(_norm_cell(c) for c in r) for r in rows]
    if not order_sensitive:
        norm.sort(key=repr)
    return tuple(norm)


def results_equal(
    rows_a: Iterable[Sequence[Any]],
    rows_b: Iterable[Sequence[Any]],
    order_sensitive: bool = False,
) -> bool:
    return canonical_rows(rows_a, order_sensitive) == canonical_rows(rows_b, order_sensitive)


def result_fingerprint(rows: Iterable[Sequence[Any]]) -> str:
    """给自一致性投票用的指纹：忽略行序。"""
    payload = repr(canonical_rows(rows, order_sensitive=False))
    return hashlib.md5(payload.encode()).hexdigest()


def order_sensitive(sql: str) -> bool:
    """gold 带 ORDER BY 且没有聚合时，行序是有意义的。"""
    s = strip_comments(sql).lower()
    return bool(re.search(r"\border\s+by\b", s))


def normalize_sql_text(sql: str) -> str:
    """仅用于 exact-match 展示，不参与执行准确率。"""
    s = strip_comments(sql).lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([(),])\s*", r"\1", s)
    return s.strip().rstrip(";")


def exact_match(pred: str, gold: str) -> bool:
    return normalize_sql_text(pred) == normalize_sql_text(gold)


def execution_accuracy(
    db_path: str | Path,
    pred_sql: str,
    gold_sql: str,
    timeout_s: float = 10.0,
) -> tuple[bool, str]:
    """返回 (是否命中, 错误类型)。错误类型用于后面做失败归因。"""
    if not pred_sql:
        return False, "empty_output"
    if not is_readonly(pred_sql):
        return False, "non_readonly"

    gold = run_sql(db_path, gold_sql, timeout_s)
    if not gold.ok:
        return False, f"gold_sql_broken: {gold.error}"

    pred = run_sql(db_path, pred_sql, timeout_s)
    if not pred.ok:
        return False, "invalid_sql"

    sensitive = order_sensitive(gold_sql)
    if not results_equal(pred.rows, gold.rows, order_sensitive=sensitive):
        return False, "wrong_result"
    return True, "ok"

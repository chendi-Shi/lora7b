"""Schema 序列化：把 sqlite 库变成模型能读的 DDL / 值提示。

NL2SQL 的准确率有很大一块来自"模型看不看得清 schema"。这里做三件事：
1. DDL 化（CREATE TABLE + 类型 + 主键 + 外键）
2. 抽样值（text 列的 distinct 前若干个）——解决 "status='shipped' 还是 'Shipped'" 这类问题
3. 缓存（一个 db 只读一次元数据）
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


@dataclass
class Column:
    name: str
    type: str
    pk: bool = False


@dataclass
class ForeignKey:
    from_col: str
    to_table: str
    to_col: str


@dataclass
class Table:
    name: str
    columns: list[Column] = field(default_factory=list)
    fks: list[ForeignKey] = field(default_factory=list)
    samples: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Schema:
    db_id: str
    tables: list[Table] = field(default_factory=list)

    @property
    def n_columns(self) -> int:
        return sum(len(t.columns) for t in self.tables)


def _qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def load_schema(db_path: str | Path, db_id: str | None = None,
                sample_rows: int = 4, max_sample_len: int = 24) -> Schema:
    p = Path(db_path)
    db_id = db_id or p.stem
    con = sqlite3.connect(f"file:{p.resolve()}?mode=ro", uri=True)
    try:
        tables = [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        schema = Schema(db_id=db_id)
        for t in tables:
            tbl = Table(name=t)
            for _cid, name, ctype, notnull, default, pk in con.execute(
                f"PRAGMA table_info({_qident(t)})"
            ):
                tbl.columns.append(Column(name=name, type=(ctype or "TEXT").upper(),
                                          pk=bool(pk)))
            for fk in con.execute(f"PRAGMA foreign_key_list({_qident(t)})"):
                # (id, seq, table, from, to, on_update, on_delete, match)
                tbl.fks.append(ForeignKey(from_col=fk[3], to_table=fk[2],
                                          to_col=fk[4] or "id"))
            if sample_rows > 0:
                tbl.samples = _sample_values(con, t, tbl, sample_rows, max_sample_len)
            schema.tables.append(tbl)
        return schema
    finally:
        con.close()


def _sample_values(con: sqlite3.Connection, table: str, tbl: Table,
                   k: int, max_len: int) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for col in tbl.columns:
        # 只给文本/低基数列抽样：数字列抽样对模型没用，还占 token
        if col.type not in ("TEXT", "VARCHAR", "CHAR", "NVARCHAR", "STRING"):
            continue
        try:
            rows = con.execute(
                f"SELECT DISTINCT {_qident(col.name)} FROM {_qident(table)} "
                f"WHERE {_qident(col.name)} IS NOT NULL LIMIT {k}"
            ).fetchall()
        except sqlite3.Error:
            continue
        vals = [str(r[0]).strip()[:max_len] for r in rows if str(r[0]).strip()]
        if 0 < len(vals) <= k:
            out[col.name] = vals
    return out


def to_ddl(schema: Schema, include_samples: bool = True) -> str:
    """生成紧凑 DDL。外键用注释形式，抽样值挂在列后注释里。"""
    lines: list[str] = []
    for t in schema.tables:
        cols: list[str] = []
        for c in t.columns:
            piece = f"  {_qident(c.name)} {c.type}"
            if c.pk:
                piece += " PRIMARY KEY"
            if include_samples and c.name in t.samples:
                vals = ", ".join(repr(v) for v in t.samples[c.name])
                # 用块注释：行注释会把后面的逗号也吃掉，生成的 DDL 就不是合法 SQL 了
                piece += f" /* e.g. {vals} */"
            cols.append(piece)
        body = ",\n".join(cols)
        fk_lines = [
            f"  FOREIGN KEY ({_qident(fk.from_col)}) REFERENCES "
            f"{_qident(fk.to_table)}({_qident(fk.to_col)})"
            for fk in t.fks
        ]
        all_lines = [body] + fk_lines if fk_lines else [body]
        lines.append(f"CREATE TABLE {_qident(t.name)} (\n"
                     + ",\n".join(all_lines) + "\n);")
    return "\n\n".join(lines)


def to_markdown(schema: Schema) -> str:
    out: list[str] = []
    for t in schema.tables:
        out.append(f"### {t.name}")
        out.append("| column | type | note |")
        out.append("|---|---|---|")
        for c in t.columns:
            note = "PK" if c.pk else ""
            if c.name in t.samples:
                note = (note + " " if note else "") + "e.g. " + ", ".join(t.samples[c.name])
            out.append(f"| {c.name} | {c.type} | {note} |")
        out.append("")
    return "\n".join(out)


@lru_cache(maxsize=256)
def cached_schema(db_path: str, sample_rows: int = 4) -> Schema:
    return load_schema(db_path, sample_rows=sample_rows)

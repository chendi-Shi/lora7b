"""Prompt 构造。

原则：
- schema 放 system 里，问题放 user 里。Qwen 的 chat template 会把它包成
  <|im_start|>system ... <|im_end|> 的形式，训练/推理共用同一套 messages。
- 输出约束写死：只输出一条 SQL、不要解释、不要 markdown。减少后缀污染，
  这样 extract_sql 才稳。
"""

from __future__ import annotations

SYSTEM_TEMPLATE = (
    "You are a senior SQLite analyst. Given a database schema, write exactly one "
    "SQLite query that answers the question.\n"
    "Rules:\n"
    "1. Output ONLY the SQL statement. No explanation, no markdown fences.\n"
    "2. Use only tables and columns present in the schema.\n"
    "3. Quote identifiers with double quotes when they contain spaces or mixed case.\n"
    "4. Prefer explicit JOIN ... ON over implicit joins.\n"
    "5. If the question asks for a ranking or top-N, add ORDER BY and LIMIT.\n"
    "\n"
    "### Database schema\n{schema}\n"
)

USER_TEMPLATE = "### Question\n{question}"


def build_messages(schema_ddl: str, question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(schema=schema_ddl.strip())},
        {"role": "user", "content": USER_TEMPLATE.format(question=question.strip())},
    ]


def build_messages_with_answer(schema_ddl: str, question: str, sql: str) -> list[dict[str, str]]:
    msgs = build_messages(schema_ddl, question)
    msgs.append({"role": "assistant", "content": sql.strip().rstrip(";") + ";"})
    return msgs

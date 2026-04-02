from __future__ import annotations

import logging
import re

from openai import OpenAI

from app.services.query_normalization import (
    CANONICAL_COURSES,
    CANONICAL_STATES,
    fix_sql_state_and_course,
    normalize_user_question,
)

_STATE_LIST = ", ".join(sorted(CANONICAL_STATES))
_COURSE_LIST = ", ".join(sorted(CANONICAL_COURSES))

SQL_SYSTEM_PROMPT = f"""
You are an expert PostgreSQL query generator for a NEET counselling assistant.

Hard rules:
1) Output SQL only, no markdown and no explanation.
2) Only generate a single SELECT query on table neet_cutoffs.
3) Allowed columns to use: state, college_name, category, air_rank, score, fee, course, round.
4) Never use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, GRANT, REVOKE.
5) Always include LIMIT 50.
6) Prefer filtering by state, rank, and category when present in user question.
7) Use case-insensitive filtering with ILIKE where useful.
8) For rank conditions, use air_rank unless user explicitly says state rank.
9) Sort by better rank first (air_rank ASC) where ranking is relevant.
10) State names in the database are EXACTLY one of: {_STATE_LIST}
    Use these exact spellings in state filters (Jammu & Kashmir is stored as J&K; Tamil Nadu as TAMILNADU).
11) Course names in the database are EXACTLY one of: {_COURSE_LIST}
    Use these exact spellings in course filters (e.g. BDS not bds when matching stored values; ILIKE is fine).
"""
logger = logging.getLogger("neet_assistant.sql")


def _clean_sql(raw_sql: str) -> str:
    sql = raw_sql.strip()
    sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"^```\s*", "", sql)
    sql = re.sub(r"\s*```$", "", sql)
    return sql.strip().rstrip(";")


def _is_safe_select(sql: str) -> bool:
    lowered = sql.lower().strip()
    if not lowered.startswith("select"):
        return False

    blocked = [
        " insert ",
        " update ",
        " delete ",
        " drop ",
        " alter ",
        " truncate ",
        " create ",
        " grant ",
        " revoke ",
        " execute ",
        " call ",
        ";",
    ]
    padded = f" {lowered} "
    return all(token not in padded for token in blocked)


def _ensure_limit(sql: str, limit: int = 50) -> str:
    if re.search(r"\blimit\s+\d+\b", sql, flags=re.IGNORECASE):
        return re.sub(r"\blimit\s+\d+\b", f"LIMIT {limit}", sql, flags=re.IGNORECASE)
    return f"{sql} LIMIT {limit}"


def generate_sql(client: OpenAI, user_question: str) -> str:
    normalized = normalize_user_question(user_question)
    logger.info("Generating SQL for question length=%d", len(user_question))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SQL_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": normalized.strip()},
        ],
        temperature=0,
    )

    raw_sql = response.choices[0].message.content or ""
    sql = _ensure_limit(_clean_sql(raw_sql), limit=50)
    sql = fix_sql_state_and_course(sql)
    logger.info("SQL after state/course normalization: %s", sql)

    if not _is_safe_select(sql):
        logger.warning("SQL blocked by safety checks: %s", sql)
        raise ValueError("Generated SQL failed safety checks.")
    logger.info("SQL passed safety checks")
    return sql


def generate_counsellor_answer(client: OpenAI, user_question: str, data: list[dict]) -> str:
    logger.info("Generating counsellor answer for rows=%d", len(data))
    answer_prompt = f"""
You are a student-friendly NEET counsellor.

User question:
{user_question}

Query results (JSON):
{data}

Your job:
- Give a concise and helpful counselling response.
- Highlight best colleges from the result.
- Mention rank trend or cutoff range if visible.
- Mention fee insights when available.
- If no data is found, explain that clearly and suggest better search filters.
- Keep tone simple, practical, and supportive.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful NEET counselling assistant."},
            {"role": "user", "content": answer_prompt.strip()},
        ],
        temperature=0.3,
    )
    return (response.choices[0].message.content or "").strip()

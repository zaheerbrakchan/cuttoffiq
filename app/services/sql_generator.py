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

Semantics of air_rank (critical):
- Each row's air_rank is the All India Rank of the candidate who was allotted that seat (lower number = more competitive seat).
- When the user states THEIR OWN rank (e.g. "my rank is 4356", "I have AIR 8000", "which colleges can I get"):
  you want seats where that allotment is at least as "easy" as their rank — use **air_rank >= <their_rank>**.
  Do NOT use air_rank <= <their_rank> for that case; that returns more competitive seats they usually cannot get.
- When the user asks for "top" / "best" / "under rank 5000" in the sense of **within the first N ranks** (competitive list),
  use **air_rank <= N** and ORDER BY air_rank ASC.
- If both state and rank apply, combine filters with AND.

Hard rules:
1) Output SQL only, no markdown and no explanation.
2) Only generate a single SELECT query on table neet_cutoffs.
3) Allowed columns to use: state, college_name, category, air_rank, score, fee, course, round.
4) Never use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, GRANT, REVOKE.
5) Always include LIMIT 50.
6) Prefer filtering by state, rank, and category when present in user question.
7) Use case-insensitive filtering with ILIKE where useful.
8) For rank conditions, use air_rank unless user explicitly says state rank.
9) For "my rank / which colleges can I get", ORDER BY air_rank ASC (show nearer cutoffs first).
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


# User states their own AIR (must match "my rank is 4356", "my AIR 8000", etc.)
_USER_OWN_RANK_PATTERN = re.compile(
    r"(?:"
    r"my\s+rank\s+is|my\s+air\s+is|my\s+neet\s+rank\s+is|"
    r"my\s+rank|my\s+air|"
    r"i\s+(?:have|got)\s+(?:an?\s+)?(?:air|rank)\s+(?:of\s+)?|"
    r"rank\s+is|air\s*(?:rank\s+)?is"
    r")\s*[:\s]*(\d{1,7})",
    re.IGNORECASE,
)
_ELIGIBILITY_HINT = re.compile(
    r"(which\s+colleges|can\s+i\s+get|colleges\s+i\s+can|eligible|options?\s+(?:for|with))",
    re.IGNORECASE,
)


def _extract_user_air(question: str) -> int | None:
    m = _USER_OWN_RANK_PATTERN.search(question)
    if not m:
        return None
    return int(m.group(1))


def _is_eligibility_question(question: str) -> bool:
    q = question.lower()
    if _extract_user_air(question) is None:
        return False
    if _ELIGIBILITY_HINT.search(q):
        return True
    if "my rank" in q or "my air" in q:
        return True
    return False


def fix_air_rank_eligibility_sql(question: str, sql: str) -> str:
    """
    If the user asked about their own rank and eligibility, the model often wrongly uses
    air_rank <= R. Correct to air_rank >= R for allotment-based cutoff data.
    """
    if not _is_eligibility_question(question):
        return sql
    user_air = _extract_user_air(question)
    if user_air is None:
        return sql
    # Flip air_rank <= user_air  ->  air_rank >= user_air (same number)
    pattern = re.compile(
        rf"\bair_rank\s*<=\s*{user_air}\b",
        re.IGNORECASE,
    )
    if pattern.search(sql):
        fixed = pattern.sub(f"air_rank >= {user_air}", sql)
        logger.info(
            "Adjusted eligibility filter: air_rank <= %s -> air_rank >= %s",
            user_air,
            user_air,
        )
        return fixed
    return sql


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
    sql = fix_air_rank_eligibility_sql(normalized, sql)
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
- air_rank in the data is the allotted candidate's AIR (lower = more competitive). If the user gave their own rank,
  only describe colleges from the result as realistic options; do NOT say they can get a seat if that row's air_rank
  is a much lower (better) number than their rank unless the data clearly supports it.
- Highlight colleges from the result that match the user's question.
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

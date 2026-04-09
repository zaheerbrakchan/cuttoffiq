from __future__ import annotations

import logging
import re

from openai import OpenAI

from app.services.query_normalization import (
    CANONICAL_COURSES,
    CANONICAL_STATES,
    fix_sql_state_and_course,
    normalize_user_question,
    resolve_state,
)

_STATE_LIST = ", ".join(sorted(CANONICAL_STATES))
_COURSE_LIST = ", ".join(sorted(CANONICAL_COURSES))

SQL_SYSTEM_PROMPT = f"""
You are an expert PostgreSQL query generator for a NEET counselling assistant.

score vs air_rank (NEVER mix these — critical):
- Column **score** = NEET **marks** (UG score, typically up to ~720). Phrases: "I scored 540", "my score is 580", "getting 475 marks", "NEET score", "marks".
- Column **air_rank** = **All India Rank** only (positive integer, can be 1 to lakhs). Phrases: "my rank is 4356", "AIR 12000", "all India rank", "under rank 5000".
- If the user gave a **mark/score** (e.g. 540), you MUST filter using **score**, never put that number in **air_rank** conditions.
- If the user gave a **rank/AIR**, use **air_rank** only for that number.
- For lists like "score above 580" or "minimum 600 marks": use **score >= 580** (or BETWEEN as needed).
- For "my score is S, which colleges" (eligibility by marks): use **score <= S** to compare against allotted scores in rows (same idea as rank eligibility but on the score column).

Semantics of air_rank (when user meant RANK, not marks):
- Each row's air_rank is the All India Rank of the candidate who was allotted that seat (lower number = more competitive seat).
- When the user states THEIR OWN **rank** (e.g. "my rank is 4356", "I have AIR 8000", "which colleges can I get"):
  use **air_rank >= <their_rank>** for eligibility-style lists (not air_rank <= for that case).
- When the user asks for "top" / "best" / "under rank 5000" in the sense of **within the first N ranks** (competitive list),
  use **air_rank <= N** and ORDER BY air_rank ASC.
- If both state and rank apply, combine filters with AND.

Hard rules:
1) Output SQL only, no markdown and no explanation.
2) Only generate a single SELECT query on table neet_ug_2025_cutoffs.
3) Allowed columns to use: state, air_rank, state_rank, college_code, college_name, institution_name, college_type, course, category, sub_category, seat_type, quota, domicile, eligibility, score, round.
   Use college_type when the user asked for government / private / deemed / AIIMS etc.
4) Never use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, GRANT, REVOKE.
5) Always include LIMIT 50.
6) Prefer filtering by state, category, and either **score** (marks) or **air_rank** (rank) — never swap them.
7) Use case-insensitive filtering with ILIKE where useful.
8) For **rank** conditions only, use air_rank unless user explicitly says state rank.
9) For "my rank / which colleges can I get" (rank-based), ORDER BY air_rank ASC when using air_rank.
10) State names in the database are EXACTLY one of: {_STATE_LIST}
    Use these exact spellings in state filters (Jammu & Kashmir is stored as J&K; Tamil Nadu as TAMILNADU).
11) Course names in the database are EXACTLY one of: {_COURSE_LIST}
    Use these exact spellings in course filters (e.g. BDS not bds when matching stored values; ILIKE is fine).
12) Domicile column semantics:
    - DOMICILE: seats for home-state candidates
    - NON-DOMICILE: seats for outside-state candidates
    - OPEN: seats open to both
    If state-level query is for student's own state, prefer domicile IN ('DOMICILE','OPEN').
    If query is for another state, prefer domicile IN ('NON-DOMICILE','OPEN').
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


def _strip_limit(sql: str) -> str:
    return re.sub(r"\blimit\s+\d+\b", "", sql, flags=re.IGNORECASE).strip()


def _extract_primary_metric_target(context: str) -> tuple[str, int] | None:
    """
    Returns ("air_rank", value) or ("score", value) from user context when available.
    Preference: explicit rank/AIR first, then explicit score/marks.
    """
    user_air = _extract_user_air(context)
    if user_air is not None:
        return ("air_rank", user_air)
    score_nums = sorted(_numbers_tied_to_neet_score(context))
    if score_nums:
        return ("score", score_nums[0])
    return None


def apply_distinct_colleges(sql: str, *, context: str, limit: int = 50) -> str:
    """
    Return one best row per college_name to avoid duplicate colleges across rounds/seat rows.
    Keeps strongest (lowest) AIR rank row per college by default.
    """
    if not re.search(r"\bcollege_name\b", sql, flags=re.IGNORECASE):
        return _ensure_limit(sql, limit=limit)

    base = _strip_limit(sql)
    target = _extract_primary_metric_target(context)
    if target and target[0] == "air_rank":
        n = target[1]
        rank_expr = (
            f"ABS(COALESCE(air_rank, 9999999) - {n}) ASC, "
            "air_rank ASC NULLS LAST, score DESC NULLS LAST"
        )
        final_order = (
            f"ABS(COALESCE(air_rank, 9999999) - {n}) ASC, "
            "air_rank ASC NULLS LAST"
        )
    elif target and target[0] == "score":
        n = target[1]
        rank_expr = (
            f"ABS(COALESCE(score, -1) - {n}) ASC, "
            "score DESC NULLS LAST, air_rank ASC NULLS LAST"
        )
        final_order = (
            f"ABS(COALESCE(score, -1) - {n}) ASC, "
            "score DESC NULLS LAST"
        )
    else:
        rank_expr = "air_rank ASC NULLS LAST, score DESC NULLS LAST"
        final_order = "air_rank ASC NULLS LAST"

    wrapped = f"""
SELECT * FROM (
    SELECT
        q.*,
        ROW_NUMBER() OVER (
            PARTITION BY college_name
            ORDER BY {rank_expr}
        ) AS _rownum_college
    FROM (
        {base}
    ) q
) dedup
WHERE _rownum_college = 1
ORDER BY {final_order}
LIMIT {limit}
""".strip()
    return wrapped


def _append_where_condition(sql: str, condition: str) -> str:
    match = re.search(r"\b(order\s+by|limit)\b", sql, flags=re.IGNORECASE)
    if match:
        cut = match.start()
        head = sql[:cut].rstrip()
        tail = sql[cut:].lstrip()
    else:
        head = sql.rstrip()
        tail = ""
    if re.search(r"\bwhere\b", head, flags=re.IGNORECASE):
        out = f"{head} AND ({condition})"
    else:
        out = f"{head} WHERE ({condition})"
    return f"{out} {tail}".strip()


def apply_domicile_eligibility(
    sql: str,
    *,
    user_home_state: str | None,
) -> str:
    """
    Add domicile eligibility filter when a specific state is queried and domicile
    wasn't explicitly asked. This uses user's home state profile.
    """
    if not user_home_state:
        return sql

    m = re.search(
        r"\bstate\s*(?:=|ilike)\s*['\"]([^'\"]+)['\"]",
        sql,
        flags=re.IGNORECASE,
    )
    if not m:
        return sql
    raw_target = m.group(1).strip().strip("%")
    target_state = resolve_state(raw_target) or raw_target.strip().upper()
    home_state = resolve_state(user_home_state) or user_home_state.strip().upper()

    if not target_state or target_state == "MCC":
        return sql

    is_home = target_state == home_state
    if is_home:
        domicile_condition = "domicile IN ('DOMICILE', 'OPEN')"
    else:
        domicile_condition = "domicile IN ('NON-DOMICILE', 'OPEN')"

    # If model already included domicile, normalize/override to expected rule.
    if re.search(r"\bdomicile\b", sql, flags=re.IGNORECASE):
        fixed = re.sub(
            r"\bdomicile\s+in\s*\([^)]+\)",
            domicile_condition,
            sql,
            flags=re.IGNORECASE,
        )
        fixed = re.sub(
            r"\bdomicile\s*(?:=|ilike)\s*['\"][^'\"]+['\"]",
            domicile_condition,
            fixed,
            flags=re.IGNORECASE,
        )
    else:
        fixed = _append_where_condition(sql, domicile_condition)

    logger.info(
        "Applied domicile rule: target_state=%s home_state=%s condition=%s",
        target_state,
        home_state,
        domicile_condition,
    )
    return fixed


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


def _numbers_tied_to_neet_score(context: str) -> set[int]:
    """Numbers the user clearly stated as marks/score (not AIR)."""
    found: set[int] = set()
    ctx = context
    # "score ... 540", "scoring 540", "NEET score of 540", "540 marks"
    for m in re.finditer(
        r"(?:neet\s*(?:ug\s*)?score|ug\s*score|marks?|scoring|scored|mark\s+of)"
        r"\D{0,100}(\d{2,4})\b",
        ctx,
        re.IGNORECASE,
    ):
        v = int(m.group(1))
        if 1 <= v <= 720:
            found.add(v)
    for m in re.finditer(
        r"\b(\d{2,4})\b\D{0,50}(?:marks?|points?|score)\b",
        ctx,
        re.IGNORECASE,
    ):
        v = int(m.group(1))
        if 1 <= v <= 720:
            found.add(v)
    for m in re.finditer(
        r"(?:i\s*(?:'m|am)\s*)?scoring\s+(\d{2,4})\b",
        ctx,
        re.IGNORECASE,
    ):
        v = int(m.group(1))
        if 1 <= v <= 720:
            found.add(v)
    return found


def _numbers_tied_to_air_rank(context: str) -> set[int]:
    """Numbers the user clearly stated as rank/AIR."""
    found: set[int] = set()
    for m in re.finditer(
        r"(?:\bair\b|all\s*india\s*rank|\brank|opening\s*rank)\D{0,100}(\d{1,7})\b",
        context,
        re.IGNORECASE,
    ):
        v = int(m.group(1))
        if v >= 1:
            found.add(v)
    for m in re.finditer(
        r"\b(\d{1,7})\b\D{0,40}(?:\bair\b|all\s*india\s*rank|\brank\b)",
        context,
        re.IGNORECASE,
    ):
        v = int(m.group(1))
        if v >= 1:
            found.add(v)
    return found


def _wants_score_above_threshold(context: str, n: int) -> bool:
    """User asked for colleges/scores 'above' N marks (e.g. score above 580)."""
    sn = str(n)
    low = context.lower()
    return bool(
        re.search(rf"(?:above|over|at least|more than|minimum|≥)\s*{sn}\b", low)
        or re.search(rf"\b{sn}\s*(?:or more|and above|\+)", low)
        or re.search(rf"score\s*(?:of\s*)?(?:above|over|at least)\s*{sn}\b", low)
    )


def fix_air_rank_score_confusion(context: str, sql: str) -> str:
    """
    If the LLM put a NEET *mark* into air_rank, rewrite to use column score.
    Does not change SQL when the same number is clearly an AIR in context.
    """
    score_nums = _numbers_tied_to_neet_score(context)
    rank_nums = _numbers_tied_to_air_rank(context)
    if not score_nums:
        return sql

    out = sql
    for n in score_nums:
        if n in rank_nums:
            continue
        wants_above = _wants_score_above_threshold(context, n)
        # Default: eligibility by marks → score <= n (allotted marks at/below user's level)
        ge_repl = f"score >= {n}" if wants_above else f"score <= {n}"
        le_repl = f"score >= {n}" if wants_above else f"score <= {n}"

        pat_ge = re.compile(rf"\bair_rank\s*>=\s*{n}\b", re.IGNORECASE)
        pat_le = re.compile(rf"\bair_rank\s*<=\s*{n}\b", re.IGNORECASE)
        pat_eq = re.compile(rf"\bair_rank\s*=\s*{n}\b", re.IGNORECASE)

        if pat_ge.search(out):
            out = pat_ge.sub(ge_repl, out)
            logger.info(
                "Rewrote mistaken air_rank>=%s → %s (NEET marks context)",
                n,
                ge_repl,
            )
        if pat_le.search(out):
            out = pat_le.sub(le_repl, out)
            logger.info(
                "Rewrote mistaken air_rank<=%s → %s (NEET marks context)",
                n,
                le_repl,
            )
        if pat_eq.search(out):
            out = pat_eq.sub(f"score = {n}", out)
            logger.info("Rewrote mistaken air_rank=%s → score=%s (NEET marks context)", n, n)

    return out


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


def generate_sql(
    client: OpenAI,
    user_question: str,
    *,
    user_home_state: str | None = None,
) -> str:
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
    sql = apply_domicile_eligibility(sql, user_home_state=user_home_state)
    sql = fix_air_rank_score_confusion(normalized, sql)
    sql = fix_air_rank_eligibility_sql(normalized, sql)
    sql = apply_distinct_colleges(sql, context=normalized, limit=50)
    logger.info("SQL after state/course normalization: %s", sql)

    if not _is_safe_select(sql):
        logger.warning("SQL blocked by safety checks: %s", sql)
        raise ValueError("Generated SQL failed safety checks.")
    logger.info("SQL passed safety checks")
    return sql


def generate_counsellor_answer(
    client: OpenAI,
    user_question: str,
    data: list[dict],
    *,
    data_year: str = "2025",
) -> str:
    logger.info("Generating counsellor answer for rows=%d", len(data))
    answer_prompt = f"""
You are a student-friendly NEET counsellor.

User question:
{user_question}

Query results (JSON):
{data}

Your job:
- Give a concise and helpful counselling response.
- **score** = NEET marks; **air_rank** = All India Rank. Never call a score a "rank" or vice versa.
- air_rank in the data is the allotted candidate's AIR (lower = more competitive). If the user gave their own **rank**,
  only describe colleges from the result as realistic options; do NOT say they can get a seat if that row's air_rank
  is a much lower (better) number than their rank unless the data clearly supports it.
- If the user gave **marks/score**, refer to those as marks or score, not AIR.
- Highlight colleges from the result that match the user's question.
- Mention rank trend or cutoff range if visible.
- Mention quota/seat-type trends when available.
- If user asked about another state and results are empty/limited, mention that domicile/eligibility rules may be the reason.
- If no data is found, explain that clearly and suggest better search filters.
- Keep tone simple, practical, and supportive.
- Do not add a year disclaimer yourself; the app will append a standard note about the data year.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful NEET counselling assistant."},
            {"role": "user", "content": answer_prompt.strip()},
        ],
        temperature=0.3,
    )
    body = (response.choices[0].message.content or "").strip()
    note = (
        f"\n\nNote: This information is based on allotment trends from the year {data_year}."
    )
    return f"{body}{note}"

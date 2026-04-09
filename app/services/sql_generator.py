from __future__ import annotations

import logging
import re

from openai import OpenAI

from app.services.query_normalization import (
    CANONICAL_CATEGORIES,
    CANONICAL_COLLEGE_TYPES,
    CANONICAL_COURSES,
    CANONICAL_DOMICILE,
    CANONICAL_STATES,
    fix_sql_state_and_course,
    normalize_user_question,
    resolve_state,
)

_STATE_LIST = ", ".join(sorted(CANONICAL_STATES))
_COURSE_LIST = ", ".join(sorted(CANONICAL_COURSES))
_COLLEGE_TYPE_LIST = ", ".join(sorted(CANONICAL_COLLEGE_TYPES))
_CATEGORY_LIST = ", ".join(sorted(CANONICAL_CATEGORIES))
_DOMICILE_LIST = ", ".join(sorted(CANONICAL_DOMICILE))

SQL_SYSTEM_PROMPT = f"""
You are a PostgreSQL expert for a NEET UG counselling platform. Your job is to convert a student's question into a precise SQL query.

---

TABLE: neet_ug_2025_cutoffs
KEY COLUMNS: state, course, college_type, college_name, category, sub_category, seat_type, domicile, air_rank, score, round

---

EXACT DATABASE VALUES — Use only these spellings in SQL:

states:       {_STATE_LIST}
course:       {_COURSE_LIST}
college_type: {_COLLEGE_TYPE_LIST}
category:     {_CATEGORY_LIST}
domicile:     {_DOMICILE_LIST}

---

CRITICAL LOGIC RULES:

1. SCORE vs RANK (never mix):
   - score = NEET marks (0–720) → filter: score >= [student_score] - 30 AND score <= [student_score] + 20
   - air_rank = All India Rank (higher = worse) → filter: air_rank >= [student_rank] * 0.85 AND air_rank <= [student_rank] * 1.4
   (This range captures realistic "safe", "moderate", and "reach" colleges)

2. DOMICILE LOGIC:
   - Student's home state = queried state → domicile IN ('DOMICILE', 'OPEN')
   - Student in different state → domicile IN ('NON-DOMICILE', 'OPEN')
   - MCC / All India queries → domicile IN ('NON-DOMICILE', 'OPEN')

3. PROFILE vs QUERY:
   - Values mentioned in the CURRENT question ALWAYS override stored profile.
   - Use profile values only when not specified in the question.

4. CATEGORY HANDLING:
   - If category is OBC/OBC-NCL, also include GENERAL rows (students can sometimes compete in open seats).
   - If EWS, include GENERAL as fallback.
   - Use: category IN ('OBC', 'OBC-NCL', 'GENERAL') for OBC students.

5. GENERAL/HYPOTHETICAL QUESTIONS:
   - If no specific category is mentioned, omit the category filter entirely.
   - If no specific state is mentioned, omit the state filter (or use MCC if "all India" is implied).

6. RANKING OUTPUT:
   - For rank-based queries: ORDER BY air_rank ASC
   - For score-based queries: ORDER BY score DESC
   - Always: LIMIT 50

---

HARD RULES:
- Output raw SQL only. No markdown. No explanation. No backticks.
- Single SELECT query on table neet_ug_2025_cutoffs only.
- Never use INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE/CREATE/GRANT/REVOKE.

OUTPUT EXAMPLE:
SELECT college_name, state, course, category, domicile, air_rank, score, round
FROM neet_ug_2025_cutoffs
WHERE state = 'KARNATAKA'
  AND course = 'MBBS'
  AND category IN ('OBC', 'OBC-NCL', 'GENERAL')
  AND domicile IN ('DOMICILE', 'OPEN')
  AND air_rank BETWEEN 40000 AND 70000
ORDER BY air_rank ASC
LIMIT 50
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


def _is_mcc_or_all_india_query(sql: str) -> bool:
    """Check if SQL query targets MCC/AIQ/All India (not a specific state)."""
    # Only check the SQL itself, not context (which may have example text)
    # Check if SQL has state = 'MCC' or similar
    if re.search(r"\bstate\s*(?:=|ilike)\s*['\"]%?mcc%?['\"]", sql, flags=re.IGNORECASE):
        return True
    # Check if SQL has NO state filter at all (could be all-india query)
    if not re.search(r"\bstate\s*(?:=|ilike)", sql, flags=re.IGNORECASE):
        return True
    return False


def _is_personalized_eligibility_question(context: str) -> bool:
    """Check if the user is asking about what they can get (eligibility), not just cutoff info."""
    low = context.lower()
    eligibility_hints = [
        "which colleges can i get",
        "can i get",
        "colleges i can",
        "options for me",
        "my options",
        "what are my",
        "eligible for",
        "where can i get",
        "chances of getting",
        "realistic options",
        "will i get",
        "am i eligible",
        "seats i can",
        "options with my",
        "my rank",
        "my score",
        "my neet",
    ]
    return any(hint in low for hint in eligibility_hints)


def apply_eligibility_filters(
    sql: str,
    *,
    user_home_state: str | None,
    user_category: str | None = None,
    user_college_types: list[str] | None = None,
    user_context: str = "",
) -> str:
    """
    Intelligently add domicile, category, and college_type filters to queries for eligibility questions.
    - Only for eligibility-type questions (user asking what they can get)
    - Skip for general cutoff lookups, specific college queries, trend comparisons
    - Adds filters based on user preferences when not explicitly mentioned in query
    """
    # Only add filters for eligibility questions
    if not _is_personalized_eligibility_question(user_context):
        logger.info("Not an eligibility question, skipping auto-filters")
        return sql
    
    result_sql = sql
    
    # --- CATEGORY FILTER ---
    has_category_filter = bool(
        re.search(r"\bcategory\s+in\s*\(", sql, flags=re.IGNORECASE) or
        re.search(r"\bcategory\s*(?:=|ilike)\s*['\"]", sql, flags=re.IGNORECASE)
    )
    
    if not has_category_filter and user_category:
        # Add category filter for better results
        category_condition = f"category ILIKE '%{user_category}%'"
        result_sql = _append_where_condition(result_sql, category_condition)
        logger.info("Applied category filter: %s", category_condition)
    elif has_category_filter:
        logger.info("Category filter already present in SQL")
    
    # --- COLLEGE TYPE FILTER ---
    has_college_type_filter = bool(
        re.search(r"\bcollege_type\s+in\s*\(", result_sql, flags=re.IGNORECASE) or
        re.search(r"\bcollege_type\s*(?:=|ilike)\s*['\"]", result_sql, flags=re.IGNORECASE)
    )
    
    # Only add college_type filter if user has preferences and not "ALL"
    if not has_college_type_filter and user_college_types:
        # Skip if user selected "ALL" or list is empty
        if "ALL" not in user_college_types and len(user_college_types) > 0:
            if len(user_college_types) == 1:
                college_type_condition = f"college_type = '{user_college_types[0]}'"
            else:
                types_str = "', '".join(user_college_types)
                college_type_condition = f"college_type IN ('{types_str}')"
            result_sql = _append_where_condition(result_sql, college_type_condition)
            logger.info("Applied college_type filter: %s", college_type_condition)
    elif has_college_type_filter:
        logger.info("College type filter already present in SQL")
    
    # --- DOMICILE FILTER ---
    has_domicile_filter = bool(
        re.search(r"\bdomicile\s+in\s*\(", result_sql, flags=re.IGNORECASE) or
        re.search(r"\bdomicile\s*(?:=|ilike)\s*['\"]", result_sql, flags=re.IGNORECASE)
    )
    
    if has_domicile_filter:
        logger.info("Domicile filter already present in SQL, skipping auto-add")
        return result_sql
    
    # Determine target state from SQL
    m = re.search(
        r"\bstate\s*(?:=|ilike)\s*['\"]([^'\"]+)['\"]",
        result_sql,
        flags=re.IGNORECASE,
    )
    
    target_state = None
    if m:
        raw_target = m.group(1).strip().strip("%")
        target_state = resolve_state(raw_target) or raw_target.strip().upper()
    
    # Normalize home state
    home_state = None
    if user_home_state:
        home_state = resolve_state(user_home_state) or user_home_state.strip().upper()
    
    # PRIORITY 1: Check if target state matches home state
    if target_state and home_state and target_state == home_state:
        domicile_condition = "domicile IN ('DOMICILE', 'OPEN')"
        logger.info(
            "Applied domicile rule: target_state=%s matches home_state=%s → %s",
            target_state, home_state, domicile_condition,
        )
    # PRIORITY 2: Check if MCC/All India query
    elif _is_mcc_or_all_india_query(result_sql) or target_state == "MCC":
        domicile_condition = "domicile IN ('NON-DOMICILE', 'OPEN')"
        logger.info("Applied domicile rule for MCC/All India: %s", domicile_condition)
    else:
        domicile_condition = "domicile IN ('NON-DOMICILE', 'OPEN')"
        logger.info(
            "Applied domicile rule: target_state=%s differs from home_state=%s → %s",
            target_state, home_state, domicile_condition,
        )

    return _append_where_condition(result_sql, domicile_condition)


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
    user_category: str | None = None,
    user_college_types: list[str] | None = None,
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
    sql = apply_eligibility_filters(
        sql,
        user_home_state=user_home_state,
        user_category=user_category,
        user_college_types=user_college_types,
        user_context=normalized,
    )
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
You are Anuj, a warm and knowledgeable NEET UG counselling assistant. You are talking with a student who is stressed and needs clear, honest, and caring guidance.

---

STUDENT'S QUESTION:
{user_question}

DATABASE RESULTS (JSON):
{data}

---

YOUR RESPONSE GUIDELINES:

**TONE & STYLE:**
- Talk like a caring senior who has seen hundreds of NEET students — not like a bot reading from a table.
- Use "you" naturally. Keep sentences short. Avoid jargon unless the student used it first.
- It's okay to say "honestly" or "look" to sound real.
- Never sound robotic. Never start with "Based on the data provided..."

**STRUCTURE YOUR ANSWER LIKE THIS:**

1. ACKNOWLEDGE (1 line): Reflect back what they asked in human terms. If they sound anxious, address that briefly.
   Example: "With a score of 480 and OBC category in Karnataka — you've got some solid options, let me walk you through them."

2. MAIN ANSWER: Present the results clearly.
   - If colleges found: Group by college_type (Government first, then Private, then Deemed) or by state if cross-state.
   - Show: College name | Course | Category | Last year's cutoff (rank or score) | Round
   - Highlight 2–3 "you can likely get this" colleges vs 2–3 "stretch" options.
   - For rank queries: use air_rank. For score queries: use score. NEVER mix them up.

3. HONEST INSIGHT (1–2 lines): Give a real counselling nudge.
   Example: "The cutoffs here are from {data_year} rounds — actual next year cutoffs may shift slightly, so treat these as estimates."

4. NEXT STEP (1 line): Suggest what they should do or ask next.
   Example: "Want me to also check options in neighboring states like Tamil Nadu or Andhra?"

---

**IF NO DATA FOUND:**
Don't just say "no results." Say something like:
"Honestly, looking at last year's data, I couldn't find seats in that state for your category at this rank range. That doesn't mean there's no hope — it might mean the competition was very high there. Want me to check other states, or look at private/deemed colleges instead?"

**SPECIAL CASES:**
- If result has only Private/Deemed colleges and student wanted Government: Acknowledge the gap kindly and offer alternatives.
- If student's rank is borderline: Be honest — "This is a close call, you're right at the edge of last year's cutoff."
- If question was hypothetical (not about their profile): Respond naturally without referencing their profile.

---

FORMAT:
- Use simple bullet points or a small table for college lists (max 10–12 rows shown).
- Bold college names.
- Keep total response under 300 words unless the question genuinely needs more.
- End with a warm, action-oriented sentence — never a cold full stop.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are Anuj, a warm and experienced NEET counselling assistant who genuinely cares about helping students."},
            {"role": "user", "content": answer_prompt.strip()},
        ],
        temperature=0.4,
    )
    body = (response.choices[0].message.content or "").strip()
    note = (
        f"\n\n_Note: This information is based on allotment trends from {data_year}._"
    )
    return f"{body}{note}"

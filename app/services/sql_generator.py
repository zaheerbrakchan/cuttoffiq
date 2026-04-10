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
You are a PostgreSQL expert for a NEET UG counselling platform. Convert student questions into SQL queries.

---

TABLE: neet_ug_2025_cutoffs
COLUMNS: state, course, college_type, college_name, category, sub_category, seat_type, domicile, air_rank, score, round

EXACT DATABASE VALUES:
states:       {_STATE_LIST}
course:       {_COURSE_LIST}
college_type: {_COLLEGE_TYPE_LIST}
category:     {_CATEGORY_LIST}
domicile:     {_DOMICILE_LIST}

---

CRITICAL: TARGET STATE vs HOME STATE (DIFFERENT CONCEPTS!)

⚠️ **state** column in SQL = TARGET STATE (where to search for colleges)
⚠️ **home_state** = student's domicile (affects eligibility)

These are DIFFERENT:
- "colleges in Bihar" → state = 'BIHAR' (target)
- "home state is Kerala" → affects category and domicile filter

---

⚠️⚠️ CRITICAL CATEGORY + DOMICILE RULES ⚠️⚠️

**CASE 1: HOME STATE = TARGET STATE (State Counselling in own state)**
- Student's category + sub-category APPLY (reservation benefits)
- domicile IN ('DOMICILE', 'OPEN')
- Example: Home=Rajasthan, Target=Rajasthan, Category=ST
  → category = 'ST' AND domicile IN ('DOMICILE', 'OPEN')

**CASE 2: MCC / AIQ (All India Quota)**
- Student's category APPLIES (reservation benefits in AIQ)
- domicile = 'OPEN' (AIQ seats are open to all)
- Example: Home=Rajasthan, Target=MCC, Category=ST
  → category = 'ST' AND domicile = 'OPEN'

**CASE 3: DIFFERENT STATE (State Counselling in another state)**
- Student's category does NOT apply (no reservation in other states)
- Treat as GENERAL category only
- domicile IN ('NON-DOMICILE', 'OPEN')
- Example: Home=Rajasthan, Target=Bihar, Category=ST
  → category = 'GENERAL' AND domicile IN ('NON-DOMICILE', 'OPEN')
  (NOT category = 'ST' - ST benefits don't apply in Bihar for Rajasthan domicile)

---

COUNSELLING TYPE

- State Counselling: Colleges run by state government
- MCC/AIQ: All India Quota seats (15% govt + central institutes)

If counselling_type = "state" → Apply Case 1 or Case 3 rules
If counselling_type = "mcc" → Apply Case 2 rules

---

UNDERSTANDING THE CONTEXT

The context contains conversation data. THREE types of queries:

1. **SELF QUERY** (user asking about themselves)
   - PROFILE_SQL_HINTS will be provided → USE these values
   - Profile fills defaults for: home_state, category, score/rank, course
   - User's conversational overrides take priority
   - If PROFILE_SQL_HINTS includes preferred college_type, SQL MUST include that exact college_type predicate.
     * Example single type: college_type = 'GOVERNMENT'
     * Example multiple: college_type IN ('GOVERNMENT', 'Private')
   - NEVER broaden beyond requested/hinted college_type (do NOT include Private when asked for Government only).

2. **FRIEND QUERY** (about someone else)
   - NO profile hints (profile doesn't apply to friends)
   - Use ONLY values from the conversation for that person
   
3. **GENERAL QUERY** (hypothetical, like "student with 650 marks")
   - Use ONLY explicitly stated values
   - No profile needed

---

AGGREGATING DATA FROM CONVERSATION

User messages build up data across turns. Look for:
- Target state: "colleges in Bihar", "options in Karnataka" → use in state = 
- Home state: "home state is Kerala", "domicile is UP" → affects category/domicile
- Counselling type: "state counselling", "MCC", "AIQ" → determines which rules apply
- Category: "ST", "OBC", "General" → Apply based on cases above!
- Rank: "rank 150000", "AIR 150000" → air_rank >= 150000
- Score: "score 450", "450 marks" → score <= 450

---

STRING LITERALS:
- Use EXACT spellings from the lists above
- CORRECT: college_type = 'GOVERNMENT', state = 'KARNATAKA'
- Case: User input may be lowercase ("st", "kerala") - convert to UPPERCASE in SQL

---

CRITICAL LOGIC RULES:

1. SCORE vs RANK (never mix):
   - score = NEET marks (0-720) → use score <= [value]
   - air_rank = All India Rank → use air_rank >= [value]
   - Never use BETWEEN for eligibility questions

2. CATEGORY + DOMICILE (based on cases above):
   - Same state → Use student's category, domicile IN ('DOMICILE', 'OPEN')
   - MCC/AIQ → Use student's category, domicile = 'OPEN'
   - Different state → Use GENERAL only, domicile IN ('NON-DOMICILE', 'OPEN')

3. CATEGORY FALLBACK:
   - OBC/OBC-NCL → include GENERAL (category IN ('OBC', 'OBC-NCL', 'GENERAL'))
   - When student's category applies (Case 1, 2)

4. OUTPUT:
   - ORDER BY air_rank ASC (for rank) or score DESC (for score)
   - Always LIMIT 50

5. STRICT COLLEGE_TYPE RULE:
   - When a strict college_type directive is present in the user prompt, treat it as mandatory.
   - Include an exact college_type filter in WHERE and do not substitute or widen values.

---

OUTPUT: Raw SQL only. No markdown, no backticks, no explanation.

EXAMPLE 1 (Same state - Category applies):
Home=Karnataka, Target=Karnataka, Category=OBC, State Counselling
SELECT college_name, college_type, category, sub_category, seat_type, quota, eligibility, air_rank, score
FROM neet_ug_2025_cutoffs
WHERE state = 'KARNATAKA'
  AND course = 'MBBS'
  AND category IN ('OBC', 'OBC-NCL', 'GENERAL')
  AND domicile IN ('DOMICILE', 'OPEN')
  AND score <= 450
ORDER BY score DESC
LIMIT 50

EXAMPLE 2 (MCC - Category applies):
Home=Rajasthan, Target=MCC, Category=ST
SELECT college_name, college_type, category, sub_category, seat_type, quota, eligibility, air_rank, score
FROM neet_ug_2025_cutoffs
WHERE state = 'MCC'
  AND course = 'MBBS'
  AND category = 'ST'
  AND domicile = 'OPEN'
  AND air_rank >= 150000
ORDER BY air_rank ASC
LIMIT 50

EXAMPLE 3 (Different state - GENERAL only):
Home=Rajasthan, Target=Bihar, Category=ST (but doesn't apply!)
SELECT college_name, college_type, category, sub_category, seat_type, quota, eligibility, air_rank, score
FROM neet_ug_2025_cutoffs
WHERE state = 'BIHAR'
  AND course = 'MBBS'
  AND category = 'GENERAL'
  AND domicile IN ('NON-DOMICILE', 'OPEN')
  AND air_rank >= 150000
ORDER BY air_rank ASC
LIMIT 50

EXAMPLE (home_state Kerala, target_state Karnataka):
SELECT college_name, college_type, category, sub_category, seat_type, quota, eligibility, air_rank, score
FROM neet_ug_2025_cutoffs
WHERE state = 'KARNATAKA'
  AND course = 'MBBS'
  AND category IN ('OBC', 'OBC-NCL', 'GENERAL')
  AND domicile IN ('NON-DOMICILE', 'OPEN')
  AND score <= 450
ORDER BY score DESC
LIMIT 50
"""
logger = logging.getLogger("neet_assistant.sql")
_MAX_LOG_CHARS = 5000


def _clip(text: str, limit: int = _MAX_LOG_CHARS) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated {len(text) - limit} chars]"


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


def ensure_output_columns(sql: str) -> str:
    """
    Ensure result projection always includes both air_rank and score.
    If query uses SELECT *, leave unchanged.
    """
    m = re.search(r"^\s*select\s+(.*?)\s+from\s", sql, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return sql
    select_list = m.group(1).strip()
    if re.search(r"^\*$", select_list) or re.search(r"\b\w+\.\*\b", select_list):
        return sql

    needed: list[str] = []
    if not re.search(r"\bair_rank\b", select_list, flags=re.IGNORECASE):
        needed.append("air_rank")
    if not re.search(r"\bscore\b", select_list, flags=re.IGNORECASE):
        needed.append("score")
    if not needed:
        return sql

    new_select_list = f"{select_list}, {', '.join(needed)}"
    start, end = m.span(1)
    fixed = f"{sql[:start]}{new_select_list}{sql[end:]}"
    logger.info("Added missing output columns to SELECT: %s", ", ".join(needed))
    return fixed


def _extract_latest_student_message(context: str) -> str:
    marker = "Student (latest message):"
    idx = context.rfind(marker)
    if idx == -1:
        return context
    return context[idx + len(marker) :].strip()


def _extract_recent_student_messages(context: str) -> list[str]:
    msgs: list[str] = []
    for line in (context or "").splitlines():
        line = line.strip()
        if line.startswith("Student (latest message):"):
            content = line.split("Student (latest message):", 1)[1].strip()
            if content:
                msgs.append(content)
        elif line.startswith("Student:"):
            content = line.split("Student:", 1)[1].strip()
            if content:
                msgs.append(content)
    if not msgs and context.strip():
        msgs.append(context.strip())
    return msgs


def _extract_rank_value(text: str) -> int | None:
    # Prefer explicit "my rank" pattern first.
    own = _extract_user_air(text)
    if own is not None:
        return own
    # Then accept generic AIR/rank mentions (friend/he/she statements).
    ranks = sorted(_numbers_tied_to_air_rank(text))
    if ranks:
        return ranks[-1]
    return None


def _extract_primary_metric_target(context: str) -> tuple[str, int] | None:
    """
    Returns ("air_rank", value) or ("score", value) from user context when available.
    Preference: explicit rank/AIR first, then explicit score/marks.
    """
    latest = _extract_latest_student_message(context)

    # Prefer explicit metric from latest user message.
    user_air = _extract_user_air(latest)
    if user_air is not None:
        return ("air_rank", user_air)
    score_nums = sorted(_numbers_tied_to_neet_score(latest))
    if score_nums:
        return ("score", score_nums[-1])

    # IMPORTANT: Do not fallback to older memory turns.
    # Metric target must come from latest message only to avoid cross-turn leakage.
    return None


def _is_options_style_question(context: str) -> bool:
    low = context.lower()
    return bool(
        re.search(
            r"\b(which colleges|can (?:i|he|she) get|options?|possible colleges|looking option|eligible)\b",
            low,
        )
    )


def enforce_score_ceiling_for_options(context: str, sql: str) -> str:
    """
    For options/eligibility questions with stated NEET marks, ensure SQL never
    fetches rows above the student's own score.
    """
    latest = _extract_latest_student_message(context)
    if not _is_options_style_question(latest):
        return sql

    score_nums = sorted(_numbers_tied_to_neet_score(latest))
    if not score_nums:
        return sql
    score_ceiling = score_nums[-1]

    out = sql

    def _rewrite_between(m: re.Match[str]) -> str:
        a = int(m.group(1))
        b = int(m.group(2))
        low = min(a, b)
        high = min(max(a, b), score_ceiling)
        if low > high:
            low = high
        return f"score BETWEEN {low} AND {high}"

    out = re.sub(
        r"\bscore\s+between\s+(\d{1,4})\s+and\s+(\d{1,4})\b",
        _rewrite_between,
        out,
        flags=re.IGNORECASE,
    )

    has_score_cap = bool(
        re.search(rf"\bscore\s*<=\s*{score_ceiling}\b", out, flags=re.IGNORECASE)
        or re.search(
            rf"\bscore\s+between\s+\d{{1,4}}\s+and\s+{score_ceiling}\b",
            out,
            flags=re.IGNORECASE,
        )
    )
    if not has_score_cap:
        out = _append_where_condition(out, f"score <= {score_ceiling}")
    logger.info("Applied options score ceiling: score <= %s", score_ceiling)
    return out


def enforce_strict_metric_rules(
    context: str,
    sql: str,
    *,
    extracted: dict[str, object] | None = None,
) -> str:
    """
    Strict eligibility rule (latest message only):
    - score-based: score <= student's score (no ranges)
    - rank-based: air_rank >= student's rank (no ranges)
    """
    extracted = extracted or {}
    mode = str(extracted.get("query_mode", "")).lower()
    metric_type = str(extracted.get("metric_type", "")).lower()
    metric_value_raw = extracted.get("metric_value")
    metric_value = int(metric_value_raw) if isinstance(metric_value_raw, (int, float)) else None

    # Apply strict metric rewriting only for eligibility mode.
    # If mode is missing/unknown, keep backward-compatible behavior.
    if mode and mode not in ("eligibility", "unknown"):
        return sql

    latest = _extract_latest_student_message(context)
    metric_source = latest
    if not _numbers_tied_to_neet_score(metric_source) and _extract_rank_value(metric_source) is None:
        msgs = _extract_recent_student_messages(context)
        # Look backward for nearest student message carrying score/rank.
        for msg in reversed(msgs[:-1]):
            if _numbers_tied_to_neet_score(msg) or _extract_rank_value(msg) is not None:
                metric_source = msg
                break
    out = sql

    # STRICT SCORE RULE (prefer LLM extracted metric when available)
    score_nums = [metric_value] if metric_type == "score" and metric_value is not None else sorted(_numbers_tied_to_neet_score(metric_source))
    if score_nums:
        n = score_nums[-1]
        out = re.sub(
            r"\bscore\s+between\s+\d{1,4}\s+and\s+\d{1,4}\b",
            f"score <= {n}",
            out,
            flags=re.IGNORECASE,
        )
        out = re.sub(r"\bscore\s*>=\s*\d{1,4}\b", f"score <= {n}", out, flags=re.IGNORECASE)
        out = re.sub(r"\bscore\s*<\s*\d{1,4}\b", f"score <= {n}", out, flags=re.IGNORECASE)
        out = re.sub(r"\bscore\s*=\s*\d{1,4}\b", f"score <= {n}", out, flags=re.IGNORECASE)
        out = re.sub(r"\bscore\s*<=\s*\d{1,4}\b", f"score <= {n}", out, flags=re.IGNORECASE)
        if not re.search(r"\bscore\s*<=\s*\d{1,4}\b", out, flags=re.IGNORECASE):
            out = _append_where_condition(out, f"score <= {n}")
        logger.info("Enforced STRICT score rule: score <= %s (source=%s)", n, metric_source)

    # STRICT RANK RULE (prefer LLM extracted metric when available)
    rank = metric_value if metric_type == "rank" and metric_value is not None else _extract_rank_value(metric_source)
    if rank is not None:
        out = re.sub(
            r"\bair_rank\s+between\s+\d{1,7}\s+and\s+\d{1,7}\b",
            f"air_rank >= {rank}",
            out,
            flags=re.IGNORECASE,
        )
        out = re.sub(r"\bair_rank\s*<=\s*\d{1,7}\b", f"air_rank >= {rank}", out, flags=re.IGNORECASE)
        out = re.sub(r"\bair_rank\s*<\s*\d{1,7}\b", f"air_rank >= {rank}", out, flags=re.IGNORECASE)
        out = re.sub(r"\bair_rank\s*=\s*\d{1,7}\b", f"air_rank >= {rank}", out, flags=re.IGNORECASE)
        out = re.sub(r"\bair_rank\s*>=\s*\d{1,7}\b", f"air_rank >= {rank}", out, flags=re.IGNORECASE)
        if not re.search(r"\bair_rank\s*>=\s*\d{1,7}\b", out, flags=re.IGNORECASE):
            out = _append_where_condition(out, f"air_rank >= {rank}")
        logger.info("Enforced STRICT AIR rule: air_rank >= %s (source=%s)", rank, metric_source)

    return out


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
        "can he get",
        "can she get",
        "colleges i can",
        "colleges he can",
        "colleges she can",
        "options for me",
        "looking for",
        "help with options",
        "for my friend",
        "his rank",
        "her rank",
        "his score",
        "her score",
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


def _sql_hint_escape(value: str) -> str:
    return (value or "").replace("'", "''")


def _build_profile_sql_hints(
    *,
    user_home_state: str | None,
    user_category: str | None,
    user_college_types: list[str] | None,
) -> str:
    """
    Injected into the SQL LLM user message so the model applies exact DB tokens
    (no post-hoc string splitting or regex fixes).
    """
    types = [t for t in (user_college_types or []) if t and str(t).strip() and str(t).strip().upper() != "ALL"]
    if not (user_home_state or user_category or types):
        return ""

    lines = [
        "PROFILE_SQL_HINTS (mandatory when present — copy these tokens verbatim into WHERE; "
        "each is ONE string literal, never character-by-character):",
    ]
    if user_home_state:
        h = _sql_hint_escape(user_home_state.strip().upper())
        lines.append(f"- Student home state (state column): '{h}'")
    if user_category:
        c = _sql_hint_escape(str(user_category).strip().upper())
        lines.append(
            f"- Reservation category (category column): include '{c}' per CATEGORY HANDLING rules "
            f"(expand with related categories when rules say so)."
        )
    if len(types) == 1:
        t0 = _sql_hint_escape(str(types[0]).strip())
        lines.append(f"- Preferred college_type: college_type = '{t0}'")
    elif len(types) > 1:
        in_list = ", ".join(f"'{_sql_hint_escape(str(t).strip())}'" for t in types)
        lines.append(f"- Preferred college_type: college_type IN ({in_list})")
    return "\n".join(lines)


def apply_eligibility_filters(
    sql: str,
    *,
    user_home_state: str | None,
    user_category: str | None = None,
    user_context: str = "",
    is_eligibility: bool | None = None,
    subject: str | None = None,
) -> str:
    """
    Add domicile and category filters for eligibility questions when needed.
    college_type is handled by the SQL LLM via PROFILE_SQL_HINTS in generate_sql.
    
    For friend queries (subject='friend'), we skip most profile-based filters
    since the profile belongs to the logged-in user, not to their friend.
    """
    # For friend queries, skip profile-based eligibility filters
    # The friend has different category/domicile situation
    is_friend_query = (subject or "").lower() == "friend"
    if is_friend_query:
        logger.info("Friend query detected - skipping profile-based eligibility filters")
        return sql
    
    # Prefer LLM-extracted mode when available; fallback to phrase heuristic.
    effective_eligibility = is_eligibility if is_eligibility is not None else _is_personalized_eligibility_question(user_context)
    if not effective_eligibility:
        logger.info("Not an eligibility question, skipping auto-filters")
        return sql
    
    result_sql = sql

    # Determine target state from SQL first (needed for category + domicile rules)
    m = re.search(
        r"\bstate\s*(?:=|ilike)\s*['\"]([^'\"]+)['\"]",
        result_sql,
        flags=re.IGNORECASE,
    )
    target_state = None
    if m:
        raw_target = m.group(1).strip().strip("%")
        target_state = resolve_state(raw_target) or raw_target.strip().upper()

    home_state = None
    if user_home_state:
        home_state = resolve_state(user_home_state) or user_home_state.strip().upper()

    is_mcc_or_all_india = _is_mcc_or_all_india_query(result_sql) or target_state == "MCC"
    is_other_state = bool(target_state and home_state and target_state != home_state and not is_mcc_or_all_india)

    # --- CATEGORY FILTER ---
    has_category_filter = bool(
        re.search(r"\bcategory\s+in\s*\(", result_sql, flags=re.IGNORECASE)
        or re.search(r"\bcategory\s*(?:=|ilike)\s*['\"]", result_sql, flags=re.IGNORECASE)
    )

    if is_other_state:
        # Other-state counselling should ALWAYS be treated as open/general bucket.
        # Remove existing category predicates first, then enforce GENERAL/OPEN.
        result_sql = re.sub(
            r"\s+and\s+\(?\s*category\s*(?:=|ilike)\s*'[^']+'\s*\)?",
            "",
            result_sql,
            flags=re.IGNORECASE,
        )
        result_sql = re.sub(
            r"\s+and\s+\(?\s*category\s+in\s*\([^)]+\)\s*\)?",
            "",
            result_sql,
            flags=re.IGNORECASE,
        )
        category_condition = "(category ILIKE '%GENERAL%' OR category ILIKE '%GEN%' OR category ILIKE '%OPEN%')"
        result_sql = _append_where_condition(result_sql, category_condition)
        logger.info(
            "Applied enforced other-state category rule (home=%s, target=%s): %s",
            home_state,
            target_state,
            category_condition,
        )
    elif has_category_filter:
        logger.info("Category filter already present in SQL")
    elif user_category:
        category_condition = f"category ILIKE '%{user_category}%'"
        result_sql = _append_where_condition(result_sql, category_condition)
        logger.info("Applied category filter: %s", category_condition)

    # college_type: SQL LLM + PROFILE_SQL_HINTS in generate_sql (no post-append here).

    # --- DOMICILE FILTER ---
    has_domicile_filter = bool(
        re.search(r"\bdomicile\s+in\s*\(", result_sql, flags=re.IGNORECASE) or
        re.search(r"\bdomicile\s*(?:=|ilike)\s*['\"]", result_sql, flags=re.IGNORECASE)
    )
    
    if has_domicile_filter:
        logger.info("Domicile filter already present in SQL, skipping auto-add")
        return result_sql
    
    # PRIORITY 1: Check if target state matches home state
    if target_state and home_state and target_state == home_state:
        domicile_condition = "domicile IN ('DOMICILE', 'OPEN')"
        logger.info(
            "Applied domicile rule: target_state=%s matches home_state=%s → %s",
            target_state, home_state, domicile_condition,
        )
    # PRIORITY 2: Check if MCC/All India query
    elif is_mcc_or_all_india:
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
    r"(which\s+colleges|can\s+(?:i|he|she)\s+get|colleges\s+(?:i|he|she)\s+can|eligible|options?\s+(?:for|with)|looking\s+for|help\s+with\s+options?)",
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
    latest = _extract_latest_student_message(context)
    # Use latest message only to avoid pulling old scores/ranks from memory history.
    score_nums = _numbers_tied_to_neet_score(latest)
    rank_nums = _numbers_tied_to_air_rank(latest)
    if not score_nums:
        return sql

    out = sql
    for n in score_nums:
        if n in rank_nums:
            continue
        wants_above = _wants_score_above_threshold(latest, n)
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
    latest = _extract_latest_student_message(question)
    if not _is_eligibility_question(latest):
        return sql
    user_air = _extract_user_air(latest)
    if user_air is None:
        return sql
    out = sql

    # For "what can I get" with given rank, keep only worse-or-equal ranks (>= user rank).
    # 1) Remove any upper-bound component from BETWEEN / <= style filters.
    out = re.sub(
        rf"\bair_rank\s+between\s+\d+\s+and\s+\d+\b",
        f"air_rank >= {user_air}",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        rf"\bair_rank\s*<=\s*\d+\b",
        f"air_rank >= {user_air}",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        rf"\bair_rank\s*=\s*{user_air}\b",
        f"air_rank >= {user_air}",
        out,
        flags=re.IGNORECASE,
    )
    # Normalize any existing lower-bound to the current user's rank.
    out = re.sub(
        rf"\bair_rank\s*>=\s*\d+\b",
        f"air_rank >= {user_air}",
        out,
        flags=re.IGNORECASE,
    )

    if not re.search(r"\bair_rank\s*>=\s*\d+\b", out, flags=re.IGNORECASE):
        out = _append_where_condition(out, f"air_rank >= {user_air}")

    logger.info("Enforced eligibility AIR rule from latest message: air_rank >= %s", user_air)
    return out


def generate_sql(
    client: OpenAI,
    user_question: str,
    *,
    user_home_state: str | None = None,
    user_category: str | None = None,
    user_college_types: list[str] | None = None,
    extracted: dict[str, object] | None = None,
    request_id: str | None = None,
) -> str:
    rid = request_id or "-"
    normalized = normalize_user_question(user_question)
    logger.info("[%s] Generating SQL for question length=%d", rid, len(user_question))
    extracted = extracted or {}
    mode = str(extracted.get("query_mode", "")).lower()
    metric_type = str(extracted.get("metric_type", "")).lower()
    metric_value = extracted.get("metric_value")
    
    # Get query type: self / friend / general
    query_type = str(extracted.get("query_type", extracted.get("subject", ""))).lower()
    use_profile = extracted.get("use_profile_defaults", False) or query_type == "self"
    
    # Profile hints only for SELF queries
    # For FRIEND or GENERAL queries, the profile doesn't apply
    if query_type in ("friend", "general"):
        profile_hints = ""
        logger.info("[%s] Query type '%s' - skipping profile hints", rid, query_type)
    elif use_profile:
        profile_hints = _build_profile_sql_hints(
            user_home_state=user_home_state,
            user_category=user_category,
            user_college_types=user_college_types,
        )
        logger.info("[%s] Query type '%s' with use_profile=True - applying profile hints", rid, query_type)
    else:
        profile_hints = ""
        logger.info("[%s] Query type '%s' with use_profile=False - skipping profile hints", rid, query_type)
    
    metric_directive = ""
    if mode == "eligibility" and metric_type in ("score", "rank") and isinstance(metric_value, (int, float)):
        if metric_type == "score":
            metric_directive = f"STRICT_SQL_RULE: This is eligibility query. Use ONLY score <= {int(metric_value)}. Do not use BETWEEN."
        else:
            metric_directive = f"STRICT_SQL_RULE: This is eligibility query. Use ONLY air_rank >= {int(metric_value)}. Do not use BETWEEN."

    strict_college_type_directive = ""
    strict_types = [
        str(t).strip()
        for t in (user_college_types or [])
        if str(t).strip() and str(t).strip().upper() != "ALL"
    ]
    if strict_types:
        if len(strict_types) == 1:
            strict_college_type_directive = (
                f"STRICT_SQL_RULE: Must include exact filter college_type = '{strict_types[0]}'. "
                "Do not include any other college_type."
            )
        else:
            allowed = ", ".join(f"'{t}'" for t in strict_types)
            strict_college_type_directive = (
                f"STRICT_SQL_RULE: Must include exact filter college_type IN ({allowed}). "
                "Do not include college_type values outside this set."
            )

    parts = [
        p
        for p in (
            profile_hints.strip(),
            metric_directive.strip(),
            strict_college_type_directive.strip(),
            normalized.strip(),
        )
        if p
    ]
    llm_user_message = "\n\n".join(parts)
    logger.info("[%s] SQL LLM system prompt:\n%s", rid, _clip(SQL_SYSTEM_PROMPT.strip()))
    logger.info("[%s] SQL LLM user prompt:\n%s", rid, _clip(llm_user_message))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SQL_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": llm_user_message},
        ],
        temperature=0,
    )

    raw_sql = response.choices[0].message.content or ""
    logger.info("[%s] SQL LLM raw output:\n%s", rid, _clip(raw_sql))
    sql = _ensure_limit(_clean_sql(raw_sql), limit=50)
    logger.info("[%s] SQL after clean/limit:\n%s", rid, _clip(sql))
    sql = fix_sql_state_and_course(sql)
    logger.info("[%s] SQL after state/course normalization:\n%s", rid, _clip(sql))
    
    # Apply eligibility filters only for self queries
    sql = apply_eligibility_filters(
        sql,
        user_home_state=user_home_state if use_profile else None,
        user_category=user_category if use_profile else None,
        user_context=normalized,
        is_eligibility=(mode == "eligibility") if mode else None,
        subject=query_type,
    )
    logger.info("[%s] SQL after eligibility filters:\n%s", rid, _clip(sql))
    sql = ensure_output_columns(sql)
    sql = fix_air_rank_score_confusion(normalized, sql)
    sql = fix_air_rank_eligibility_sql(normalized, sql)
    sql = enforce_score_ceiling_for_options(normalized, sql)
    sql = enforce_strict_metric_rules(normalized, sql, extracted=extracted)
    logger.info("[%s] SQL after strict metric rules:\n%s", rid, _clip(sql))
    sql = apply_distinct_colleges(sql, context=normalized, limit=50)
    logger.info("[%s] Final SQL after distinct wrapping:\n%s", rid, _clip(sql))

    if not _is_safe_select(sql):
        logger.warning("[%s] SQL blocked by safety checks:\n%s", rid, _clip(sql))
        raise ValueError("Generated SQL failed safety checks.")
    logger.info("[%s] SQL passed safety checks", rid)
    return sql


def generate_counsellor_answer(
    client: OpenAI,
    user_question: str,
    data: list[dict],
    *,
    data_year: str = "2025",
    request_id: str | None = None,
) -> str:
    rid = request_id or "-"
    logger.info("[%s] Generating counsellor answer for rows=%d", rid, len(data))
    if not data:
        logger.info("[%s] Skipping answer LLM call due to zero rows", rid)
        return (
            "I checked this request carefully, but I couldn't find matching colleges in the current cutoff data for these exact filters.\n\n"
            "If you want, I can widen the search by trying nearby states, broader college types, or a different category/state combination."
            f"\n\n_Note: This information is based on allotment trends from {data_year}._"
        )

    answer_prompt = f"""
You are Anuj, a warm and knowledgeable NEET UG counselling assistant. Provide clear, honest, and helpful guidance.

IMPORTANT TONE RULES:
- Be warm but professional
- NEVER assume emotions ("I understand you're stressed", "Don't worry", "I know this is difficult")
- Don't project feelings - just acknowledge and help directly
- Start with the facts, not emotional assumptions

---

STUDENT'S QUESTION:
{user_question}

DATABASE RESULTS (JSON):
{data}

---

YOUR RESPONSE GUIDELINES:

**CRITICAL ACCURACY RULES:**
- Use ONLY the current STUDENT'S QUESTION and the DATABASE RESULTS in this prompt.
- Do NOT use memory from earlier chat turns.
- Do NOT mention any college, rank, score, or state that is not present in DATABASE RESULTS.
- If a value is missing in a row, show "-" instead of guessing.

**TONE & STYLE:**
- Talk like a knowledgeable senior, not a bot
- Use "you" naturally. Keep sentences short.
- Be direct and helpful without assuming emotions
- Never start with "Based on the data provided..." or "I understand you're stressed/anxious..."

**STRUCTURE YOUR ANSWER LIKE THIS:**

1. ACKNOWLEDGE (1 line): Reflect what they asked factually.
   Example: "Here are the MBBS options in Bihar with AIR 150000 in General category."
   NOT: "I understand you're feeling anxious about..."

2. DOMICILE NOTE (if applicable): If searching in a state different from home state, mention:
   - "Since [target_state] is not your domicile state, you're eligible for Non-Domicile/Open seats only."
   - "Government college domicile seats require you to be from that state."
   - This affects which colleges they can actually apply to.

3. MAIN ANSWER: Present the results clearly.
   - Group by college_type (Government first, then Private, then Deemed)
   - ALWAYS use this table format with these columns:
     Institution Name | College Type | Category | Sub Category | Seat Type | Quota | Eligibility | AIR | Score
   - Map database columns: college_name→Institution Name, air_rank→AIR
   - If a value is missing, show "-"
   - Highlight realistic options vs stretch options

4. HONEST INSIGHT (1-2 lines): Give real counselling guidance.
   Example: "The cutoffs here are from {data_year} rounds — actual cutoffs may shift slightly."

5. NEXT STEP (1 line): Suggest what they can do next.
   Example: "Want me to check MCC counselling options too, or look at neighboring states?"

---

**CATEGORY + DOMICILE ELIGIBILITY (explain when relevant):**

CASE 1 - Same State (State Counselling in home state):
- "Since [state] is your home state, your [category] reservation applies."
- "You're eligible for Domicile and Open seats."

CASE 2 - MCC/AIQ:
- "In MCC/All India Quota, your [category] reservation applies."
- "AIQ seats are open to candidates from all states."

CASE 3 - Different State:
- "Since [target] is not your home state, your [category] reservation doesn't apply here."
- "In [target] state counselling, you'll be considered under General category."
- "You can only apply for Non-Domicile/Open seats (mostly Private colleges)."

**IF NO DATA FOUND:**
Say clearly: "I couldn't find matching colleges with these filters. Would you like me to check private colleges, MCC counselling, or nearby states?"

**SPECIAL CASES:**
- If only Private results when student wanted Government: Explain domicile restriction
- If searching different state: Remind that category benefits don't apply
- For MCC queries: Note that category benefits DO apply

---

FORMAT:
- Use simple bullet points or a small table (max 10-12 rows)
- Table headers: Institution Name | College Type | Category | Sub Category | Seat Type | Quota | Eligibility | AIR | Score
- Bold institution names
- Keep response under 300 words
- End with an actionable next step
"""
    logger.info("[%s] Answer LLM user prompt:\n%s", rid, _clip(answer_prompt.strip()))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are Anuj, a warm and experienced NEET counselling assistant who genuinely cares about helping students."},
            {"role": "user", "content": answer_prompt.strip()},
        ],
        temperature=0.4,
    )
    body = (response.choices[0].message.content or "").strip()
    logger.info("[%s] Answer LLM raw output:\n%s", rid, _clip(body))
    note = (
        f"\n\n_Note: This information is based on allotment trends from {data_year}._"
    )
    return f"{body}{note}"

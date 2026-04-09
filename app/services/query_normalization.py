"""
Map user-facing state/course wording to exact values stored in `neet_ug_2025_cutoffs`.

DB uses short codes (e.g. J&K, TAMILNADU) while users often say full names.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("neet_assistant.normalize")

# Exact values as stored in the database (state column).
CANONICAL_STATES: frozenset[str] = frozenset(
    {
        "ANDHRA",
        "ARUNACHAL",
        "BIHAR",
        "CHHATTISGARH",
        "DELHI",
        "GUJARAT",
        "HARYANA",
        "HIMACHAL",
        "J&K",
        "JHARKHAND",
        "KARNATAKA",
        "KERALA",
        "MADHYA PRADESH",
        "MCC",
        "NAGALAND",
        "ODISHA",
        "PUNJAB",
        "RAJASTHAN",
        "TAMILNADU",
        "TELANGANA",
        "UTTAR PRADESH",
        "UTTARAKHAND",
        "WEST BENGAL",
    }
)

# Exact values as stored in the database (course column).
CANONICAL_COURSES: frozenset[str] = frozenset(
    {
        "B.Sc. Nursing",
        "BDS",
        "MBBS",
    }
)

# Lowercase alias -> canonical state (longer / more specific phrases first).
STATE_ALIASES: dict[str, str] = {
    "jammu and kashmir": "J&K",
    "jammu & kashmir": "J&K",
    "jammu kashmir": "J&K",
    "jammu": "J&K",
    "jk": "J&K",
    "j&k": "J&K",
    "tamil nadu": "TAMILNADU",
    "tamilnadu": "TAMILNADU",
    "madhya pradesh": "MADHYA PRADESH",
    "mp": "MADHYA PRADESH",
    "m p": "MADHYA PRADESH",
    "uttar pradesh": "UTTAR PRADESH",
    "up": "UTTAR PRADESH",
    "u p": "UTTAR PRADESH",
    "west bengal": "WEST BENGAL",
    "wb": "WEST BENGAL",
    "w b": "WEST BENGAL",
    "arunachal pradesh": "ARUNACHAL",
    "himachal pradesh": "HIMACHAL",
    "hp": "HIMACHAL",
    "h p": "HIMACHAL",
    "andhra pradesh": "ANDHRA",
    "andra": "ANDHRA",
    "ap": "ANDHRA",
    "a p": "ANDHRA",
    "uttarakhand": "UTTARAKHAND",
    "uk": "UTTARAKHAND",
    "u k": "UTTARAKHAND",
    "chhattisgarh": "CHHATTISGARH",
    "cg": "CHHATTISGARH",
    "c g": "CHHATTISGARH",
    "orissa": "ODISHA",
    "telangana": "TELANGANA",
    "karnataka": "KARNATAKA",
    "rajasthan": "RAJASTHAN",
    "punjab": "PUNJAB",
    "bihar": "BIHAR",
    "kerala": "KERALA",
    "gujarat": "GUJARAT",
    "haryana": "HARYANA",
    "jharkhand": "JHARKHAND",
    "nagaland": "NAGALAND",
    "delhi": "DELHI",
    "tn": "TAMILNADU",
    "t n": "TAMILNADU",
    # All India / AIQ rounds are stored with state = MCC in this dataset.
    "all india quota": "MCC",
    "all india": "MCC",
    "aiq": "MCC",
    "all india counselling": "MCC",
    "mcc counselling": "MCC",
    "medical counselling committee": "MCC",
}

COURSE_ALIASES: dict[str, str] = {
    "mbbs": "MBBS",
    "bds": "BDS",
    "b.sc. nursing": "B.Sc. Nursing",
    "bsc nursing": "B.Sc. Nursing",
    "b.sc nursing": "B.Sc. Nursing",
    "bachelor of science nursing": "B.Sc. Nursing",
}


def _normalize_key(s: str) -> str:
    # Normalize punctuation-heavy forms like "U.P.", "M.P." and "J & K".
    s = s.strip().lower().replace(".", "")
    s = re.sub(r"\s*&\s*", "&", s)
    return re.sub(r"\s+", " ", s)


def resolve_state(text: str) -> str | None:
    """Return canonical state if text matches an alias or exact canonical."""
    key = _normalize_key(text)
    if key in STATE_ALIASES:
        return STATE_ALIASES[key]
    # e.g. "Jammu And Kashmir" -> "jammu and kashmir"
    key_and = _normalize_key(text.replace("&", " and "))
    if key_and in STATE_ALIASES:
        return STATE_ALIASES[key_and]
    key_amp = _normalize_key(text.replace(" and ", " & "))
    if key_amp in STATE_ALIASES:
        return STATE_ALIASES[key_amp]
    upper = text.strip().upper()
    if upper in CANONICAL_STATES:
        return upper
    return None


def resolve_course(text: str) -> str | None:
    key = _normalize_key(text)
    if key == "nursing":
        return "B.Sc. Nursing"
    if key in COURSE_ALIASES:
        return COURSE_ALIASES[key]
    upper = text.strip().upper()
    if upper == "MBBS":
        return "MBBS"
    if upper == "BDS":
        return "BDS"
    if "B.SC" in upper and "NURS" in upper:
        return "B.Sc. Nursing"
    if upper in CANONICAL_COURSES:
        return upper
    return None


def normalize_user_question(question: str) -> str:
    """
    Replace common state/course phrases in the user question so the model
    tends to emit correct ILIKE literals.
    """
    q = question
    # States: longest phrases first
    phrase_order = sorted(STATE_ALIASES.keys(), key=len, reverse=True)
    for phrase in phrase_order:
        canon = STATE_ALIASES[phrase]
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        q = pattern.sub(canon, q)

    # Courses
    for phrase in sorted(COURSE_ALIASES.keys(), key=len, reverse=True):
        canon = COURSE_ALIASES[phrase]
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        q = pattern.sub(canon, q)
    if q != question:
        logger.info("Normalized question: %r -> %r", question, q)
    return q


_ILIKE_STATE = re.compile(
    r"(?P<prefix>\bstate\s+ILIKE\s+)(?P<quote>['\"])(?P<val>[^'\"]*)(?P=quote)",
    re.IGNORECASE,
)
_ILIKE_COURSE = re.compile(
    r"(?P<prefix>\bcourse\s+ILIKE\s+)(?P<quote>['\"])(?P<val>[^'\"]*)(?P=quote)",
    re.IGNORECASE,
)


def _fix_ilike_match(
    m: re.Match[str],
    resolver,
    field_name: str,
) -> str:
    prefix = m.group("prefix")
    quote = m.group("quote")
    val = m.group("val")
    had_leading = val.startswith("%")
    had_trailing = val.endswith("%")
    core = val.strip("%").strip()
    if not core:
        return m.group(0)
    resolved = resolver(core)
    if resolved is None:
        return m.group(0)
    new_val = resolved
    if had_leading or had_trailing:
        new_val = f"%{resolved}%"
    logger.info("SQL %s ILIKE fix: %r -> %r", field_name, val, new_val)
    return f"{prefix}{quote}{new_val}{quote}"


_EQ_STATE = re.compile(
    r"(?P<prefix>\bstate\s*=\s*)(?P<quote>['\"])(?P<val>[^'\"]*)(?P=quote)",
    re.IGNORECASE,
)
_EQ_COURSE = re.compile(
    r"(?P<prefix>\bcourse\s*=\s*)(?P<quote>['\"])(?P<val>[^'\"]*)(?P=quote)",
    re.IGNORECASE,
)


def _fix_eq_match(m: re.Match[str], resolver, field_name: str) -> str:
    prefix = m.group("prefix")
    quote = m.group("quote")
    val = m.group("val")
    resolved = resolver(val.strip())
    if resolved is None:
        return m.group(0)
    logger.info("SQL %s = fix: %r -> %r", field_name, val, resolved)
    return f"{prefix}{quote}{resolved}{quote}"


def fix_sql_state_and_course(sql: str) -> str:
    """Rewrite state/course ILIKE (and =) literals to match DB canonical values."""
    out = _ILIKE_STATE.sub(
        lambda m: _fix_ilike_match(m, resolve_state, "state"),
        sql,
    )
    out = _ILIKE_COURSE.sub(
        lambda m: _fix_ilike_match(m, resolve_course, "course"),
        out,
    )
    out = _EQ_STATE.sub(
        lambda m: _fix_eq_match(m, resolve_state, "state"),
        out,
    )
    out = _EQ_COURSE.sub(
        lambda m: _fix_eq_match(m, resolve_course, "course"),
        out,
    )
    return out

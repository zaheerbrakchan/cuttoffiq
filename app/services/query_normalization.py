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
        "GOA",
        "GUJARAT",
        "HARYANA",
        "HIMACHAL",
        "J&K",
        "JHARKHAND",
        "KARNATAKA",
        "KERALA",
        "MADHYA PRADESH",
        "MAHARASHTRA",
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

# Exact values as stored in the database (college_type column).
CANONICAL_COLLEGE_TYPES: frozenset[str] = frozenset(
    {
        "AIIMS",
        "AMU",
        "BHU",
        "DEEMED",
        "GOVERNMENT",
        "JAMIA MILLIA",
        "JIPMER",
        "Private",
        "PRIVATE",
    }
)

# Lowercase alias -> canonical college_type
COLLEGE_TYPE_ALIASES: dict[str, str] = {
    "government": "GOVERNMENT",
    "govt": "GOVERNMENT",
    "gov": "GOVERNMENT",
    "government college": "GOVERNMENT",
    "government medical college": "GOVERNMENT",
    "gmc": "GOVERNMENT",
    "state government": "GOVERNMENT",
    "private": "Private",
    "pvt": "Private",
    "private college": "Private",
    "self-financed": "Private",
    "self financed": "Private",
    "deemed": "DEEMED",
    "deemed university": "DEEMED",
    "deemed to be university": "DEEMED",
    "central": "GOVERNMENT",
    "central government": "GOVERNMENT",
    "central university": "GOVERNMENT",
    "aiims": "AIIMS",
    "all india institute": "AIIMS",
    "jipmer": "JIPMER",
    "amu": "AMU",
    "aligarh muslim university": "AMU",
    "aligarh": "AMU",
    "bhu": "BHU",
    "banaras hindu university": "BHU",
    "banaras": "BHU",
    "jamia": "JAMIA MILLIA",
    "jamia millia": "JAMIA MILLIA",
    "jamia millia islamia": "JAMIA MILLIA",
}

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
    "tamilnad": "TAMILNADU",
    "madhya pradesh": "MADHYA PRADESH",
    "mp": "MADHYA PRADESH",
    "m p": "MADHYA PRADESH",
    "uttar pradesh": "UTTAR PRADESH",
    "up": "UTTAR PRADESH",
    "u p": "UTTAR PRADESH",
    "west bengal": "WEST BENGAL",
    "wb": "WEST BENGAL",
    "w b": "WEST BENGAL",
    "bengal": "WEST BENGAL",
    "arunachal pradesh": "ARUNACHAL",
    "arunachal": "ARUNACHAL",
    "himachal pradesh": "HIMACHAL",
    "himachal": "HIMACHAL",
    "hp": "HIMACHAL",
    "h p": "HIMACHAL",
    "andhra pradesh": "ANDHRA",
    "andhra": "ANDHRA",
    "andra": "ANDHRA",
    "ap": "ANDHRA",
    "a p": "ANDHRA",
    "uttarakhand": "UTTARAKHAND",
    "uttrakhand": "UTTARAKHAND",
    "uk": "UTTARAKHAND",
    "u k": "UTTARAKHAND",
    "chhattisgarh": "CHHATTISGARH",
    "chattisgarh": "CHHATTISGARH",
    "chhatisgarh": "CHHATTISGARH",
    "cg": "CHHATTISGARH",
    "c g": "CHHATTISGARH",
    "orissa": "ODISHA",
    "odisha": "ODISHA",
    "telangana": "TELANGANA",
    "telengana": "TELANGANA",
    "karnataka": "KARNATAKA",
    "karnatak": "KARNATAKA",
    "banglore": "KARNATAKA",
    "bangalore": "KARNATAKA",
    "rajasthan": "RAJASTHAN",
    "rajsthan": "RAJASTHAN",
    "punjab": "PUNJAB",
    "panjab": "PUNJAB",
    "bihar": "BIHAR",
    "kerala": "KERALA",
    "kerela": "KERALA",
    "kerla": "KERALA",
    "keral": "KERALA",
    "gujarat": "GUJARAT",
    "gujrat": "GUJARAT",
    "gujrath": "GUJARAT",
    "gujurat": "GUJARAT",
    "haryana": "HARYANA",
    "hariyana": "HARYANA",
    "jharkhand": "JHARKHAND",
    "jharkand": "JHARKHAND",
    "jarkhand": "JHARKHAND",
    "nagaland": "NAGALAND",
    "delhi": "DELHI",
    "new delhi": "DELHI",
    "maharashtra": "MAHARASHTRA",
    "maharastra": "MAHARASHTRA",
    "maharashtr": "MAHARASHTRA",
    "mh": "MAHARASHTRA",
    "mumbai": "MAHARASHTRA",
    "pune": "MAHARASHTRA",
    "goa": "GOA",
    "tn": "TAMILNADU",
    "t n": "TAMILNADU",
    # All India / AIQ rounds are stored with state = MCC in this dataset.
    "all india quota": "MCC",
    "all india": "MCC",
    "aiq": "MCC",
    "all india counselling": "MCC",
    "mcc counselling": "MCC",
    "medical counselling committee": "MCC",
    "mcc": "MCC",
}

COURSE_ALIASES: dict[str, str] = {
    "mbbs": "MBBS",
    "bds": "BDS",
    "b.sc. nursing": "B.Sc. Nursing",
    "bsc nursing": "B.Sc. Nursing",
    "b.sc nursing": "B.Sc. Nursing",
    "bachelor of science nursing": "B.Sc. Nursing",
}

# Exact values as stored in the database (category column).
# Note: Categories vary by state, these are common ones
CANONICAL_CATEGORIES: frozenset[str] = frozenset(
    {
        "GENERAL",
        "OBC",
        "SC",
        "ST",
        "EWS",
        "OBC-A",
        "OBC-B",
        "OBC-NCL",
        "VJ",
        "NT-B",
        "NT-C",
        "NT-D",
        "SBC",
        "SEBC",
        "MBC",
        "BCM",
        "BC",
    }
)

# Category aliases - map common variations to canonical forms
CATEGORY_ALIASES: dict[str, str] = {
    "general": "GENERAL",
    "gen": "GENERAL",
    "ur": "GENERAL",
    "unreserved": "GENERAL",
    "open": "GENERAL",
    "obc": "OBC",
    "other backward class": "OBC",
    "other backward classes": "OBC",
    "obc-ncl": "OBC-NCL",
    "obc ncl": "OBC-NCL",
    "obc non creamy layer": "OBC-NCL",
    "sc": "SC",
    "scheduled caste": "SC",
    "schedule caste": "SC",
    "st": "ST",
    "scheduled tribe": "ST",
    "schedule tribe": "ST",
    "ews": "EWS",
    "economically weaker section": "EWS",
    "economically weaker": "EWS",
    "bc": "BC",
    "backward class": "BC",
    "mbc": "MBC",
    "most backward class": "MBC",
    "sebc": "SEBC",
    "sbc": "SBC",
    "bcm": "BCM",
    "vj": "VJ",
    "nt-b": "NT-B",
    "nt-c": "NT-C",
    "nt-d": "NT-D",
    "ntb": "NT-B",
    "ntc": "NT-C",
    "ntd": "NT-D",
}

# Domicile values in database
CANONICAL_DOMICILE: frozenset[str] = frozenset(
    {
        "DOMICILE",
        "NON-DOMICILE",
        "OPEN",
    }
)

DOMICILE_ALIASES: dict[str, str] = {
    "domicile": "DOMICILE",
    "home state": "DOMICILE",
    "home-state": "DOMICILE",
    "resident": "DOMICILE",
    "state quota": "DOMICILE",
    "non-domicile": "NON-DOMICILE",
    "non domicile": "NON-DOMICILE",
    "nondomicile": "NON-DOMICILE",
    "outside state": "NON-DOMICILE",
    "other state": "NON-DOMICILE",
    "all india": "NON-DOMICILE",
    "open": "OPEN",
    "all": "OPEN",
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


def resolve_college_type(text: str) -> str | None:
    """Return canonical college_type if text matches an alias or exact canonical."""
    key = _normalize_key(text)
    if key in COLLEGE_TYPE_ALIASES:
        return COLLEGE_TYPE_ALIASES[key]
    # Check exact match (case-insensitive)
    for canon in CANONICAL_COLLEGE_TYPES:
        if canon.lower() == key:
            return canon
    return None


def resolve_category(text: str) -> str | None:
    """Return canonical category if text matches an alias or exact canonical."""
    key = _normalize_key(text)
    if key in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[key]
    # Check exact match (case-insensitive)
    upper = text.strip().upper()
    if upper in CANONICAL_CATEGORIES:
        return upper
    # Partial match for variations like OBC-A, OBC-B
    for canon in CANONICAL_CATEGORIES:
        if canon.lower() == key or key == canon.lower():
            return canon
    return None


def resolve_domicile(text: str) -> str | None:
    """Return canonical domicile if text matches an alias or exact canonical."""
    key = _normalize_key(text)
    if key in DOMICILE_ALIASES:
        return DOMICILE_ALIASES[key]
    upper = text.strip().upper()
    if upper in CANONICAL_DOMICILE:
        return upper
    return None


def normalize_user_question(question: str) -> str:
    """
    Replace common state/course phrases in the user question so the model
    tends to emit correct ILIKE literals.
    
    Only replaces WHOLE WORDS to avoid corrupting text like "applicable" -> "ANDHRAplicable"
    """
    q = question
    # States: longest phrases first, use word boundaries to avoid partial replacements
    phrase_order = sorted(STATE_ALIASES.keys(), key=len, reverse=True)
    for phrase in phrase_order:
        canon = STATE_ALIASES[phrase]
        # Use word boundaries to only match whole words/phrases
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
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

_ILIKE_COLLEGE_TYPE = re.compile(
    r"(?P<prefix>\bcollege_type\s+ILIKE\s+)(?P<quote>['\"])(?P<val>[^'\"]*)(?P=quote)",
    re.IGNORECASE,
)
_EQ_COLLEGE_TYPE = re.compile(
    r"(?P<prefix>\bcollege_type\s*=\s*)(?P<quote>['\"])(?P<val>[^'\"]*)(?P=quote)",
    re.IGNORECASE,
)

_ILIKE_CATEGORY = re.compile(
    r"(?P<prefix>\bcategory\s+ILIKE\s+)(?P<quote>['\"])(?P<val>[^'\"]*)(?P=quote)",
    re.IGNORECASE,
)
_EQ_CATEGORY = re.compile(
    r"(?P<prefix>\bcategory\s*=\s*)(?P<quote>['\"])(?P<val>[^'\"]*)(?P=quote)",
    re.IGNORECASE,
)

_ILIKE_DOMICILE = re.compile(
    r"(?P<prefix>\bdomicile\s+ILIKE\s+)(?P<quote>['\"])(?P<val>[^'\"]*)(?P=quote)",
    re.IGNORECASE,
)
_EQ_DOMICILE = re.compile(
    r"(?P<prefix>\bdomicile\s*=\s*)(?P<quote>['\"])(?P<val>[^'\"]*)(?P=quote)",
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
    """Rewrite all column literals (state, course, college_type, category, domicile) to match DB canonical values."""
    out = sql
    
    # State
    out = _ILIKE_STATE.sub(
        lambda m: _fix_ilike_match(m, resolve_state, "state"),
        out,
    )
    out = _EQ_STATE.sub(
        lambda m: _fix_eq_match(m, resolve_state, "state"),
        out,
    )
    
    # Course
    out = _ILIKE_COURSE.sub(
        lambda m: _fix_ilike_match(m, resolve_course, "course"),
        out,
    )
    out = _EQ_COURSE.sub(
        lambda m: _fix_eq_match(m, resolve_course, "course"),
        out,
    )
    
    # College type
    out = _ILIKE_COLLEGE_TYPE.sub(
        lambda m: _fix_ilike_match(m, resolve_college_type, "college_type"),
        out,
    )
    out = _EQ_COLLEGE_TYPE.sub(
        lambda m: _fix_eq_match(m, resolve_college_type, "college_type"),
        out,
    )
    
    # Category
    out = _ILIKE_CATEGORY.sub(
        lambda m: _fix_ilike_match(m, resolve_category, "category"),
        out,
    )
    out = _EQ_CATEGORY.sub(
        lambda m: _fix_eq_match(m, resolve_category, "category"),
        out,
    )
    
    # Domicile
    out = _ILIKE_DOMICILE.sub(
        lambda m: _fix_ilike_match(m, resolve_domicile, "domicile"),
        out,
    )
    out = _EQ_DOMICILE.sub(
        lambda m: _fix_eq_match(m, resolve_domicile, "domicile"),
        out,
    )
    
    return out

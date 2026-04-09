"""
Onboarding service for first-time users.

Collects:
1. NEET score / expected rank
2. Course preference (MBBS India / BDS India / MBBS Abroad)
3. Home/domicile state
4. Category (based on home state)
5. Sub-category (based on home state)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

from app.services.query_normalization import (
    CANONICAL_STATES,
    resolve_state,
    resolve_state_from_message,
)

logger = logging.getLogger("neet_assistant.onboarding")

# Onboarding steps in order
ONBOARDING_STEPS = [
    "intro",  # Welcome + confirmation to proceed
    "neet_score",
    "course",
    "home_state",
    "category",
    "sub_category",
    "college_type",  # Multi-select: Government, Private, AIIMS, etc.
]

# College type options (multi-select)
COLLEGE_TYPE_OPTIONS = [
    {"value": "GOVERNMENT", "label": "Government"},
    {"value": "Private", "label": "Private"},
    {"value": "DEEMED", "label": "Deemed"},
    {"value": "AIIMS", "label": "AIIMS"},
    {"value": "JIPMER", "label": "JIPMER"},
    {"value": "BHU", "label": "BHU"},
    {"value": "AMU", "label": "AMU"},
    {"value": "JAMIA MILLIA", "label": "Jamia Millia"},
    {"value": "ALL", "label": "All types (no preference)"},
]

# Course options
COURSE_OPTIONS = [
    {"value": "MBBS_INDIA", "label": "MBBS in India"},
    {"value": "BDS_INDIA", "label": "BDS in India"},
    {"value": "MBBS_ABROAD", "label": "MBBS Abroad"},
]

# Onboarding "course" step values only (not NEET course column MBBS/BDS).
COURSE_PREFERENCE_VALUES = frozenset(opt["value"] for opt in COURSE_OPTIONS)


def normalize_misplaced_course_category(preferences: dict | None) -> dict:
    """
    Fix LLM mistakes: MBBS_INDIA / BDS_INDIA / MBBS_ABROAD must live under `course`,
    not reservation `category`.

    Also coerces `college_type` from a bare string into a one-element list so callers
    never treat it like an iterable of characters.
    """
    prefs = dict(preferences or {})
    raw = prefs.get("category")
    if not isinstance(raw, str):
        _coerce_college_type_to_list(prefs)
        return prefs
    canonical = None
    stripped = raw.strip()
    if stripped in COURSE_PREFERENCE_VALUES:
        canonical = stripped
    else:
        u = stripped.upper()
        for v in COURSE_PREFERENCE_VALUES:
            if v.upper() == u:
                canonical = v
                break
    if not canonical:
        _coerce_college_type_to_list(prefs)
        return prefs
    if not prefs.get("course"):
        prefs["course"] = canonical
    prefs.pop("category", None)
    logger.info("Normalized misplaced course preference from category to course=%s", canonical)
    _coerce_college_type_to_list(prefs)
    return prefs


def _coerce_college_type_to_list(prefs: dict) -> None:
    """LLM/onboarding sometimes store college_type as a string; keep a list for multi-select."""
    ct = prefs.get("college_type")
    if isinstance(ct, str):
        s = ct.strip()
        prefs["college_type"] = [s] if s else []
    elif ct is not None and not isinstance(ct, list):
        prefs["college_type"] = []


# State options for display (user-friendly names)
STATE_OPTIONS = [
    {"value": "ANDHRA", "label": "Andhra Pradesh"},
    {"value": "ARUNACHAL", "label": "Arunachal Pradesh"},
    {"value": "BIHAR", "label": "Bihar"},
    {"value": "CHHATTISGARH", "label": "Chhattisgarh"},
    {"value": "DELHI", "label": "Delhi"},
    {"value": "GOA", "label": "Goa"},
    {"value": "GUJARAT", "label": "Gujarat"},
    {"value": "HARYANA", "label": "Haryana"},
    {"value": "HIMACHAL", "label": "Himachal Pradesh"},
    {"value": "J&K", "label": "Jammu & Kashmir"},
    {"value": "JHARKHAND", "label": "Jharkhand"},
    {"value": "KARNATAKA", "label": "Karnataka"},
    {"value": "KERALA", "label": "Kerala"},
    {"value": "MADHYA PRADESH", "label": "Madhya Pradesh"},
    {"value": "MAHARASHTRA", "label": "Maharashtra"},
    {"value": "NAGALAND", "label": "Nagaland"},
    {"value": "ODISHA", "label": "Odisha"},
    {"value": "PUNJAB", "label": "Punjab"},
    {"value": "RAJASTHAN", "label": "Rajasthan"},
    {"value": "TAMILNADU", "label": "Tamil Nadu"},
    {"value": "TELANGANA", "label": "Telangana"},
    {"value": "UTTAR PRADESH", "label": "Uttar Pradesh"},
    {"value": "UTTARAKHAND", "label": "Uttarakhand"},
    {"value": "WEST BENGAL", "label": "West Bengal"},
]

# Fallback categories (used only if database query fails)
FALLBACK_CATEGORIES = [
    {"value": "GENERAL", "label": "General"},
    {"value": "OBC", "label": "OBC"},
    {"value": "SC", "label": "SC"},
    {"value": "ST", "label": "ST"},
    {"value": "EWS", "label": "EWS"},
]

# Fallback sub-categories (used only if database query fails)
FALLBACK_SUB_CATEGORIES = [
    {"value": "NONE", "label": "None / Not Applicable"},
]


def db_categories_to_options(categories: list[str]) -> list[dict]:
    """Convert database category strings to option dicts."""
    if not categories:
        return FALLBACK_CATEGORIES.copy()
    return [{"value": cat, "label": cat} for cat in categories]


def db_sub_categories_to_options(sub_categories: list[str]) -> list[dict]:
    """Convert database sub-category strings to option dicts."""
    # Always include "None" as first option
    options = [{"value": "NONE", "label": "None / Not Applicable"}]
    if sub_categories:
        for sub in sub_categories:
            if sub and sub.strip():
                options.append({"value": sub, "label": sub})
    return options


def db_college_types_to_options(college_types: list[str]) -> list[dict]:
    """Convert database college_type strings to option dicts."""
    if not college_types:
        return COLLEGE_TYPE_OPTIONS.copy()
    # Keep original DB values but title-case label for readability.
    return [{"value": ct, "label": ct.title()} for ct in college_types if ct and ct.strip()]


def get_step_confirmation(step: str, preferences: dict) -> str:
    """Generate a friendly confirmation for the just-completed step."""
    if step == "intro":
        return ""  # No confirmation needed for intro, just proceed
    
    elif step == "neet_score" or step == "neet_score_clarify":
        score_info = preferences.get("neet_score", {})
        # Don't show confirmation if score is not yet confirmed (pending clarification)
        if not score_info or not score_info.get("type") or not score_info.get("value"):
            return ""
        if score_info.get("type") == "score":
            return f"📊 {score_info.get('value')} marks"
        else:
            return f"📊 AIR {score_info.get('value')}"
    
    elif step == "course":
        course = preferences.get("course", "").replace("_", " ")
        return f"📚 {course}"
    
    elif step == "home_state":
        state = preferences.get("home_state", "")
        for opt in STATE_OPTIONS:
            if opt["value"] == state:
                return f"🏠 {opt['label']}"
        return f"🏠 {state}"
    
    elif step == "category":
        return f"👤 {preferences.get('category', '')}"
    
    elif step == "sub_category":
        sub = preferences.get("sub_category", "")
        if sub == "NONE":
            return "✓ No special quota"
        return f"✓ {sub}"
    
    elif step == "college_type":
        college_types = preferences.get("college_type", [])
        if isinstance(college_types, list):
            if "ALL" in college_types or not college_types:
                return "🏫 All college types"
            return f"🏫 {', '.join(college_types)}"
        return f"🏫 {college_types}"
    
    return "✓"


@dataclass
class OnboardingStatus:
    is_complete: bool
    current_step: str | None
    next_question: str | None
    options: list[dict] | None
    preferences: dict


def _format_options_text(options: list[dict]) -> str:
    """Format options for display in chat message."""
    lines = []
    for i, opt in enumerate(options, 1):
        lines.append(f"  {i}) {opt['label']}")
    return "\n".join(lines)


def _get_welcome_message() -> str:
    return (
        "Hello! 👋 I'm Anuj, your AI NEET Counselling Assistant.\n\n"
        "I can help you with:\n"
        "• Predict colleges you can get based on your NEET score/rank\n"
        "• Compare options across different states and categories\n"
        "• Explore cutoff trends from 2025 counselling data\n"
        "• Answer questions about counselling process, domicile rules, etc.\n\n"
        "Would you like me to help you find colleges based on your profile?"
    )


def get_onboarding_question(
    step: str,
    preferences: dict,
    *,
    db_categories: list[dict] | None = None,
    db_sub_categories: list[dict] | None = None,
    db_college_types: list[dict] | None = None,
) -> tuple[str, list[dict] | None]:
    """Get the question and options for a given onboarding step."""
    if step == "intro":
        return (
            _get_welcome_message(),
            None,
        )
    
    elif step == "neet_score":
        return (
            "Great! Let me collect a few quick details.\n\n"
            "What's your NEET score (e.g., 540 marks) or expected rank (e.g., AIR 15000)?",
            None,
        )
    
    elif step == "course":
        options = COURSE_OPTIONS
        return (
            "Which course are you interested in?\n\n" + _format_options_text(options),
            options,
        )
    
    elif step == "home_state":
        options = STATE_OPTIONS
        return (
            "Which is your home/domicile state?",
            options,
        )
    
    elif step == "category":
        # Use dynamic categories from database if provided
        options = db_categories if db_categories else FALLBACK_CATEGORIES
        return (
            "What's your reservation category?\n\n" + _format_options_text(options),
            options,
        )
    
    elif step == "sub_category":
        # Use dynamic sub-categories from database if provided
        options = db_sub_categories if db_sub_categories else FALLBACK_SUB_CATEGORIES
        return (
            "Please select your sub-category (or choose None):\n\n" + _format_options_text(options),
            options,
        )
    
    elif step == "college_type":
        options = db_college_types if db_college_types else COLLEGE_TYPE_OPTIONS
        return (
            "What type of colleges are you interested in?\n\n"
            "You can select multiple options:\n\n"
            + _format_options_text(options),
            options,
        )
    
    return ("", None)


def check_onboarding_status(
    preferences: dict,
    *,
    db_categories: list[dict] | None = None,
    db_sub_categories: list[dict] | None = None,
    db_college_types: list[dict] | None = None,
) -> OnboardingStatus:
    """
    Check if onboarding is complete and determine the current step.
    """
    if not preferences:
        preferences = {}
    
    # Check if we're waiting for score/rank clarification
    if preferences.get("_pending_neet_value") and not preferences.get("neet_score"):
        pending_value = preferences["_pending_neet_value"]
        question = (
            f"Just to confirm - is {pending_value} your NEET score (out of 720) "
            f"or your All India Rank?\n\n"
            f"1. Score (marks out of 720)\n"
            f"2. Rank (All India Rank)"
        )
        return OnboardingStatus(
            is_complete=False,
            current_step="neet_score_clarify",
            next_question=question,
            options=[
                {"value": "score", "label": "Score (marks out of 720)"},
                {"value": "rank", "label": "Rank (All India Rank)"},
            ],
            preferences=preferences,
        )
    
    # Check which steps are completed
    for step in ONBOARDING_STEPS:
        if step not in preferences or preferences[step] is None:
            question, options = get_onboarding_question(
                step, preferences,
                db_categories=db_categories,
                db_sub_categories=db_sub_categories,
                db_college_types=db_college_types,
            )
            
            return OnboardingStatus(
                is_complete=False,
                current_step=step,
                next_question=question,
                options=options,
                preferences=preferences,
            )
    
    return OnboardingStatus(
        is_complete=True,
        current_step=None,
        next_question=None,
        options=None,
        preferences=preferences,
    )


def _parse_score_or_rank(text: str) -> dict | None:
    """
    Parse NEET score or rank from user input. Think like a human:
    - "540 marks" or "scored 540" → clearly score
    - "670 rank" or "AIR 12000" → clearly rank
    - Number > 720 → obviously rank (NEET max is 720)
    - Just "540" alone → ambiguous, needs clarification
    
    Returns:
        {"type": "score"|"rank", "value": int} - if clearly identified
        {"type": "needs_clarification", "value": int} - if ambiguous (plain number ≤720)
        None - if no number found
    """
    original = text.strip()
    text_lower = original.lower()
    
    # Check for explicit RANK patterns first (keywords like rank, AIR, all india)
    rank_patterns = [
        r"(?:rank|air|all\s*india\s*rank|neet\s*rank)[:\s-]*(\d{1,7})",
        r"(\d{1,7})\s*(?:rank|air|all\s*india)",
        r"air\s*[:\s-]*(\d{1,7})",
        r"rank\s*[:\s-]*(\d{1,7})",
    ]
    for pattern in rank_patterns:
        m = re.search(pattern, text_lower, re.IGNORECASE)
        if m:
            rank = int(m.group(1))
            if rank >= 1:
                return {"type": "rank", "value": rank}
    
    # Check for explicit SCORE patterns (keywords like marks, score, scored)
    score_patterns = [
        r"(?:neet\s*)?(?:score|marks?|scored)[:\s-]*(\d{2,3})",
        r"(\d{2,3})\s*(?:marks?|score|out\s*of|/\s*720)",
        r"scoring\s*(\d{2,3})",
        r"(?:i\s+)?got\s+(\d{2,3})",
    ]
    for pattern in score_patterns:
        m = re.search(pattern, text_lower, re.IGNORECASE)
        if m:
            score = int(m.group(1))
            if 1 <= score <= 720:
                return {"type": "score", "value": score}
    
    # Extract any standalone number
    numbers = re.findall(r"\b\d+\b", text_lower)
    if numbers:
        num = int(numbers[0])
        
        # Number > 720 is OBVIOUSLY a rank (NEET max score is 720)
        if num > 720:
            return {"type": "rank", "value": num}
        
        # Number <= 720 without explicit context - AMBIGUOUS, ask for clarification
        # Could be a score of 540 or a rank of 540 (top ranker!)
        if 1 <= num <= 720:
            return {"type": "needs_clarification", "value": num}
    
    return None


def _match_option(text: str, options: list[dict]) -> dict | None:
    """Match user input to an option."""
    text = text.strip().lower()
    
    # Check for number selection (1, 2, 3, etc.)
    if text.isdigit():
        idx = int(text) - 1
        if 0 <= idx < len(options):
            return options[idx]
    
    # Check for exact value match
    for opt in options:
        if text == opt["value"].lower():
            return opt
        if text == opt["label"].lower():
            return opt
    
    # Check for partial match (guarded to avoid tiny-token false positives like "st" from "state")
    for opt in options:
        opt_value = opt["value"].lower()
        opt_label = opt["label"].lower()
        if len(text) >= 3 and (text in opt_value or text in opt_label):
            return opt
        if len(opt_value) >= 3 and opt_value in text:
            return opt
        if len(opt_label) >= 3 and opt_label in text:
            return opt
    
    return None


def process_onboarding_response(
    user_input: str,
    current_step: str,
    preferences: dict,
    *,
    db_categories: list[dict] | None = None,
    db_sub_categories: list[dict] | None = None,
    db_college_types: list[dict] | None = None,
) -> tuple[dict, str | None]:
    """
    Process user response for current onboarding step.
    
    Returns:
        (updated_preferences, error_message or None)
    """
    user_input = user_input.strip()
    updated_prefs = preferences.copy() if preferences else {}

    # Global correction handling: user may correct home state at any later step.
    # Example: "sorry my state was UP"
    lower_input = user_input.lower()
    state_correction_hint = (
        "home state" in lower_input
        or "domicile" in lower_input
        or re.search(r"\bstate\b", lower_input) is not None
    )
    if state_correction_hint:
        corrected_state = resolve_state_from_message(user_input)
        if corrected_state and corrected_state != updated_prefs.get("home_state"):
            updated_prefs["home_state"] = corrected_state
            # Dependent fields must be recollected for the new state.
            updated_prefs.pop("category", None)
            updated_prefs.pop("sub_category", None)
            updated_prefs["_correction_note"] = (
                f"Got it, I've updated the home state to {corrected_state}. Let's continue."
            )
            return updated_prefs, None
    
    if current_step == "intro":
        # User confirms they want to proceed with college prediction
        lower = user_input.lower()
        
        # Check for positive response
        positive_patterns = [
            "yes", "yeah", "yep", "sure", "ok", "okay", "proceed",
            "help", "find", "predict", "college", "start", "go",
            "let's", "show", "tell", "please", "hi", "hello",
        ]
        
        if any(pattern in lower for pattern in positive_patterns) or len(lower) < 30:
            # Any short response or positive signal = proceed
            updated_prefs["intro"] = "confirmed"
            return updated_prefs, None
        
        # Even longer messages - user probably wants help
        updated_prefs["intro"] = "confirmed"
        return updated_prefs, None
    
    elif current_step == "neet_score":
        parsed = _parse_score_or_rank(user_input)
        if not parsed:
            return updated_prefs, (
                "I couldn't understand that. Please enter your NEET score (e.g., 550 marks) "
                "or rank (e.g., AIR 15000)."
            )
        
        # Check if we need clarification (ambiguous number ≤720 without context)
        if parsed.get("type") == "needs_clarification":
            # Store the value temporarily and ask for clarification
            updated_prefs["_pending_neet_value"] = parsed["value"]
            return updated_prefs, None  # No error - will trigger clarification step
        
        updated_prefs["neet_score"] = parsed
        return updated_prefs, None
    
    elif current_step == "neet_score_clarify":
        # User is clarifying whether the pending value is score or rank
        pending_value = preferences.get("_pending_neet_value")
        if not pending_value:
            # No pending value, restart score collection
            return updated_prefs, "Let's start again. What's your NEET score or rank?"
        
        lower = user_input.lower()
        
        # Check if user said it's a score
        if any(word in lower for word in ["score", "marks", "1", "first"]):
            updated_prefs["neet_score"] = {"type": "score", "value": pending_value}
            updated_prefs.pop("_pending_neet_value", None)
            return updated_prefs, None
        
        # Check if user said it's a rank
        if any(word in lower for word in ["rank", "air", "2", "second"]):
            updated_prefs["neet_score"] = {"type": "rank", "value": pending_value}
            updated_prefs.pop("_pending_neet_value", None)
            return updated_prefs, None
        
        # Didn't understand - ask again
        return updated_prefs, (
            f"Just to confirm - is {pending_value} your NEET score (out of 720) "
            f"or your All India Rank?\n\n"
            f"Reply: score or rank"
        )
    
    elif current_step == "course":
        options = COURSE_OPTIONS
        matched = _match_option(user_input, options)
        if not matched:
            # Try fuzzy matching
            lower = user_input.lower()
            if "mbbs" in lower and "abroad" in lower:
                matched = {"value": "MBBS_ABROAD", "label": "MBBS Abroad"}
            elif "bds" in lower:
                matched = {"value": "BDS_INDIA", "label": "BDS in India"}
            elif "mbbs" in lower:
                matched = {"value": "MBBS_INDIA", "label": "MBBS in India"}
        
        if not matched:
            return updated_prefs, (
                "Please select a valid course option:\n"
                "1. MBBS in India\n"
                "2. BDS in India\n"
                "3. MBBS Abroad"
            )
        updated_prefs["course"] = matched["value"]
        return updated_prefs, None
    
    elif current_step == "home_state":
        # Try to resolve state
        resolved = resolve_state(user_input)
        if resolved:
            updated_prefs["home_state"] = resolved
            return updated_prefs, None
        
        # Try matching from options
        matched = _match_option(user_input, STATE_OPTIONS)
        if matched:
            updated_prefs["home_state"] = matched["value"]
            return updated_prefs, None
        
        return updated_prefs, (
            "I couldn't recognize that state. Please enter a valid Indian state name "
            "(e.g., Karnataka, Tamil Nadu, Delhi, etc.)"
        )
    
    elif current_step == "category":
        # Use dynamic categories from database if provided
        options = db_categories if db_categories else FALLBACK_CATEGORIES
        matched = _match_option(user_input, options)
        
        if not matched:
            # Try matching input directly (user might type the exact category)
            lower = user_input.strip().upper()
            for opt in options:
                if lower == opt["value"].upper() or lower in opt["value"].upper():
                    matched = opt
                    break
        
        if not matched:
            return updated_prefs, (
                "Please select a valid category from the options above."
            )
        updated_prefs["category"] = matched["value"]
        return updated_prefs, None
    
    elif current_step == "sub_category":
        # Use dynamic sub-categories from database if provided
        options = db_sub_categories if db_sub_categories else FALLBACK_SUB_CATEGORIES
        matched = _match_option(user_input, options)
        
        if not matched:
            lower = user_input.lower()
            if "none" in lower or "no" in lower or "na" in lower or "not" in lower or lower == "1":
                matched = {"value": "NONE", "label": "None / Not Applicable"}
            else:
                # Try matching input directly
                upper = user_input.strip().upper()
                for opt in options:
                    if upper == opt["value"].upper() or upper in opt["value"].upper():
                        matched = opt
                        break
        
        if not matched:
            return updated_prefs, (
                "Please select a valid sub-category from the options, or say 'None' if not applicable."
            )
        updated_prefs["sub_category"] = matched["value"]
        return updated_prefs, None
    
    elif current_step == "college_type":
        # Multi-select: user can select multiple options like "1, 2, 3" or "Government, Private"
        selected_types = []
        user_input_lower = user_input.lower().strip()
        
        # Check for "all" or similar
        if user_input_lower in ["all", "any", "all types", "no preference", "9"]:
            selected_types = ["ALL"]
        else:
            # Split by comma, space, or "and"
            parts = re.split(r'[,\s]+|(?:\s+and\s+)', user_input)
            parts = [p.strip() for p in parts if p.strip()]
            
            options = db_college_types if db_college_types else COLLEGE_TYPE_OPTIONS
            for part in parts:
                matched = _match_option(part, options)
                if matched and matched["value"] not in selected_types:
                    selected_types.append(matched["value"])
                else:
                    # Try direct match
                    part_upper = part.upper()
                    for opt in options:
                        if (part_upper == opt["value"].upper() or 
                            part_upper in opt["value"].upper() or
                            opt["value"].upper() in part_upper):
                            if opt["value"] not in selected_types:
                                selected_types.append(opt["value"])
                            break
        
        if not selected_types:
            return updated_prefs, (
                "Please select at least one college type. You can choose multiple options "
                "(e.g., 1, 2 or Government, Private) or say 'All' for no preference."
            )
        
        updated_prefs["college_type"] = selected_types
        return updated_prefs, None
    
    return updated_prefs, "Unknown onboarding step."


def get_onboarding_complete_message(preferences: dict) -> str:
    """Generate a professional message after onboarding with immediate offer to help."""
    score_info = preferences.get("neet_score", {})
    if score_info.get("type") == "score":
        score_text = f"{score_info.get('value')} marks"
    else:
        score_text = f"AIR {score_info.get('value')}"
    
    course = preferences.get("course", "").replace("_", "/")
    home_state = preferences.get("home_state", "")
    category = preferences.get("category", "")
    sub_category = preferences.get("sub_category", "")
    college_types = preferences.get("college_type", [])
    
    # Find state label
    state_label = home_state
    for opt in STATE_OPTIONS:
        if opt["value"] == home_state:
            state_label = opt["label"]
            break
    
    sub_cat_text = f" ({sub_category})" if sub_category and sub_category != "NONE" else ""
    
    # College type text
    if isinstance(college_types, list):
        if "ALL" in college_types or not college_types:
            college_type_text = "All types"
        else:
            college_type_text = ", ".join(college_types)
    else:
        college_type_text = college_types or "All types"
    
    return (
        f"Thank you! ✅ Your profile is saved:\n\n"
        f"📊 NEET: {score_text}\n"
        f"📚 Course: {course}\n"
        f"🏠 State: {state_label}\n"
        f"👤 Category: {category}{sub_cat_text}\n"
        f"🏫 College Type: {college_type_text}\n\n"
        f"Based on your profile, shall I find the best college options for you?\n\n"
        f"Reply 'Yes' to proceed, or ask me anything specific like:\n"
        f"• \"Show government colleges in my state\"\n"
        f"• \"What are my MCC/All India options?\"\n"
        f"• \"Compare private vs government colleges\""
    )


def format_preferences_for_context(preferences: dict) -> str:
    """Format user preferences as context string for LLM."""
    if not preferences:
        return ""
    
    parts = []
    
    score_info = preferences.get("neet_score", {})
    if score_info:
        if score_info.get("type") == "score":
            parts.append(f"NEET Score: {score_info.get('value')} marks")
        else:
            parts.append(f"NEET Rank: AIR {score_info.get('value')}")
    
    if preferences.get("course"):
        parts.append(f"Course: {preferences['course'].replace('_', ' ')}")
    
    if preferences.get("home_state"):
        parts.append(f"Home State: {preferences['home_state']}")
    
    if preferences.get("category"):
        parts.append(f"Category: {preferences['category']}")
    
    if preferences.get("sub_category") and preferences.get("sub_category") != "NONE":
        parts.append(f"Sub-category: {preferences['sub_category']}")
    
    college_types = preferences.get("college_type", [])
    if college_types:
        if isinstance(college_types, list):
            if "ALL" not in college_types:
                parts.append(f"Preferred College Types: {', '.join(college_types)}")
        else:
            parts.append(f"Preferred College Type: {college_types}")
    
    if not parts:
        return ""
    
    return "User Profile: " + ", ".join(parts)


def get_profile_confirmation_message(preferences: dict) -> str:
    """Generate a professional profile confirmation message."""
    score_info = preferences.get("neet_score", {})
    if score_info.get("type") == "score":
        score_text = f"{score_info.get('value')} marks"
    else:
        score_text = f"AIR {score_info.get('value')}"
    
    course = preferences.get("course", "").replace("_", "/")
    home_state = preferences.get("home_state", "")
    category = preferences.get("category", "")
    sub_category = preferences.get("sub_category", "")
    college_types = preferences.get("college_type", [])
    
    # Find state label
    state_label = home_state
    for opt in STATE_OPTIONS:
        if opt["value"] == home_state:
            state_label = opt["label"]
            break
    
    sub_cat_text = f" under {sub_category} quota" if sub_category and sub_category != "NONE" else ""
    
    # College type text
    if isinstance(college_types, list):
        if "ALL" in college_types or not college_types:
            college_type_text = "All types"
        else:
            college_type_text = ", ".join(college_types)
    else:
        college_type_text = college_types or "All types"
    
    return (
        f"Welcome back! 👋\n\n"
        f"I have your profile on file:\n"
        f"• NEET: {score_text}\n"
        f"• Course: {course}\n"
        f"• State: {state_label}\n"
        f"• Category: {category}{sub_cat_text}\n"
        f"• College Type: {college_type_text}\n\n"
        f"Would you like me to find the best college options based on this profile?\n\n"
        f"Reply 'Yes' to proceed, or tell me what specific information you need."
    )


def is_profile_confirmation_response(user_input: str) -> tuple[bool, str]:
    """
    Check if user's response confirms using their profile or wants something else.
    
    Returns:
        (is_confirmation, interpretation)
        - is_confirmation: True if user wants to use profile, False if they want something specific
        - interpretation: 'use_profile' | 'specific_query' | 'show_confirmation'
    """
    lower = user_input.strip().lower()
    
    # Check for queries that want to use their profile/state - treat as confirmation
    profile_query_patterns = [
        "my state", "in my state", "my home state",
        "show in my", "colleges in my", "options in my",
        "based on my", "for my profile", "using my profile",
    ]
    
    for pattern in profile_query_patterns:
        if pattern in lower:
            return True, "use_profile"
    
    # Check for clear "yes" / confirmation responses
    yes_patterns = [
        "yes", "yeah", "yep", "sure", "ok", "okay", "proceed",
        "find colleges", "show colleges", "go ahead", "that's right",
        "correct", "based on my profile", "use my profile", "my profile",
        "find me colleges", "show me colleges", "recommend", "suggest",
        "find options", "show options", "what can i get", "which colleges",
    ]
    
    for pattern in yes_patterns:
        if pattern in lower:
            return True, "use_profile"
    
    # Check for specific queries that mention different criteria
    # These suggest user wants to explore something OTHER than their profile
    specific_patterns = [
        r"colleges?\s+in\s+\w+",  # "colleges in Delhi"
        r"what\s+if\s+i\s+had",  # "what if I had 600 marks"
        r"with\s+\d+\s+marks",  # "with 550 marks"
        r"for\s+\d+\s+rank",  # "for 15000 rank"
        r"different\s+state",  # "different state"
        r"another\s+state",  # "another state"
        r"change\s+",  # "change my score"
        r"mcc",  # MCC / All India
        r"all india",  # All India
        r"private",  # private colleges
        r"deemed",  # deemed universities
    ]
    
    import re
    for pattern in specific_patterns:
        if re.search(pattern, lower):
            return False, "specific_query"
    
    # Greetings or vague messages - show confirmation first
    greetings_or_vague = [
        "hi", "hello", "hey", "hii", "hiii",
        "help me", "need help", "need your help",
    ]
    
    for pattern in greetings_or_vague:
        if lower == pattern or lower.startswith(pattern + " "):
            return False, "show_confirmation"
    
    # If message is very short (1-2 words) and vague, show confirmation
    if len(lower.split()) <= 2 and not any(word in lower for word in ["show", "find", "college", "option"]):
        return False, "show_confirmation"
    
    # Otherwise, treat as specific query and process directly
    return False, "specific_query"


def needs_profile_confirmation(preferences: dict, recent_chats: list) -> bool:
    """
    Check if we should ask for profile confirmation.
    
    Conditions:
    - Onboarding is complete (all preferences filled)
    - Profile hasn't been confirmed yet (profile_confirmed not True)
    - User hasn't started querying yet (recent_chats has only onboarding messages)
    """
    # Check if onboarding is complete
    required_keys = ["neet_score", "course", "home_state", "category", "sub_category", "college_type"]
    if not all(key in preferences and preferences[key] for key in required_keys):
        return False
    
    # Check if already confirmed
    if preferences.get("profile_confirmed"):
        return False
    
    return True

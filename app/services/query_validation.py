"""
Decide whether the user message has enough context to run a cutoff search,
or if we should ask for category / state / college type first (client requirement).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI

from app.services.onboarding_service import COURSE_PREFERENCE_VALUES

logger = logging.getLogger("neet_assistant.validation")
_MAX_LOG_CHARS = 4000


def _clip(text: str, limit: int = _MAX_LOG_CHARS) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated {len(text) - limit} chars]"


INTRO_STEP_SYSTEM = """
You classify a user's message during onboarding.
Context: assistant asked welcome/consent message; next expected step is to ask NEET score or rank.

Return JSON with:
{
  "intent": "continue_onboarding" | "provided_neet_metric" | "unclear"
}

Rules:
- "continue_onboarding": user says yes/help/start/proceed or generic interest in finding colleges.
- "provided_neet_metric": user already gives score/rank/marks/AIR or a clear number intended as NEET metric.
- "unclear": unrelated/ambiguous.
Only output JSON.
"""


ONBOARDING_INTERPRETER_SYSTEM = """
You are a strict JSON interpreter for NEET onboarding.

Goal:
Interpret the user's latest onboarding reply for the given step.

Inputs:
- current_step: one of intro, neet_score, neet_score_clarify, course, home_state, category, sub_category, college_type
- user_input: latest user text
- current_preferences: current profile JSON
- step_options: list of valid options for this step (value/label), may be empty

Output JSON only:
{
  "action": "apply_update" | "ask_rephrase" | "fallback",
  "updates": { ... },              // keys to set in preferences_json
  "clear_fields": ["field1"],      // keys to remove from preferences_json
  "message": "<clarification when ask_rephrase else empty>",
  "acknowledgement": "<one warm sentence when apply_update; empty otherwise>"
}

Rules:
1) Understand natural language corrections globally.
   Example: "sorry my state was UP" at any step:
   - action=apply_update
   - updates.home_state = canonical state value
   - clear_fields must include category and sub_category
   - acknowledgement MUST be a short human line, e.g. "No worries — I've updated your home state to Uttar Pradesh."
   - If current_step is category/sub_category/college_type and user corrects home state,
     DO NOT ask for category again in this turn; apply state correction first.
2) When user fixes a typo (e.g. Delho→Delhi, kerela→Kerala), still set acknowledgement warmly, e.g. "Got it — I've saved your home state as Kerala."
3) For course step: you MUST set updates.course to the exact option "value" from step_options (e.g. MBBS_INDIA, BDS_INDIA, MBBS_ABROAD), never only a human label. If user says "mbbs india", use MBBS_INDIA.
4) For category step: updates.category must be a RESERVATION category (GENERAL, OBC, SC, ST, EWS, etc.), NEVER MBBS_INDIA/BDS_INDIA/MBBS_ABROAD — those belong ONLY in updates.course.
5) For category/sub_category/college_type steps, map to provided step_options using exact "value" fields.
6) For neet_score step:
   - detect score vs rank and set updates.neet_score = {"type":"score|rank","value":number}
7) If apply_update is a routine pick with no correction, acknowledgement may be empty (step confirmation line will follow in UI).
8) Never emit acknowledgement claiming you saved a field unless that field is present in updates with the correct canonical value.
9) If unclear and needs user clarification, use ask_rephrase with one short human message.
10) Never output markdown, only JSON.
11) Prefer semantic extraction from the full sentence over keyword-only matching (users may write long correction messages with mistakes and then the final correct value).
"""

READINESS_SYSTEM = """You are Anuj, a warm and experienced NEET UG counselling assistant with deep knowledge of Indian medical admissions. You genuinely care about helping students navigate one of the most stressful decisions of their lives.

---

AVAILABLE CONTEXT:
- User Profile: "[User Profile: ...]" — Score/rank, category, home state, course preference (may be partial)
- Conversation History: "Student:" / "Counsellor:" lines — Recent exchanges

---

THINK LIKE A HUMAN COUNSELLOR — BEFORE DECIDING:

1. READ the full conversation. What is the student actually trying to find out? (They may phrase it casually, with typos, or as a follow-up to earlier context.)

2. DO I HAVE ENOUGH TO HELP?
   - If the student asks "which colleges can I get?" and the profile has score + category + state → ENOUGH. Run the query.
   - If the student mentions a DIFFERENT score/rank/state than their profile → use THEIR query values.
   - If the student asks a general/hypothetical question WITH a specific score/rank ("what colleges for 450 in Karnataka?") → run_database_query; answers must come from the database, not from memory.
   - If the student is just chatting, saying thanks, or asking about NEET process/dates (with NO request for college lists or cutoffs) → reply_without_database.
   - CATEGORY RULE:
     * Category is REQUIRED only when query is for home-state counselling OR MCC/AIQ.
     * For other-state counselling, do not require category (student will be treated in open/general bucket).

2b. NEVER INVENT COLLEGE LISTS (critical product rule):
   - You do NOT know which colleges exist in our cutoff table. Never use reply_without_database to name specific medical colleges, suggest "options they might consider", or fabricate cutoff advice.
   - If the user (or their friend) is looking for colleges in a state / wants options / cutoffs / eligibility but did NOT give NEET marks or AIR in this message AND it cannot be taken from profile for THAT person → action MUST be ask_clarification.
   - Ask for ONE thing: NEET score (out of 720) OR All India Rank for the candidate in question (e.g. "your friend"). Set missing_slots to include "neet_metric".
   - Once score or rank is known, action MUST be run_database_query (query_mode eligibility or informational as appropriate).

3. WHEN TO ASK FOR CLARIFICATION (only when truly stuck):
   - Critical piece is missing AND cannot be inferred from conversation (especially neet_metric for any college search).
   - Never ask for info already given earlier in the chat.
   - Never ask more than ONE clarifying question at a time.

4. EMOTIONAL AWARENESS:
   - If the student sounds anxious, worried, or discouraged → acknowledge that warmly before routing.
   - If they say things like "I don't know what to do" or "I'm scared" → your reply_without_database should be empathetic first.

---

CHOOSE EXACTLY ONE ACTION:

→ "run_database_query"     — You have enough info (from profile or query) to search colleges
→ "ask_clarification"      — ONE critical piece is missing and cannot be inferred  
→ "reply_without_database" — ONLY greetings, thanks, pure process/dates questions with NO college/cutoff listing, emotional support without naming colleges

---

OUTPUT: Respond with valid JSON only. No extra text.

{
  "action": "run_database_query" | "ask_clarification" | "reply_without_database",
  "message": "<if ask_clarification: the single warm question to ask | if reply_without_database: the full friendly response | if run_database_query: empty string>",
  "extracted": {
    "query_mode": "eligibility" | "informational" | "unknown",
    "metric_type": "score" | "rank" | "none",
    "metric_value": <number or null>,
    "subject": "self" | "friend" | "general" | "unknown",
    "home_state_for_query": "<home state of the candidate in this query if mentioned, else empty>",
    "target_state": "<state/MCC if present else empty>",
    "category": "<GENERAL/OBC/SC/ST/EWS if present else empty>",
    "college_type": "<GOVERNMENT/PRIVATE/DEEMED/ALL if present else empty>",
    "use_profile_defaults": true | false,
    "missing_slots": ["state_or_mcc", "category", "college_type", "neet_metric"],
    "needs_confirmation": true | false,
    "confirmation_question": "<single question if needs_confirmation=true else empty>"
  }
}
"""


@dataclass(frozen=True)
class QueryGateResult:
    action: Literal["run_database_query", "ask_clarification", "reply_without_database"]
    message: str
    extracted: dict


def _parse_json_object(raw: str) -> dict:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _gate_college_search_without_metric(question: str, extracted: dict) -> bool:
    """
    True when the gate chose reply_without_database but the user is clearly asking
    for colleges/cutoffs in a state without providing NEET marks or AIR.
    """
    if extracted.get("metric_value") is not None:
        return False
    if str(extracted.get("metric_type", "none")).lower() not in ("none", ""):
        return False
    target = str(extracted.get("target_state", "")).strip()
    if not target:
        return False
    low = (question or "").lower()
    college_intent = any(
        w in low
        for w in (
            "college",
            "colleges",
            "mbbs",
            "bds",
            "cutoff",
            "cut off",
            "eligible",
            "eligibility",
            "get in",
            "which college",
            "which colleges",
            "looking for",
            "options",
            "friend",
            "cousin",
            "someone",
        )
    )
    if not college_intent:
        return False
    # Process-only questions sometimes mention a state; avoid overriding if no college intent.
    return True


def gate_user_query(
    client: OpenAI,
    user_question: str,
    *,
    request_id: str | None = None,
) -> QueryGateResult:
    """Single LLM call to route: search vs clarify vs small talk."""
    rid = request_id or "-"
    logger.info("[%s] Gate LLM system prompt:\n%s", rid, _clip(READINESS_SYSTEM))
    logger.info("[%s] Gate LLM user prompt:\n%s", rid, _clip(user_question.strip()))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": READINESS_SYSTEM},
            {"role": "user", "content": user_question.strip()},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = (response.choices[0].message.content or "").strip()
    logger.info("[%s] Gate LLM raw output:\n%s", rid, _clip(raw))
    try:
        data = _parse_json_object(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("[%s] Query gate JSON parse failed: %s — raw=%r", rid, exc, raw[:200])
        return QueryGateResult(
            action="ask_clarification",
            message=(
                "Thanks for sharing that. If you want me to suggest possible colleges, please tell me your "
                "category (e.g. GENERAL, OBC, SC, ST, EWS), whether you want your state or MCC/all India, "
                "and if you prefer government, private, or deemed colleges."
            ),
            extracted={
                "query_mode": "unknown",
                "metric_type": "none",
                "metric_value": None,
                "subject": "unknown",
                "home_state_for_query": "",
                "target_state": "",
                "category": "",
                "college_type": "",
                "use_profile_defaults": False,
                "missing_slots": [],
                "needs_confirmation": False,
                "confirmation_question": "",
            },
        )

    action = data.get("action", "ask_clarification")
    message = (data.get("message") or "").strip()

    if action not in ("run_database_query", "ask_clarification", "reply_without_database"):
        action = "ask_clarification"
    if action == "run_database_query":
        message = ""
    elif action == "reply_without_database" and not message:
        message = (
            "Hi! When you want cutoff-based college options, share your category, "
            "state or MCC/all India, and government vs private vs deemed preference."
        )
    elif not message:
        message = (
            "Thanks for sharing. If you're looking for college options, please share your category, "
            "state (or MCC/all India), and preferred college type (government / private / deemed)."
        )

    extracted = data.get("extracted")
    if not isinstance(extracted, dict):
        extracted = {}

    query_mode = str(extracted.get("query_mode", "unknown")).lower()
    if query_mode not in ("eligibility", "informational", "unknown"):
        query_mode = "unknown"
    metric_type = str(extracted.get("metric_type", "none")).lower()
    if metric_type not in ("score", "rank", "none"):
        metric_type = "none"
    metric_value = extracted.get("metric_value")
    if isinstance(metric_value, str) and metric_value.strip().isdigit():
        metric_value = int(metric_value.strip())
    if not isinstance(metric_value, (int, float)):
        metric_value = None
    subject = str(extracted.get("subject", "unknown")).lower()
    if subject not in ("self", "friend", "general", "unknown"):
        subject = "unknown"
    use_profile_defaults = bool(extracted.get("use_profile_defaults", False))
    college_type = str(extracted.get("college_type", "")).strip().upper()
    if college_type not in ("", "GOVERNMENT", "PRIVATE", "DEEMED", "ALL", "AIIMS", "JIPMER", "BHU", "AMU"):
        college_type = ""
    raw_missing = extracted.get("missing_slots", [])
    if not isinstance(raw_missing, list):
        raw_missing = []
    missing_slots = [
        str(x).strip().lower()
        for x in raw_missing
        if str(x).strip().lower()
        in ("state_or_mcc", "category", "college_type", "neet_metric")
    ]
    needs_confirmation = bool(extracted.get("needs_confirmation", False))
    confirmation_question = str(extracted.get("confirmation_question", "")).strip()

    extracted_clean = {
        "query_mode": query_mode,
        "metric_type": metric_type,
        "metric_value": int(metric_value) if isinstance(metric_value, (int, float)) else None,
        "subject": subject,
        "home_state_for_query": str(extracted.get("home_state_for_query", "")).strip(),
        "target_state": str(extracted.get("target_state", "")).strip(),
        "category": str(extracted.get("category", "")).strip(),
        "college_type": college_type,
        "use_profile_defaults": use_profile_defaults,
        "missing_slots": missing_slots,
        "needs_confirmation": needs_confirmation,
        "confirmation_question": confirmation_question,
    }

    # Safety net: never answer college/cutoff questions from model memory.
    if action == "reply_without_database" and _gate_college_search_without_metric(
        user_question, extracted_clean
    ):
        action = "ask_clarification"
        place = str(extracted_clean.get("target_state", "")).strip() or "that state"
        message = (
            "I only suggest colleges using our live cutoff database, not from memory. "
            f"Please share their NEET marks (out of 720) or All India Rank so I can search cutoffs for {place}."
        )
        ms = list(extracted_clean["missing_slots"])
        if "neet_metric" not in ms:
            ms.append("neet_metric")
        extracted_clean["missing_slots"] = ms

    logger.info(
        "[%s] Query gate parsed: action=%s, mode=%s, metric=%s:%s, extracted=%s",
        rid,
        action,
        extracted_clean["query_mode"],
        extracted_clean["metric_type"],
        extracted_clean["metric_value"],
        extracted_clean,
    )
    return QueryGateResult(action=action, message=message, extracted=extracted_clean)


def classify_intro_step_intent(
    client: OpenAI,
    user_text: str,
    *,
    request_id: str | None = None,
) -> Literal["continue_onboarding", "provided_neet_metric", "unclear"]:
    rid = request_id or "-"
    prompt = (user_text or "").strip()
    logger.info("[%s] Intro-intent LLM prompt:\n%s", rid, _clip(prompt))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": INTRO_STEP_SYSTEM.strip()},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = (response.choices[0].message.content or "").strip()
    logger.info("[%s] Intro-intent LLM raw output:\n%s", rid, _clip(raw))
    try:
        data = _parse_json_object(raw)
        intent = str(data.get("intent", "unclear")).strip().lower()
        if intent in ("continue_onboarding", "provided_neet_metric", "unclear"):
            return intent  # type: ignore[return-value]
    except Exception:
        pass

    # Fallback only if LLM output malformed.
    low = prompt.lower()
    if re.search(r"\b(?:air|rank|score|marks?)\b", low) or re.search(r"\b\d{2,7}\b", low):
        return "provided_neet_metric"
    if re.search(r"\b(?:yes|yeah|yep|ok|okay|start|proceed|help|find)\b", low):
        return "continue_onboarding"
    return "unclear"


def interpret_onboarding_response(
    client: OpenAI,
    *,
    current_step: str,
    user_input: str,
    current_preferences: dict,
    step_options: list[dict] | None,
    request_id: str | None = None,
) -> dict:
    """
    LLM-first onboarding parser. Returns normalized JSON contract:
    {
      action: apply_update|ask_rephrase|fallback,
      updates: dict,
      clear_fields: list[str],
      message: str,
      acknowledgement: str
    }
    """
    rid = request_id or "-"
    payload = {
        "current_step": current_step,
        "user_input": user_input,
        "current_preferences": current_preferences or {},
        "step_options": step_options or [],
    }
    logger.info("[%s] Onboarding interpreter input: %s", rid, _clip(json.dumps(payload, ensure_ascii=True)))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": ONBOARDING_INTERPRETER_SYSTEM.strip()},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = (response.choices[0].message.content or "").strip()
    logger.info("[%s] Onboarding interpreter raw output:\n%s", rid, _clip(raw))
    try:
        data = _parse_json_object(raw)
    except Exception:
        return {
            "action": "fallback",
            "updates": {},
            "clear_fields": [],
            "message": "",
            "acknowledgement": "",
        }

    action = str(data.get("action", "fallback")).strip().lower()
    if action not in ("apply_update", "ask_rephrase", "fallback"):
        action = "fallback"
    updates = data.get("updates")
    if not isinstance(updates, dict):
        updates = {}
    # Fix common LLM mistake: course preference written under category.
    cat_mis = updates.get("category")
    if isinstance(cat_mis, str):
        for v in COURSE_PREFERENCE_VALUES:
            if cat_mis.strip().upper() == v.upper():
                updates = dict(updates)
                updates.pop("category", None)
                if not updates.get("course"):
                    updates["course"] = v
                break
    clear_fields = data.get("clear_fields")
    if not isinstance(clear_fields, list):
        clear_fields = []
    clear_fields = [str(x).strip() for x in clear_fields if str(x).strip()]
    message = str(data.get("message", "")).strip()
    acknowledgement = str(data.get("acknowledgement", "")).strip()
    return {
        "action": action,
        "updates": updates,
        "clear_fields": clear_fields,
        "message": message,
        "acknowledgement": acknowledgement,
    }

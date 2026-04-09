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

logger = logging.getLogger("neet_assistant.validation")

READINESS_SYSTEM = """You are the intake layer for a NEET UG cutoff assistant backed by a real database.

The user message may be a multi-turn transcript: lines starting with "Student:" or "Counsellor:", ending with "Student (latest message): ...". Use the ENTIRE active thread — earlier turns in this thread may already state NEET score, rank, category, state, or college type.

Critical counseling behavior:
- Never pre-assume state/category/college type from user profile defaults or old historical assumptions.
- Run database query only when required details are explicitly provided by the student in the active conversation.
- If the student only shares score/rank and not clear intent for colleges, ask a warm follow-up first.

Your job: choose ONE action:

1) "run_database_query" — ONLY if ALL of the following are clearly present across the thread (usually by the latest message, or combined with earlier messages):
   - Reservation category (GENERAL / OBC / SC / ST / EWS or common synonyms like OC, gen, open category), OR the user explicitly says they want "all categories" / "any category".
   - Geography: a specific state name OR MCC / all India / AIQ (all-India counselling), OR the user clearly names one region.
   - College type: government, private, deemed, AIIMS, JIPMER, BHU, AMU, OR explicit phrases like "any type", "both govt and private", "doesn't matter".

2) "ask_clarification" — User is asking for colleges, cutoffs, options, ranks, scores, or guidance, but one or more of category / geography / college type is missing or too vague. Write a short, warm counselor-style message that:
   - Thanks them for what they shared (score/rank if given).
   - First confirms intent naturally (e.g. "Are you looking for possible colleges based on this score/rank?").
   - Asks ONLY for what is still missing (category, which state or MCC/AIQ, and government vs private vs deemed if not covered).
   - Do NOT invent search results.

3) "reply_without_database" — Greeting, thanks, off-topic, or general NEET info that does not need listing colleges from a database. Reply briefly and kindly; no SQL.

Respond with valid JSON only (no markdown), shape:
{"action":"run_database_query"|"ask_clarification"|"reply_without_database","message":""}
Use "message" for clarification text or conversational reply; use empty string for message when action is run_database_query.
"""


@dataclass(frozen=True)
class QueryGateResult:
    action: Literal["run_database_query", "ask_clarification", "reply_without_database"]
    message: str


def _parse_json_object(raw: str) -> dict:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def gate_user_query(client: OpenAI, user_question: str) -> QueryGateResult:
    """Single LLM call to route: search vs clarify vs small talk."""
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
    try:
        data = _parse_json_object(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Query gate JSON parse failed: %s — raw=%r", exc, raw[:200])
        return QueryGateResult(
            action="ask_clarification",
            message=(
                "Thanks for sharing that. If you want me to suggest possible colleges, please tell me your "
                "**category** (e.g. GENERAL, OBC, SC, ST, EWS), whether you want **your state or MCC/all India**, "
                "and if you prefer **government, private, or deemed** colleges."
            ),
        )

    action = data.get("action", "ask_clarification")
    message = (data.get("message") or "").strip()

    if action not in ("run_database_query", "ask_clarification", "reply_without_database"):
        action = "ask_clarification"
    if action == "run_database_query":
        message = ""
    elif action == "reply_without_database" and not message:
        message = (
            "Hi! When you want cutoff-based college options, share your **category**, "
            "**state or MCC/all India**, and **government vs private vs deemed** preference."
        )
    elif not message:
        message = (
            "Thanks for sharing. If you're looking for college options, please share your **category**, "
            "**state (or MCC/all India)**, and preferred **college type** (government / private / deemed)."
        )

    logger.info("Query gate: action=%s", action)
    return QueryGateResult(action=action, message=message)

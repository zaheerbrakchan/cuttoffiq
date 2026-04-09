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
   - If the student asks a general/hypothetical question ("what colleges for 450 score in Karnataka?") → treat as general, no profile needed.
   - If the student is just chatting, saying thanks, or asking about NEET process/dates → reply directly.

3. WHEN TO ASK FOR CLARIFICATION (only when truly stuck):
   - Critical piece is missing AND cannot be inferred from conversation.
   - Never ask for info already given earlier in the chat.
   - Never ask more than ONE clarifying question at a time.

4. EMOTIONAL AWARENESS:
   - If the student sounds anxious, worried, or discouraged → acknowledge that warmly before routing.
   - If they say things like "I don't know what to do" or "I'm scared" → your reply_without_database should be empathetic first.

---

CHOOSE EXACTLY ONE ACTION:

→ "run_database_query"     — You have enough info (from profile or query) to search colleges
→ "ask_clarification"      — ONE critical piece is missing and cannot be inferred  
→ "reply_without_database" — Greetings, thanks, process questions, emotional support, general NEET info

---

OUTPUT: Respond with valid JSON only. No extra text.

{"action": "run_database_query" | "ask_clarification" | "reply_without_database", "message": "<if ask_clarification: the single warm question to ask | if reply_without_database: the full friendly response | if run_database_query: empty string>"}
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
                "category (e.g. GENERAL, OBC, SC, ST, EWS), whether you want your state or MCC/all India, "
                "and if you prefer government, private, or deemed colleges."
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
            "Hi! When you want cutoff-based college options, share your category, "
            "state or MCC/all India, and government vs private vs deemed preference."
        )
    elif not message:
        message = (
            "Thanks for sharing. If you're looking for college options, please share your category, "
            "state (or MCC/all India), and preferred college type (government / private / deemed)."
        )

    logger.info("Query gate: action=%s", action)
    return QueryGateResult(action=action, message=message)

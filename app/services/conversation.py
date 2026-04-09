"""
Build a single text block from chat history + latest user message for LLM calls.

Design: stateless API — client sends last N messages; no server session DB required.
Scales on Railway; users can resume in the same browser via localStorage.
"""

from __future__ import annotations

import json

from openai import OpenAI

# Cap total size to control token usage (approximate).
MAX_CONTEXT_CHARS = 12_000
# Max prior turns (user + assistant pairs) to include.
MAX_PRIOR_MESSAGES = 16
MAX_RECENT_MESSAGES = 8
SUMMARY_MAX_WORDS = 300


def build_contextual_query(
    latest_message: str,
    messages: list[dict] | None,
) -> str:
    """
    `messages`: prior turns only, each {"role": "user"|"assistant", "content": str}.
    `latest_message`: the new user text (not duplicated in messages).
    """
    latest_message = (latest_message or "").strip()
    if not latest_message:
        return ""

    if not messages:
        return latest_message

    prior = messages[-MAX_PRIOR_MESSAGES:]
    lines: list[str] = []
    for m in prior:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            lines.append(f"Counsellor: {content}")
        else:
            lines.append(f"Student: {content}")

    block = "\n".join(lines)
    if not block:
        return latest_message

    combined = f"{block}\n\nStudent (latest message): {latest_message}"
    if len(combined) <= MAX_CONTEXT_CHARS:
        return combined
    # Trim from the oldest lines while keeping the latest user message intact.
    head = block
    while len(f"{head}\n\nStudent (latest message): {latest_message}") > MAX_CONTEXT_CHARS and head:
        line_break = head.find("\n")
        if line_break == -1:
            head = ""
        else:
            head = head[line_break + 1 :].lstrip()
    return f"{head}\n\nStudent (latest message): {latest_message}" if head else latest_message


def build_contextual_query_from_memory(
    latest_message: str,
    *,
    summary_text: str,
    recent_chats: list[dict] | None,
    preferences_json: dict | None,
) -> str:
    latest = (latest_message or "").strip()
    if not latest:
        return ""
    parts: list[str] = []
    if preferences_json:
        parts.append(f"Known user preferences (JSON): {json.dumps(preferences_json, ensure_ascii=True)}")
    if summary_text:
        parts.append(f"Conversation summary so far: {summary_text.strip()}")
    if recent_chats:
        recent_block = build_contextual_query(
            latest_message=latest,
            messages=recent_chats[-MAX_RECENT_MESSAGES:],
        )
        parts.append(recent_block)
    else:
        parts.append(latest)
    combined = "\n\n".join(p for p in parts if p.strip())
    return combined[:MAX_CONTEXT_CHARS]


def append_recent_chats(
    recent_chats: list[dict] | None,
    *,
    user_text: str,
    assistant_text: str,
) -> list[dict]:
    prior = list(recent_chats or [])
    prior.append({"role": "user", "content": user_text.strip()})
    prior.append({"role": "assistant", "content": assistant_text.strip()})
    return prior[-MAX_RECENT_MESSAGES:]


def update_summary_counter(preferences_json: dict, increment: int = 1) -> dict:
    prefs = dict(preferences_json or {})
    since = int(prefs.get("_since_summary", 0))
    prefs["_since_summary"] = since + increment
    return prefs


def should_refresh_summary(
    preferences_json: dict,
    *,
    turns_interval: int,
    min_context_chars: int,
    recent_chats: list[dict],
) -> bool:
    turns = int((preferences_json or {}).get("_since_summary", 0))
    if turns < turns_interval:
        return False
    chars = sum(len((m.get("content") or "").strip()) for m in (recent_chats or []))
    return chars >= min_context_chars


def reset_summary_counter(preferences_json: dict) -> dict:
    prefs = dict(preferences_json or {})
    prefs["_since_summary"] = 0
    return prefs


def generate_compact_summary(
    client: OpenAI,
    *,
    previous_summary: str,
    recent_chats: list[dict],
    max_words: int = SUMMARY_MAX_WORDS,
) -> str:
    if not recent_chats:
        return (previous_summary or "").strip()
    transcript_lines: list[str] = []
    for m in recent_chats:
        role = "Student" if m.get("role") == "user" else "Counsellor"
        transcript_lines.append(f"{role}: {(m.get('content') or '').strip()}")
    transcript = "\n".join(transcript_lines)
    prompt = f"""
Update this NEET counselling memory summary in <= {max_words} words.
Keep only persistent facts and important constraints/preferences:
- category, state/MCC preference, college type, course, score/rank, quota/round preferences
- important clarifications asked and answered
Avoid fluff, greetings, and repetitive details.

Previous summary:
{previous_summary or '(empty)'}

Recent chat:
{transcript}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt.strip()}],
        temperature=0.1,
    )
    return (response.choices[0].message.content or "").strip()

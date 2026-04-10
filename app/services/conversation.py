"""
Build a single text block from chat history + latest user message for LLM calls.

Design: stateless API — client sends last N messages; no server session DB required.
Scales on Railway; users can resume in the same browser via localStorage.

CONTEXT ISOLATION STRATEGY:
To prevent LLMs from confusing data across different queries/subjects:
1. Format context with clear separation between historical and current data
2. Include explicit instructions for the LLM about data ownership
3. For SQL generation, use only user messages (not assistant responses with result numbers)
4. Let the LLM decide subject/intent - no hardcoded regex patterns
5. CRITICAL: Distinguish TARGET STATE from HOME STATE clearly
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


def _truncate_for_context(text: str, max_chars: int) -> str:
    """Truncate text but preserve key structural info."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


def build_isolated_context(
    latest_message: str,
    messages: list[dict] | None,
    *,
    include_assistant_responses: bool = True,
    max_user_history: int = 6,
) -> str:
    """
    Build context with clear formatting for intelligent LLM processing.
    
    The LLM will handle THREE query types:
    1. SELF - user asking about themselves (use profile as defaults)
    2. FRIEND - user asking about someone else (use only conversation data)
    3. GENERAL - hypothetical query (use only explicit values)
    """
    latest = (latest_message or "").strip()
    if not latest:
        return ""
    
    if not messages:
        return f"Student (current message): {latest}"
    
    # Build context with clear structure
    lines: list[str] = []
    
    # Instructions for the LLM
    lines.append("=== CONVERSATION CONTEXT ===")
    lines.append("")
    lines.append("QUERY TYPES TO IDENTIFY:")
    lines.append("• SELF QUERY: User asks about themselves ('my options', 'can I get', 'for me')")
    lines.append("  → Use PROFILE data as defaults, conversation overrides profile")
    lines.append("• FRIEND QUERY: User asks about someone else ('my friend', 'his score', 'for her')")
    lines.append("  → Use ONLY conversation data for that person, NOT profile")
    lines.append("• GENERAL QUERY: Hypothetical ('student with 650', 'if someone has')")
    lines.append("  → Use ONLY explicitly stated values")
    lines.append("")
    lines.append("DATA RULES:")
    lines.append("• User messages = INPUT data (score, rank, state, category) - USE these")
    lines.append("• Counsellor search results = OUTPUT data - do NOT use as query parameters")
    lines.append("• Multi-turn: aggregate data across user messages for current subject")
    lines.append("• Multiple subjects: use ONLY the current/latest subject's data")
    lines.append("")
    lines.append("⚠️ TARGET STATE vs HOME STATE (CRITICAL!):")
    lines.append("• TARGET STATE: 'colleges in Bihar', 'options in Karnataka' → where to SEARCH")
    lines.append("• HOME STATE: 'home state is Kerala', 'domicile UP' → affects eligibility filter")
    lines.append("• These are DIFFERENT! A query may have target=BIHAR but home=KERALA")
    lines.append("• When someone says 'home state' or 'domicile', do NOT change target state!")
    lines.append("")
    lines.append("RECOGNIZE LOWERCASE:")
    lines.append("• 'st', 'St', 'ST' all mean category=ST - accept as valid answer!")
    lines.append("• 'obc' = OBC, 'gen' = GENERAL, 'kerela' = KERALA")
    lines.append("")
    lines.append("--- CONVERSATION ---")
    lines.append("")
    
    # Process messages
    turn_count = 0
    for m in (messages or [])[-MAX_PRIOR_MESSAGES:]:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        
        if role == "user":
            turn_count += 1
            if turn_count <= max_user_history:
                lines.append(f"[Turn {turn_count}] Student: {content}")
        elif role == "assistant" and include_assistant_responses:
            truncated = _truncate_for_context(content, 400)
            lines.append(f"[Turn {turn_count}] Counsellor: {truncated}")
    
    lines.append("")
    lines.append("--- CURRENT MESSAGE ---")
    lines.append(f"Student: {latest}")
    
    combined = "\n".join(lines)
    return combined[:MAX_CONTEXT_CHARS]


def build_contextual_query(
    latest_message: str,
    messages: list[dict] | None,
    *,
    isolate_context: bool = False,
) -> str:
    """
    `messages`: prior turns only, each {"role": "user"|"assistant", "content": str}.
    `latest_message`: the new user text (not duplicated in messages).
    
    If isolate_context=True, uses new tagged format to prevent data confusion.
    """
    latest_message = (latest_message or "").strip()
    if not latest_message:
        return ""

    if isolate_context:
        return build_isolated_context(latest_message, messages)

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


def build_user_only_context(
    latest_message: str,
    messages: list[dict] | None,
    max_prior_user_msgs: int = 4,
) -> str:
    """
    Build context using ONLY user messages (excludes assistant responses).
    
    Why exclude assistant responses for SQL generation:
    - Assistant responses contain SEARCH RESULTS (college names, cutoffs)
    - Those numbers are OUTPUT data, not INPUT parameters
    - Including them can confuse the SQL LLM into using result numbers as query params
    
    User messages contain the actual INPUT:
    - Score/rank values the user stated
    - State, category, course preferences
    - Who the query is about (self/friend)
    """
    latest = (latest_message or "").strip()
    if not latest:
        return ""
    
    if not messages:
        return f"Student (current): {latest}"
    
    # Extract only user messages - these contain the actual query parameters
    user_msgs: list[str] = []
    for m in (messages or []):
        if m.get("role") == "user":
            content = (m.get("content") or "").strip()
            if content:
                user_msgs.append(content)
    
    # Take recent user messages to understand the conversation flow
    prior_user = user_msgs[-max_prior_user_msgs:] if user_msgs else []
    
    if not prior_user:
        return f"Student (current): {latest}"
    
    # Build context with clear structure
    lines = []
    lines.append("User messages from conversation (contains INPUT data):")
    for i, msg in enumerate(prior_user, 1):
        lines.append(f"  Turn {i}: {msg}")
    lines.append(f"  Current: {latest}")
    lines.append("")
    lines.append("Extract query parameters from these user messages.")
    lines.append("If multiple subjects discussed, use data for the MOST RECENT subject only.")
    
    return "\n".join(lines)


def build_contextual_query_from_memory(
    latest_message: str,
    *,
    summary_text: str,
    recent_chats: list[dict] | None,
    preferences_json: dict | None,
    isolate_context: bool = True,
) -> str:
    """
    Build context from memory with optional context isolation.
    
    When isolate_context=True (default), adds explicit markers to prevent
    the LLM from confusing historical data with current query parameters.
    """
    latest = (latest_message or "").strip()
    if not latest:
        return ""
    
    parts: list[str] = []
    
    # User preferences - these are CONFIRMED facts about the user
    if preferences_json:
        parts.append(f"[USER_PROFILE - Confirmed facts about THIS user]: {json.dumps(preferences_json, ensure_ascii=True)}")
    
    # Summary - be careful, this may contain historical data
    if summary_text:
        if isolate_context:
            parts.append(f"[CONVERSATION_SUMMARY - Historical context, verify before using]: {summary_text.strip()}")
        else:
            parts.append(f"Conversation summary so far: {summary_text.strip()}")
    
    # Recent chats with isolation
    if recent_chats:
        if isolate_context:
            recent_block = build_isolated_context(
                latest_message=latest,
                messages=recent_chats[-MAX_RECENT_MESSAGES:],
            )
        else:
            recent_block = build_contextual_query(
                latest_message=latest,
                messages=recent_chats[-MAX_RECENT_MESSAGES:],
            )
        parts.append(recent_block)
    else:
        # No history - just the current message
        parts.append(f"Student (current query): {latest}")
    
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

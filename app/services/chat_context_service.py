from __future__ import annotations

import logging

from supabase import Client

from app.services.onboarding_service import normalize_misplaced_course_category

logger = logging.getLogger("neet_assistant.chat_context")


def _sanitize_preferences(preferences: dict | None) -> dict:
    """
    Remove onboarding/runtime-only keys that should not persist as user profile data.
    """
    prefs = dict(preferences or {})
    for transient_key in ("intro", "_pending_neet_value"):
        prefs.pop(transient_key, None)
    return normalize_misplaced_course_category(prefs)


def load_user_chat_context(client: Client, user_id: int) -> dict:
    result = (
        client.table("user_chat_context")
        .select("user_id, summary_text, recent_chats, preferences_json")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    if not isinstance(rows, list) or not rows:
        logger.info("No chat context row for user_id=%s, using defaults", user_id)
        return {
            "user_id": user_id,
            "summary_text": "",
            "recent_chats": [],
            "preferences_json": {},
        }
    row = rows[0]
    return {
        "user_id": user_id,
        "summary_text": row.get("summary_text") or "",
        "recent_chats": row.get("recent_chats") or [],
        "preferences_json": _sanitize_preferences(row.get("preferences_json") or {}),
    }


def save_user_chat_context(
    client: Client,
    *,
    user_id: int,
    summary_text: str,
    recent_chats: list[dict],
    preferences_json: dict,
) -> None:
    payload = {
        "user_id": user_id,
        "summary_text": summary_text or "",
        "recent_chats": recent_chats,
        "preferences_json": _sanitize_preferences(preferences_json),
    }
    client.table("user_chat_context").upsert(payload).execute()
    logger.info("Saved chat context for user_id=%s", user_id)


def clear_user_chat_context(client: Client, user_id: int) -> None:
    """Clear chat history but KEEP user preferences (so they don't need to re-onboard)."""
    # First, load existing preferences
    existing = load_user_chat_context(client, user_id)
    
    payload = {
        "user_id": user_id,
        "summary_text": "",
        "recent_chats": [],
        "preferences_json": _sanitize_preferences(existing.get("preferences_json", {})),  # Keep preferences!
    }
    client.table("user_chat_context").upsert(payload).execute()
    logger.info("Cleared chat history for user_id=%s (preferences preserved)", user_id)

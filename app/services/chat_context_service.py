from __future__ import annotations

import logging

from supabase import Client

logger = logging.getLogger("neet_assistant.chat_context")


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
        "preferences_json": row.get("preferences_json") or {},
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
        "preferences_json": preferences_json,
    }
    client.table("user_chat_context").upsert(payload).execute()
    logger.info("Saved chat context for user_id=%s", user_id)


def clear_user_chat_context(client: Client, user_id: int) -> None:
    payload = {
        "user_id": user_id,
        "summary_text": "",
        "recent_chats": [],
        "preferences_json": {},
    }
    client.table("user_chat_context").upsert(payload).execute()
    logger.info("Cleared chat context for user_id=%s", user_id)

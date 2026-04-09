from __future__ import annotations

import json
import logging

from supabase import Client

logger = logging.getLogger("neet_assistant.user_profile")


def load_user_preferences(client: Client, user_id: int) -> dict:
    """
    Fetch `preferences` JSON from users table for a given user.
    Expected column: users.preferences (json/jsonb or text JSON).
    """
    result = (
        client.table("users")
        .select("preferences")
        .eq("id", user_id)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    if not isinstance(rows, list) or not rows:
        return {}
    pref = rows[0].get("preferences")
    if isinstance(pref, dict):
        return pref
    if isinstance(pref, str):
        try:
            parsed = json.loads(pref)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def extract_home_state(preferences: dict) -> str | None:
    """
    Extract home state from common preference layouts.
    Supports flat and nested keys.
    """
    if not isinstance(preferences, dict):
        return None
    direct_keys = ("state", "home_state", "state_preference")
    for key in direct_keys:
        value = preferences.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # Common nested payloads from registration forms.
    nested_candidates = [
        preferences.get("profile"),
        preferences.get("personal"),
        preferences.get("location"),
        preferences.get("student"),
    ]
    for node in nested_candidates:
        if not isinstance(node, dict):
            continue
        for key in direct_keys:
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None

import os
import logging

from supabase import Client, create_client

logger = logging.getLogger("neet_assistant.supabase")


def get_supabase_client() -> Client:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY environment variables.")

    return create_client(supabase_url, supabase_key)


def execute_neet_query(client: Client, sql: str) -> list[dict]:
    logger.info("Executing RPC execute_neet_query")
    result = client.rpc("execute_neet_query", {"query_text": sql}).execute()
    if result.data is None:
        logger.info("RPC returned no data")
        return []
    if isinstance(result.data, list):
        logger.info("RPC returned list rows=%d", len(result.data))
        return result.data
    logger.info("RPC returned a single row object")
    return [result.data]


def _scalar_values(rows: list[dict], key: str) -> list[str]:
    values: list[str] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            values.append(text)
    return values


def get_suggestion_context(client: Client) -> dict[str, list[str]]:
    states = execute_neet_query(
        client,
        "SELECT DISTINCT state FROM neet_ug_2025_cutoffs WHERE state IS NOT NULL ORDER BY state LIMIT 20",
    )
    categories = execute_neet_query(
        client,
        "SELECT DISTINCT category FROM neet_ug_2025_cutoffs WHERE category IS NOT NULL ORDER BY category LIMIT 10",
    )
    courses = execute_neet_query(
        client,
        "SELECT DISTINCT course FROM neet_ug_2025_cutoffs WHERE course IS NOT NULL ORDER BY course LIMIT 10",
    )
    rounds = execute_neet_query(
        client,
        "SELECT DISTINCT round FROM neet_ug_2025_cutoffs WHERE round IS NOT NULL ORDER BY round LIMIT 10",
    )
    return {
        "states": _scalar_values(states, "state"),
        "categories": _scalar_values(categories, "category"),
        "courses": _scalar_values(courses, "course"),
        "rounds": _scalar_values(rounds, "round"),
    }


def get_categories_for_state(client: Client, state: str) -> list[str]:
    """Fetch distinct categories available for a specific state from the database."""
    if not state:
        return []
    sql = f"SELECT DISTINCT category FROM neet_ug_2025_cutoffs WHERE state ILIKE '{state}' AND category IS NOT NULL ORDER BY category"
    rows = execute_neet_query(client, sql)
    return _scalar_values(rows, "category")


def get_sub_categories_for_state_and_category(client: Client, state: str, category: str) -> list[str]:
    """Fetch distinct sub-categories available for a specific state and category from the database."""
    if not state or not category:
        return []
    sql = f"SELECT DISTINCT sub_category FROM neet_ug_2025_cutoffs WHERE state ILIKE '{state}' AND category ILIKE '{category}' AND sub_category IS NOT NULL ORDER BY sub_category"
    rows = execute_neet_query(client, sql)
    return _scalar_values(rows, "sub_category")

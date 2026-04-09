from __future__ import annotations

import logging
import os
import re
import time
import uuid
import json
from pathlib import Path

from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.services.chat_context_service import (
    clear_user_chat_context,
    load_user_chat_context,
    save_user_chat_context,
)
from app.services.conversation import (
    append_recent_chats,
    build_contextual_query,
)
from app.services.openai_service import get_openai_client
from app.services.query_normalization import normalize_user_question, resolve_state
from app.services.query_validation import (
    gate_user_query,
    classify_intro_step_intent,
    interpret_onboarding_response,
)
from app.services.sql_generator import (
    generate_counsellor_answer,
    generate_sql,
)
from app.services.supabase_service import (
    execute_neet_query,
    get_supabase_client,
    get_categories_for_state,
    get_college_types_for_state,
    get_sub_categories_for_state_and_category,
)
from app.services.onboarding_service import (
    check_onboarding_status,
    process_onboarding_response,
    get_onboarding_complete_message,
    get_step_confirmation,
    format_preferences_for_context,
    db_categories_to_options,
    db_college_types_to_options,
    db_sub_categories_to_options,
    get_profile_confirmation_message,
    is_profile_confirmation_response,
    needs_profile_confirmation,
    normalize_misplaced_course_category,
)

# Fixed chips: suggest queries that work well with user profile context.
DEFAULT_SUGGESTIONS: list[str] = [
    "Which colleges can I get in my state?",
    "Show me government colleges in MCC/All India",
    "What are my options in private colleges?",
    "Top colleges within my rank range",
    "Best BDS options for me",
]

def _onboarding_field_still_missing(step: str | None, prefs: dict) -> bool:
    """True if prefs do not yet satisfy the onboarding step (LLM may have skipped updates)."""
    if not step:
        return False
    p = prefs or {}
    if step == "intro":
        return not p.get("intro")
    if step in ("neet_score", "neet_score_clarify"):
        return not p.get("neet_score")
    if step == "course":
        return not p.get("course")
    if step == "home_state":
        return not p.get("home_state")
    if step == "category":
        return not p.get("category")
    if step == "sub_category":
        return "sub_category" not in p
    if step == "college_type":
        v = p.get("college_type")
        if v is None:
            return True
        return isinstance(v, list) and len(v) == 0
    return False


def _is_llm_cross_step_update(
    step: str | None, updates: dict | None, clear_fields: list[str] | None
) -> bool:
    """
    True when LLM intentionally updated a different onboarding field than the
    current step (e.g. correcting home_state while we are on category step).
    """
    if not step:
        return False
    step_to_field = {
        "intro": "intro",
        "neet_score": "neet_score",
        "neet_score_clarify": "neet_score",
        "course": "course",
        "home_state": "home_state",
        "category": "category",
        "sub_category": "sub_category",
        "college_type": "college_type",
    }
    current_field = step_to_field.get(step)
    if not current_field:
        return False
    normalized_updates = updates if isinstance(updates, dict) else {}
    normalized_clears = {
        str(x).strip() for x in (clear_fields or []) if str(x).strip()
    }
    onboarding_fields = set(step_to_field.values())
    updated_fields = {
        k for k in normalized_updates.keys() if isinstance(k, str) and k in onboarding_fields
    }
    if any(field != current_field for field in updated_fields):
        return True
    # Also count dependent field clears as a cross-step correction signal.
    if any(field != current_field for field in normalized_clears if field in onboarding_fields):
        return True
    return False


load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("neet_assistant")
_MAX_LOG_CHARS = 4000


def _clip(text: str, limit: int = _MAX_LOG_CHARS) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated {len(text) - limit} chars]"

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="AI NEET Counselling Assistant")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def _personalize_clarification(message: str, user_home_state: str | None) -> str:
    text = (message or "").strip()
    if not text or not user_home_state:
        return text
    # If the message already asks for state, make it more counsellor-like with home-state option.
    if "which state" in text.lower() or "state (or mcc" in text.lower():
        return (
            text.replace(
                "which state you're interested in",
                f"whether you are looking in your home state ({user_home_state}) or another state/MCC",
            )
            .replace(
                "state (or MCC/all India)",
                f"home state ({user_home_state}) or another state/MCC (all India)",
            )
        )
    return text


def _resolve_own_state_phrase(text: str, user_home_state: str | None) -> str:
    if not text or not user_home_state:
        return text
    out = text
    patterns = [
        r"\bmy own state\b",
        r"\bown state\b",
        r"\bhome state\b",
        r"\bmy state\b",
    ]
    for p in patterns:
        out = re.sub(p, user_home_state, out, flags=re.IGNORECASE)
    return out


def _extract_college_types_from_text(text: str) -> list[str]:
    low = (text or "").lower()
    found: list[str] = []
    if re.search(r"\b(govt|government)\b", low):
        found.append("GOVERNMENT")
    if re.search(r"\bprivate\b", low):
        found.append("PRIVATE")
    if re.search(r"\bdeemed\b", low):
        found.append("DEEMED")
    if re.search(r"\b(aiims|jipmer|bhu|amu)\b", low):
        # Keep explicit institution-type intents if user asks.
        if "AIIMS" not in found and "aiims" in low:
            found.append("AIIMS")
        if "JIPMER" not in found and "jipmer" in low:
            found.append("JIPMER")
        if "BHU" not in found and "bhu" in low:
            found.append("BHU")
        if "AMU" not in found and "amu" in low:
            found.append("AMU")
    return found


def _build_sql_context(latest_message: str, recent_chats: list[dict] | None) -> str:
    """
    SQL-generation context: user-only short memory to avoid assistant-text contamination.
    Keeps latest message plus up to last 3 prior user messages.
    """
    latest = (latest_message or "").strip()
    if not latest:
        return ""
    users: list[str] = []
    for m in recent_chats or []:
        if m.get("role") == "user":
            content = (m.get("content") or "").strip()
            if content:
                users.append(content)
    prior = users[-3:]
    if prior:
        lines = [f"Student: {x}" for x in prior]
        lines.append(f"Student (latest message): {latest}")
        return "\n".join(lines)
    return latest


def _is_mcc_target(state_text: str | None) -> bool:
    s = (state_text or "").strip().upper()
    return s == "MCC" or s == "AIQ" or "ALL INDIA" in s


def _enrich_query_with_preferences(
    question: str,
    preferences: dict,
    combined_context: str,
) -> str:
    """
    Provide user profile as context for the LLM to use intelligently.
    
    The profile serves as defaults/context, but:
    - If user explicitly mentions different values in their query, those take precedence
    - LLM decides when profile info is relevant to the current question
    - Profile is provided as context, not forced into every query
    """
    if not preferences:
        return combined_context
    
    # Build profile summary
    profile_parts = []
    
    # Add score/rank info
    score_info = preferences.get("neet_score", {})
    if score_info:
        if score_info.get("type") == "score":
            profile_parts.append(f"NEET score: {score_info.get('value')} marks")
        else:
            profile_parts.append(f"NEET rank: AIR {score_info.get('value')}")
    
    # Add category
    category = preferences.get("category")
    if category:
        profile_parts.append(f"category: {category}")
    
    # Add home state
    home_state = preferences.get("home_state")
    if home_state:
        profile_parts.append(f"home state: {home_state}")
    
    # Add course preference
    course = preferences.get("course")
    if course:
        profile_parts.append(f"course interest: {course.replace('_', '/')}")
    
    # Add sub-category if present
    sub_category = preferences.get("sub_category")
    if sub_category and sub_category != "NONE":
        profile_parts.append(f"sub-category: {sub_category}")
    
    # college_type is normalized to a list when preferences are loaded/saved
    college_types = preferences.get("college_type") or []
    if isinstance(college_types, list) and college_types and "ALL" not in college_types:
        profile_parts.append(f"preferred college types: {', '.join(str(x) for x in college_types)}")
    
    if profile_parts:
        # Provide profile as context for the LLM to use intelligently
        prefix = "[User Profile: " + "; ".join(profile_parts) + "]\n\n"
        return prefix + combined_context
    
    return combined_context


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=16_000)


class AskRequest(BaseModel):
    question: str = Field(..., max_length=16_000)
    """Prior turns only (excludes the current `question`). Client sends full history in order."""
    messages: list[ChatMessage] | None = None


@app.get("/")
def index(request: Request):
    # Compatible with newer Starlette template API.
    return templates.TemplateResponse(request=request, name="index.html")


def _handle_question(question: str, prior_messages: list[dict] | None = None) -> dict:
    request_id = uuid.uuid4().hex[:8]
    started = time.perf_counter()
    question = (question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    data_year = (os.getenv("NEET_DATA_YEAR") or "2025").strip()

    user_id = 5  # Temporary hardcoded user for this layer.
    use_profile_defaults_for_query = False

    logger.info("[%s] Incoming question (chars=%d)", request_id, len(question))
    if prior_messages:
        logger.info("[%s] Client sent prior messages: %d (ignored for memory)", request_id, len(prior_messages))
    openai_client = get_openai_client()
    supabase_client = get_supabase_client()

    context_row = load_user_chat_context(supabase_client, user_id=user_id)
    preferences_json = context_row.get("preferences_json", {})
    recent_chats = context_row.get("recent_chats", [])
    
    # ─────────────────────────────────────────────────────────────────────
    # ONBOARDING FLOW: Check if user needs onboarding (first-time user)
    # ─────────────────────────────────────────────────────────────────────
    
    # Fetch dynamic categories/sub-categories/college_types from database based on user's state
    db_categories = None
    db_sub_categories = None
    db_college_types = None
    home_state = preferences_json.get("home_state")
    category = preferences_json.get("category")
    
    if home_state:
        # Fetch categories for this state from database
        db_cat_list = get_categories_for_state(supabase_client, home_state)
        db_categories = db_categories_to_options(db_cat_list)
        logger.info("[%s] Fetched %d categories for state %s", request_id, len(db_cat_list), home_state)
    
    if home_state and category:
        # Fetch sub-categories for this state+category from database
        db_sub_list = get_sub_categories_for_state_and_category(supabase_client, home_state, category)
        db_sub_categories = db_sub_categories_to_options(db_sub_list)
        logger.info("[%s] Fetched %d sub-categories for state %s, category %s", request_id, len(db_sub_list), home_state, category)

    if home_state:
        db_college_list = get_college_types_for_state(supabase_client, home_state)
        db_college_types = db_college_types_to_options(db_college_list)
        logger.info("[%s] Fetched %d college types for state %s", request_id, len(db_college_list), home_state)
    
    onboarding_status = check_onboarding_status(
        preferences_json,
        db_categories=db_categories,
        db_sub_categories=db_sub_categories,
        db_college_types=db_college_types,
    )

    # `intro` is stripped from persisted preferences on load; the first check can
    # look "incomplete" even when every real field is saved. Resolve that here so
    # we do not run onboarding parsing with current_step=None (which breaks LLM
    # routing and blocks the normal Q&A flow after completion).
    needs_onboarding_processing = not onboarding_status.is_complete
    current_step = onboarding_status.current_step
    already_started_onboarding = len(recent_chats) > 0
    has_some_preferences = bool(preferences_json)

    if needs_onboarding_processing:
        # If intro was already shown in this chat, do not require persisted "intro" key.
        # This keeps DB preferences clean while avoiding intro-loop regressions.
        if current_step == "intro" and already_started_onboarding:
            synthetic = dict(preferences_json or {})
            synthetic["intro"] = "confirmed"
            onboarding_status = check_onboarding_status(
                synthetic,
                db_categories=db_categories,
                db_sub_categories=db_sub_categories,
                db_college_types=db_college_types,
            )
            current_step = onboarding_status.current_step
            # Let LLM decide whether this is only a "continue" reply
            # or already contains NEET score/rank.
            intro_intent = classify_intro_step_intent(
                openai_client,
                question,
                request_id=request_id,
            )
            if current_step == "neet_score" and intro_intent == "continue_onboarding":
                answer = onboarding_status.next_question or (
                    "Great! Let's start with your NEET score or expected AIR rank."
                )
                recent = append_recent_chats(
                    recent_chats,
                    user_text=question,
                    assistant_text=answer,
                )
                save_user_chat_context(
                    supabase_client,
                    user_id=user_id,
                    summary_text=context_row.get("summary_text", ""),
                    recent_chats=recent,
                    preferences_json=preferences_json,
                )
                return {
                    "sql": None,
                    "data": [],
                    "answer": answer,
                    "needs_clarification": True,
                    "data_year": data_year,
                    "onboarding_step": current_step,
                    "onboarding_options": onboarding_status.options,
                }
            if onboarding_status.is_complete:
                needs_onboarding_processing = False

    if needs_onboarding_processing:
        logger.info("[%s] Onboarding step: %s, preferences so far: %s", request_id, current_step, list(preferences_json.keys()))
        
        # Brand new user - show welcome/intro message
        if not already_started_onboarding and current_step == "intro":
            answer = onboarding_status.next_question
            recent = append_recent_chats(
                recent_chats,
                user_text=question,
                assistant_text=answer,
            )
            save_user_chat_context(
                supabase_client,
                user_id=user_id,
                summary_text=context_row.get("summary_text", ""),
                recent_chats=recent,
                preferences_json=preferences_json,
            )
            return {
                "sql": None,
                "data": [],
                "answer": answer,
                "needs_clarification": True,
                "data_year": data_year,
                "onboarding_step": current_step,
                "onboarding_options": onboarding_status.options,
            }
        
        # Returning user with partial preferences but new session - show progress + next question
        if not already_started_onboarding and has_some_preferences:
            # Build progress summary
            progress_parts = []
            if preferences_json.get("neet_score"):
                score_info = preferences_json["neet_score"]
                if score_info.get("type") == "score":
                    progress_parts.append(f"📊 NEET: {score_info.get('value')} marks")
                else:
                    progress_parts.append(f"📊 NEET: AIR {score_info.get('value')}")
            if preferences_json.get("course"):
                progress_parts.append(f"📚 Course: {preferences_json['course'].replace('_', ' ')}")
            if preferences_json.get("home_state"):
                progress_parts.append(f"🏠 State: {preferences_json['home_state']}")
            if preferences_json.get("category"):
                progress_parts.append(f"👤 Category: {preferences_json['category']}")
            
            progress_text = "\n".join(progress_parts)
            answer = f"Welcome back! 👋 Here's what I have so far:\n\n{progress_text}\n\n{onboarding_status.next_question}"
            
            recent = append_recent_chats(
                recent_chats,
                user_text=question,
                assistant_text=answer,
            )
            save_user_chat_context(
                supabase_client,
                user_id=user_id,
                summary_text=context_row.get("summary_text", ""),
                recent_chats=recent,
                preferences_json=preferences_json,
            )
            return {
                "sql": None,
                "data": [],
                "answer": answer,
                "needs_clarification": True,
                "data_year": data_year,
                "onboarding_step": current_step,
                "onboarding_options": onboarding_status.options,
            }
        
        # Process user's response to current onboarding question
        # Use an internal onboarding snapshot that marks intro as confirmed once
        # onboarding chat has already started. This avoids intro-step regressions
        # while still keeping "intro" out of persisted preferences_json.
        prefs_for_processing = dict(preferences_json or {})
        if already_started_onboarding and "intro" not in prefs_for_processing:
            prefs_for_processing["intro"] = "confirmed"

        # LLM-first onboarding interpretation; regex parser as fallback only.
        interpreted = interpret_onboarding_response(
            openai_client,
            current_step=current_step or "",
            user_input=question,
            current_preferences=prefs_for_processing,
            step_options=onboarding_status.options or [],
            request_id=request_id,
        )
        llm_acknowledgement = str(interpreted.get("acknowledgement", "") or "").strip()
        if interpreted.get("action") == "ask_rephrase":
            updated_prefs = prefs_for_processing
            error_msg = interpreted.get("message") or (
                "Could you rephrase that once? I want to capture your preference correctly."
            )
        elif interpreted.get("action") == "apply_update":
            updated_prefs = dict(prefs_for_processing)
            updates = interpreted.get("updates") or {}
            if isinstance(updates, dict):
                updated_prefs.update(updates)
            clear_fields = interpreted.get("clear_fields") or []
            for key in clear_fields:
                updated_prefs.pop(str(key), None)
            # LLM sometimes returns acknowledgement without canonical updates — fill from rule parser.
            error_msg = None
            should_reconcile = _onboarding_field_still_missing(current_step, updated_prefs) and not _is_llm_cross_step_update(
                current_step, updates, clear_fields
            )
            if should_reconcile:
                reconciled, rerr = process_onboarding_response(
                    question,
                    current_step or "",
                    updated_prefs,
                    db_categories=db_categories,
                    db_sub_categories=db_sub_categories,
                    db_college_types=db_college_types,
                )
                if not rerr:
                    updated_prefs = reconciled
                else:
                    # Avoid misleading "saved" copy if we could not parse the step.
                    llm_acknowledgement = ""
                    error_msg = rerr
        else:
            updated_prefs, error_msg = process_onboarding_response(
                question, current_step, prefs_for_processing,
                db_categories=db_categories,
                db_sub_categories=db_sub_categories,
                db_college_types=db_college_types,
            )
        
        if error_msg:
            # Invalid response - ask again
            retry_question = onboarding_status.next_question or "Could you share that once again?"
            answer = f"{error_msg}\n\n{retry_question}"
            recent = append_recent_chats(
                recent_chats,
                user_text=question,
                assistant_text=answer,
            )
            save_user_chat_context(
                supabase_client,
                user_id=user_id,
                summary_text=context_row.get("summary_text", ""),
                recent_chats=recent,
                preferences_json=preferences_json,
            )
            return {
                "sql": None,
                "data": [],
                "answer": answer,
                "needs_clarification": True,
                "data_year": data_year,
                "onboarding_step": current_step,
                "onboarding_options": onboarding_status.options,
            }
        
        # Response accepted - check if there are more steps
        correction_note = updated_prefs.pop("_correction_note", "")
        updated_prefs = normalize_misplaced_course_category(updated_prefs)
        # Re-fetch categories/sub-categories based on updated preferences
        next_db_categories = None
        next_db_sub_categories = None
        next_db_college_types = None
        next_home_state = updated_prefs.get("home_state")
        next_category = updated_prefs.get("category")
        
        if next_home_state:
            next_cat_list = get_categories_for_state(supabase_client, next_home_state)
            next_db_categories = db_categories_to_options(next_cat_list)
        
        if next_home_state and next_category:
            next_sub_list = get_sub_categories_for_state_and_category(supabase_client, next_home_state, next_category)
            next_db_sub_categories = db_sub_categories_to_options(next_sub_list)

        if next_home_state:
            next_college_list = get_college_types_for_state(supabase_client, next_home_state)
            next_db_college_types = db_college_types_to_options(next_college_list)
        
        next_status = check_onboarding_status(
            updated_prefs,
            db_categories=next_db_categories,
            db_sub_categories=next_db_sub_categories,
            db_college_types=next_db_college_types,
        )
        
        if next_status.is_complete:
            # Onboarding complete!
            complete_msg = get_onboarding_complete_message(updated_prefs)
            answer = (
                f"{llm_acknowledgement}\n\n{complete_msg}"
                if llm_acknowledgement
                else complete_msg
            )
            recent = append_recent_chats(
                recent_chats,
                user_text=question,
                assistant_text=answer,
            )
            save_user_chat_context(
                supabase_client,
                user_id=user_id,
                summary_text=context_row.get("summary_text", ""),
                recent_chats=recent,
                preferences_json=updated_prefs,
            )
            logger.info("[%s] Onboarding complete for user %d", request_id, user_id)
            return {
                "sql": None,
                "data": [],
                "answer": answer,
                "needs_clarification": False,
                "data_year": data_year,
                "onboarding_complete": True,
            }
        else:
            # Move to next question with friendly confirmation
            confirmation = correction_note or get_step_confirmation(current_step, updated_prefs)
            lead_parts = [p for p in (llm_acknowledgement, confirmation) if p]
            if lead_parts:
                answer = "\n\n".join(lead_parts) + "\n\n" + (next_status.next_question or "")
            else:
                answer = next_status.next_question or ""
            recent = append_recent_chats(
                recent_chats,
                user_text=question,
                assistant_text=answer,
            )
            save_user_chat_context(
                supabase_client,
                user_id=user_id,
                summary_text=context_row.get("summary_text", ""),
                recent_chats=recent,
                preferences_json=updated_prefs,
            )
            return {
                "sql": None,
                "data": [],
                "answer": answer,
                "needs_clarification": True,
                "data_year": data_year,
                "onboarding_step": next_status.current_step,
                "onboarding_options": next_status.options,
            }
    
    # ─────────────────────────────────────────────────────────────────────
    # PROFILE CONFIRMATION: Before processing queries, confirm profile with user
    # ─────────────────────────────────────────────────────────────────────
    
    if needs_profile_confirmation(preferences_json, recent_chats):
        logger.info("[%s] Profile confirmation needed for user %d", request_id, user_id)
        
        is_confirmation, interpretation = is_profile_confirmation_response(question)
        logger.info("[%s] Profile confirmation response: is_confirmation=%s, interpretation=%s", 
                    request_id, is_confirmation, interpretation)
        
        if interpretation == "use_profile" and is_confirmation:
            # User confirmed to use their profile
            preferences_json["profile_confirmed"] = True
            answer = (
                "Perfect! ✅ Let me find the best options for you...\n\n"
                "I'll search for colleges based on your profile. Give me a moment!"
            )
            recent = append_recent_chats(
                recent_chats,
                user_text=question,
                assistant_text=answer,
            )
            save_user_chat_context(
                supabase_client,
                user_id=user_id,
                summary_text=context_row.get("summary_text", ""),
                recent_chats=recent,
                preferences_json=preferences_json,
            )
            # Now proceed with a profile-based query
            question = "Which colleges can I get based on my profile?"
            use_profile_defaults_for_query = True
            recent_chats = recent
            # Fall through to normal query flow
            
        elif interpretation == "specific_query":
            # User has a specific different query - mark as confirmed and process
            logger.info("[%s] User has specific query, bypassing profile confirmation", request_id)
            preferences_json["profile_confirmed"] = True
            save_user_chat_context(
                supabase_client,
                user_id=user_id,
                summary_text=context_row.get("summary_text", ""),
                recent_chats=recent_chats,
                preferences_json=preferences_json,
            )
            # Fall through to normal query flow with their specific question
            
        else:
            # interpretation == "show_confirmation" - show profile and ask
            answer = get_profile_confirmation_message(preferences_json)
            recent = append_recent_chats(
                recent_chats,
                user_text=question,
                assistant_text=answer,
            )
            save_user_chat_context(
                supabase_client,
                user_id=user_id,
                summary_text=context_row.get("summary_text", ""),
                recent_chats=recent,
                preferences_json=preferences_json,
            )
            return {
                "sql": None,
                "data": [],
                "answer": answer,
                "needs_clarification": True,
                "data_year": data_year,
                "awaiting_profile_confirmation": True,
            }
    
    # ─────────────────────────────────────────────────────────────────────
    # NORMAL QUERY FLOW: Onboarding complete, process user query
    # ─────────────────────────────────────────────────────────────────────
    
    # Extract home state from onboarding preferences (all context from user_chat_context)
    user_home_state = preferences_json.get("home_state")
    question_for_context = _resolve_own_state_phrase(question, user_home_state)
    
    # Build context with user preferences for intelligent query handling
    pref_context = format_preferences_for_context(preferences_json)
    base_context = build_contextual_query(
        question_for_context,
        recent_chats,
    )
    
    # Combine preferences with query context
    if pref_context:
        combined_norm = normalize_user_question(f"{pref_context}\n\n{base_context}")
    else:
        combined_norm = normalize_user_question(base_context)
    
    # Enrich query with defaults from preferences if not explicitly mentioned
    enriched_query = _enrich_query_with_preferences(question, preferences_json, combined_norm)

    logger.info("[%s] Base context for gate:\n%s", request_id, _clip(base_context))
    logger.info("[%s] Enriched query for gate/SQL:\n%s", request_id, _clip(enriched_query))
    gate = gate_user_query(openai_client, enriched_query, request_id=request_id)
    if gate.action == "ask_clarification":
        logger.info("[%s] Query gate: clarification required", request_id)
        clarification = _personalize_clarification(gate.message, user_home_state)
        recent = append_recent_chats(
            recent_chats,
            user_text=question,
            assistant_text=clarification,
        )
        save_user_chat_context(
            supabase_client,
            user_id=user_id,
            summary_text=context_row.get("summary_text", ""),
            recent_chats=recent,
            preferences_json=preferences_json,
        )
        return {
            "sql": None,
            "data": [],
            "answer": clarification,
            "needs_clarification": True,
            "data_year": data_year,
        }
    if gate.action == "reply_without_database":
        logger.info("[%s] Query gate: reply without database", request_id)
        recent = append_recent_chats(
            recent_chats,
            user_text=question,
            assistant_text=gate.message,
        )
        save_user_chat_context(
            supabase_client,
            user_id=user_id,
            summary_text=context_row.get("summary_text", ""),
            recent_chats=recent,
            preferences_json=preferences_json,
        )
        return {
            "sql": None,
            "data": [],
            "answer": gate.message,
            "needs_clarification": False,
            "data_year": data_year,
        }

    # Intent-aware behavior now primarily follows LLM extracted routing fields.
    extracted = gate.extracted or {}
    extracted_home_state = str(extracted.get("home_state_for_query", "")).strip()
    extracted_target_state = str(extracted.get("target_state", "")).strip()
    effective_home_state = extracted_home_state or user_home_state
    norm_home = (resolve_state(effective_home_state) or (effective_home_state or "").strip().upper()) if effective_home_state else ""
    norm_target = (resolve_state(extracted_target_state) or extracted_target_state.strip().upper()) if extracted_target_state else ""
    use_profile_defaults_llm = bool(extracted.get("use_profile_defaults", False))
    explicit_college_types = _extract_college_types_from_text(question_for_context)
    missing_slots = extracted.get("missing_slots", [])
    if not isinstance(missing_slots, list):
        missing_slots = []
    missing_slots = [str(x).strip().lower() for x in missing_slots]

    if extracted.get("needs_confirmation") and extracted.get("confirmation_question"):
        clarification = str(extracted.get("confirmation_question")).strip()
        logger.info("[%s] LLM requested confirmation: %s", request_id, clarification)
        recent = append_recent_chats(
            recent_chats,
            user_text=question,
            assistant_text=clarification,
        )
        save_user_chat_context(
            supabase_client,
            user_id=user_id,
            summary_text=context_row.get("summary_text", ""),
            recent_chats=recent,
            preferences_json=preferences_json,
        )
        return {
            "sql": None,
            "data": [],
            "answer": clarification,
            "needs_clarification": True,
            "data_year": data_year,
        }

    # Policy: category needed only for home-state or MCC/AIQ searches.
    if "category" in missing_slots:
        is_home_state_query = bool(norm_home and norm_target and norm_home == norm_target)
        is_mcc_query = _is_mcc_target(norm_target)
        if not (is_home_state_query or is_mcc_query):
            missing_slots = [s for s in missing_slots if s != "category"]
            logger.info(
                "[%s] Removed 'category' from missing slots by policy (home=%s target=%s)",
                request_id,
                norm_home,
                norm_target,
            )

    if missing_slots:
        slot_map = {
            "state_or_mcc": "state or MCC",
            "category": "category",
            "college_type": "college type (government/private/deemed)",
            "neet_metric": "NEET score (marks) or All India Rank",
        }
        human_missing = [slot_map.get(s, s) for s in missing_slots]
        clarification = (
            "Got it — before I run the search, please confirm "
            + ", ".join(human_missing)
            + "."
        )
        logger.info("[%s] LLM reported missing slots: %s", request_id, missing_slots)
        recent = append_recent_chats(
            recent_chats,
            user_text=question,
            assistant_text=clarification,
        )
        save_user_chat_context(
            supabase_client,
            user_id=user_id,
            summary_text=context_row.get("summary_text", ""),
            recent_chats=recent,
            preferences_json=preferences_json,
        )
        return {
            "sql": None,
            "data": [],
            "answer": clarification,
            "needs_clarification": True,
            "data_year": data_year,
        }

    # Use profile defaults only when LLM says so (or explicit profile mode).
    use_profile_defaults = use_profile_defaults_for_query or use_profile_defaults_llm
    user_category = preferences_json.get("category") if use_profile_defaults else None
    # Important: do NOT auto-apply stored college_type on normal queries.
    # It should only come from the current message unless user explicitly
    # asked to use full profile defaults.
    raw_ct = preferences_json.get("college_type")
    if use_profile_defaults:
        if isinstance(raw_ct, list):
            user_college_types = [str(x).strip() for x in raw_ct if str(x).strip()]
        elif isinstance(raw_ct, str) and raw_ct.strip():
            user_college_types = [raw_ct.strip()]
        else:
            user_college_types = []
    else:
        user_college_types = []
    if explicit_college_types:
        user_college_types = explicit_college_types
        logger.info(
            "[%s] Explicit college type in current query overrides profile: %s",
            request_id,
            explicit_college_types,
        )
    elif not use_profile_defaults_for_query:
        logger.info("[%s] No explicit college type in current query; skipping profile college_type default", request_id)

    sql_context = _build_sql_context(question_for_context, recent_chats)
    logger.info("[%s] SQL focused context:\n%s", request_id, _clip(sql_context))
    
    sql = generate_sql(
        openai_client,
        sql_context,
        user_home_state=effective_home_state,
        user_category=user_category,
        user_college_types=user_college_types,
        extracted=gate.extracted,
        request_id=request_id,
    )
    logger.info("[%s] Generated SQL: %s", request_id, sql)
    data = execute_neet_query(supabase_client, sql)
    logger.info("[%s] Rows returned: %d", request_id, len(data))
    preview = data[:3] if isinstance(data, list) else []
    logger.info("[%s] Data preview (first 3 rows): %s", request_id, _clip(json.dumps(preview, ensure_ascii=True)))
    answer = generate_counsellor_answer(
        openai_client, question_for_context, data, data_year=data_year, request_id=request_id
    )
    recent = append_recent_chats(
        recent_chats,
        user_text=question,
        assistant_text=answer,
    )
    save_user_chat_context(
        supabase_client,
        user_id=user_id,
        summary_text=context_row.get("summary_text", ""),
        recent_chats=recent,
        preferences_json=preferences_json,
    )
    logger.info(
        "[%s] Answer generated (chars=%d) in %.2fs",
        request_id,
        len(answer),
        time.perf_counter() - started,
    )

    return {
        "sql": sql,
        "data": data,
        "answer": answer,
        "needs_clarification": False,
        "data_year": data_year,
    }


@app.post("/ask")
def ask_json(payload: AskRequest):
    try:
        prior = [m.model_dump() for m in payload.messages] if payload.messages else None
        return JSONResponse(_handle_question(payload.question, prior))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled /ask error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask-form")
def ask_form(question: str = Form(...)):
    try:
        return JSONResponse(_handle_question(question))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled /ask-form error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/suggestions")
def suggestions():
    return {"suggestions": DEFAULT_SUGGESTIONS}


@app.get("/chat/context")
def chat_context():
    user_id = 5  # Temporary hardcoded user for this layer.
    try:
        supabase_client = get_supabase_client()
        context_row = load_user_chat_context(supabase_client, user_id=user_id)
        return {
            "user_id": user_id,
            "recent_chats": context_row.get("recent_chats", []),
            "summary_text": context_row.get("summary_text", ""),
        }
    except Exception as exc:
        logger.exception("Failed to load chat context")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat/clear")
def clear_chat():
    user_id = 5  # Temporary hardcoded user for this layer.
    try:
        supabase_client = get_supabase_client()
        clear_user_chat_context(supabase_client, user_id=user_id)
        return {"ok": True}
    except Exception as exc:
        logger.exception("Failed to clear chat context")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

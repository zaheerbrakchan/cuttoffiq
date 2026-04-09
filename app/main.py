from __future__ import annotations

import logging
import os
import re
import time
import uuid
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
from app.services.query_normalization import normalize_user_question
from app.services.query_validation import gate_user_query
from app.services.sql_generator import (
    generate_counsellor_answer,
    generate_sql,
)
from app.services.supabase_service import (
    execute_neet_query,
    get_supabase_client,
    get_categories_for_state,
    get_sub_categories_for_state_and_category,
)
from app.services.onboarding_service import (
    check_onboarding_status,
    process_onboarding_response,
    get_onboarding_complete_message,
    get_step_confirmation,
    format_preferences_for_context,
    db_categories_to_options,
    db_sub_categories_to_options,
    get_profile_confirmation_message,
    is_profile_confirmation_response,
    needs_profile_confirmation,
)

# Fixed chips: suggest queries that work well with user profile context.
DEFAULT_SUGGESTIONS: list[str] = [
    "Which colleges can I get in my state?",
    "Show me government colleges in MCC/All India",
    "What are my options in private colleges?",
    "Top colleges within my rank range",
    "Best BDS options for me",
]

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("neet_assistant")

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
    
    # Add college type preferences
    college_types = preferences.get("college_type", [])
    if college_types and isinstance(college_types, list):
        if "ALL" not in college_types:
            profile_parts.append(f"preferred college types: {', '.join(college_types)}")
    
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
    
    # Fetch dynamic categories/sub-categories from database based on user's state
    db_categories = None
    db_sub_categories = None
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
    
    onboarding_status = check_onboarding_status(
        preferences_json,
        db_categories=db_categories,
        db_sub_categories=db_sub_categories,
    )
    
    if not onboarding_status.is_complete:
        current_step = onboarding_status.current_step
        logger.info("[%s] Onboarding step: %s, preferences so far: %s", request_id, current_step, list(preferences_json.keys()))
        
        # Check if we already showed the welcome message (by checking recent_chats)
        already_started_onboarding = len(recent_chats) > 0
        has_some_preferences = bool(preferences_json)
        
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
        updated_prefs, error_msg = process_onboarding_response(
            question, current_step, preferences_json,
            db_categories=db_categories,
            db_sub_categories=db_sub_categories,
        )
        
        if error_msg:
            # Invalid response - ask again
            answer = error_msg + "\n\n" + onboarding_status.next_question
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
        # Re-fetch categories/sub-categories based on updated preferences
        next_db_categories = None
        next_db_sub_categories = None
        next_home_state = updated_prefs.get("home_state")
        next_category = updated_prefs.get("category")
        
        if next_home_state:
            next_cat_list = get_categories_for_state(supabase_client, next_home_state)
            next_db_categories = db_categories_to_options(next_cat_list)
        
        if next_home_state and next_category:
            next_sub_list = get_sub_categories_for_state_and_category(supabase_client, next_home_state, next_category)
            next_db_sub_categories = db_sub_categories_to_options(next_sub_list)
        
        next_status = check_onboarding_status(
            updated_prefs,
            db_categories=next_db_categories,
            db_sub_categories=next_db_sub_categories,
        )
        
        if next_status.is_complete:
            # Onboarding complete!
            answer = get_onboarding_complete_message(updated_prefs)
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
            confirmation = get_step_confirmation(current_step, updated_prefs)
            if confirmation:
                answer = f"{confirmation}\n\n{next_status.next_question}"
            else:
                # No confirmation needed, just show next question
                answer = next_status.next_question
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

    gate = gate_user_query(openai_client, enriched_query)
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

    # Get user preferences for better filtering
    user_category = preferences_json.get("category")
    user_college_types = preferences_json.get("college_type", [])
    
    sql = generate_sql(
        openai_client,
        enriched_query,
        user_home_state=user_home_state,
        user_category=user_category,
        user_college_types=user_college_types,
    )
    logger.info("[%s] Generated SQL: %s", request_id, sql)
    data = execute_neet_query(supabase_client, sql)
    logger.info("[%s] Rows returned: %d", request_id, len(data))
    answer = generate_counsellor_answer(
        openai_client, enriched_query, data, data_year=data_year
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

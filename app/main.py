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
from app.services.supabase_service import execute_neet_query, get_supabase_client
from app.services.user_profile_service import load_user_preferences
from app.services.user_profile_service import extract_home_state

# Fixed chips: include category + state/MCC + college type so the query gate allows a DB search (client requirement).
DEFAULT_SUGGESTIONS: list[str] = [
    "Top MBBS colleges in MCC under rank 5000 for GENERAL category in government colleges",
    "my rank is 4356, category OBC, which government colleges can I get in Karnataka",
    "Top colleges in Andhra for NEET score above 580, GENERAL category, private colleges",
    "Best BDS colleges in Jammu and Kashmir under 20000 AIR rank, EWS category, government colleges",
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
                "**state (or MCC/all India)**",
                f"**home state ({user_home_state}) or another state/MCC (all India)**",
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
    prefs = load_user_preferences(supabase_client, user_id=user_id)
    user_home_state = extract_home_state(prefs)
    question_for_context = _resolve_own_state_phrase(question, user_home_state)
    combined_norm = normalize_user_question(
        build_contextual_query(
            question_for_context,
            context_row.get("recent_chats", []),
        )
    )

    gate = gate_user_query(openai_client, combined_norm)
    if gate.action == "ask_clarification":
        logger.info("[%s] Query gate: clarification required", request_id)
        clarification = _personalize_clarification(gate.message, user_home_state)
        recent = append_recent_chats(
            context_row.get("recent_chats", []),
            user_text=question,
            assistant_text=clarification,
        )
        save_user_chat_context(
            supabase_client,
            user_id=user_id,
            summary_text=context_row.get("summary_text", ""),
            recent_chats=recent,
            preferences_json=context_row.get("preferences_json", {}),
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
            context_row.get("recent_chats", []),
            user_text=question,
            assistant_text=gate.message,
        )
        save_user_chat_context(
            supabase_client,
            user_id=user_id,
            summary_text=context_row.get("summary_text", ""),
            recent_chats=recent,
            preferences_json=context_row.get("preferences_json", {}),
        )
        return {
            "sql": None,
            "data": [],
            "answer": gate.message,
            "needs_clarification": False,
            "data_year": data_year,
        }

    sql = generate_sql(
        openai_client,
        combined_norm,
        user_home_state=user_home_state,
    )
    logger.info("[%s] Generated SQL: %s", request_id, sql)
    data = execute_neet_query(supabase_client, sql)
    logger.info("[%s] Rows returned: %d", request_id, len(data))
    answer = generate_counsellor_answer(
        openai_client, combined_norm, data, data_year=data_year
    )
    recent = append_recent_chats(
        context_row.get("recent_chats", []),
        user_text=question,
        assistant_text=answer,
    )
    save_user_chat_context(
        supabase_client,
        user_id=user_id,
        summary_text=context_row.get("summary_text", ""),
        recent_chats=recent,
        preferences_json=context_row.get("preferences_json", {}),
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

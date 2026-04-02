from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.services.openai_service import get_openai_client
from app.services.sql_generator import (
    generate_counsellor_answer,
    generate_sql,
)
from app.services.supabase_service import execute_neet_query, get_supabase_client

# Fixed chips shown in the UI (see GET /suggestions).
DEFAULT_SUGGESTIONS: list[str] = [
    "Top MBBS colleges in MCC under rank 5000",
    "my rank is 4356 which colleges i can get in Karnataka",
    "Top Colleges in Andra for the neet score above 580",
    "Best BDS colleges in jammu and kashmir under 20000 AIR rank",
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


class AskRequest(BaseModel):
    question: str


@app.get("/")
def index(request: Request):
    # Compatible with newer Starlette template API.
    return templates.TemplateResponse(request=request, name="index.html")


def _handle_question(question: str) -> dict:
    request_id = uuid.uuid4().hex[:8]
    started = time.perf_counter()
    question = (question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    logger.info("[%s] Incoming question: %s", request_id, question)
    openai_client = get_openai_client()
    supabase_client = get_supabase_client()

    sql = generate_sql(openai_client, question)
    logger.info("[%s] Generated SQL: %s", request_id, sql)
    data = execute_neet_query(supabase_client, sql)
    logger.info("[%s] Rows returned: %d", request_id, len(data))
    answer = generate_counsellor_answer(openai_client, question, data)
    logger.info(
        "[%s] Answer generated (chars=%d) in %.2fs",
        request_id,
        len(answer),
        time.perf_counter() - started,
    )

    return {"sql": sql, "data": data, "answer": answer}


@app.post("/ask")
def ask_json(payload: AskRequest):
    try:
        return JSONResponse(_handle_question(payload.question))
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

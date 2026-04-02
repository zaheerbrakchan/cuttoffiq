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
from app.services.supabase_service import (
    execute_neet_query,
    get_suggestion_context,
    get_supabase_client,
)

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
    try:
        supabase_client = get_supabase_client()
        context = get_suggestion_context(supabase_client)
        logger.info(
            "Suggestion context loaded: states=%d categories=%d courses=%d rounds=%d",
            len(context.get("states", [])),
            len(context.get("categories", [])),
            len(context.get("courses", [])),
            len(context.get("rounds", [])),
        )

        states = context.get("states", [])
        categories = context.get("categories", [])
        courses = context.get("courses", [])
        rounds = context.get("rounds", [])

        state_1 = states[0] if states else "KARNATAKA"
        state_2 = states[1] if len(states) > 1 else state_1
        category_1 = categories[0] if categories else "GENERAL"
        category_2 = categories[1] if len(categories) > 1 else category_1
        course_1 = courses[0] if courses else "MBBS"
        round_1 = rounds[0] if rounds else "R1"

        return {
            "suggestions": [
                f"Best {course_1} colleges in {state_1} under rank 5000",
                f"Top government colleges for {category_1} in {state_2}",
                f"Lowest fee {course_1} colleges in {state_1} for rank 12000",
                f"{category_2} cutoff trend in {state_2} for round {round_1}",
                f"Top colleges in {state_1} with score above 620",
            ]
        }
    except Exception:
        logger.exception("Suggestion generation failed, using fallback")
        # Non-critical endpoint fallback.
        return {
            "suggestions": [
                "Best MBBS colleges in Karnataka under rank 5000",
                "Top government colleges for OBC in Maharashtra",
                "Lowest fee MBBS options in Tamil Nadu for rank 12000",
                "General category MBBS cutoffs in Rajasthan round 2",
                "Top colleges in MCC counselling under rank 3000",
            ]
        }

# NEET Counselling Assistant (FastAPI + Supabase + OpenAI)

Single-service app with:
- FastAPI backend
- Jinja2 + Tailwind web UI
- OpenAI for NL -> SQL and counselling explanation
- Supabase RPC execution via `execute_neet_query`

## Setup

1. Create and activate virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:

```bash
copy .env.example .env
```

Then set:
- `OPENAI_API_KEY`
- `SUPABASE_URL` (HTTP project URL)
- `SUPABASE_KEY`
- `NEET_DATA_YEAR` (optional, default `2025`; shown in counsellor notes after DB results)

Incomplete questions (e.g. only a NEET score) trigger a **clarification** response asking for **category**, **state or MCC/all India**, and **government vs private vs deemed** before running SQL.

### Chat mode

The web UI is a **ChatGPT-style thread**. Active memory is stored in the `user_chat_context` table and loaded from backend endpoints. Use **Clear chat** to reset server-side context for the current user. Runtime context currently uses the **latest 8 conversation messages** (summary generation is intentionally disabled for token efficiency).

## Run

```bash
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Endpoints

- `GET /` -> UI
- `POST /ask` -> JSON question input
- `POST /ask-form` -> form input support
- `GET /suggestions` -> optional sample query suggestions

## Deploy on Railway

This repo is ready for Railway with `Procfile` + `railway.json`.

1. Push this project to GitHub.
2. In Railway: **New Project -> Deploy from GitHub Repo**.
3. Select this repository.
4. Add environment variables in Railway:
   - `OPENAI_API_KEY`
   - `SUPABASE_URL` (must be `https://...supabase.co`)
   - `SUPABASE_KEY`
   - `NEET_DATA_YEAR` (optional)
5. Deploy.

Railway start command used:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

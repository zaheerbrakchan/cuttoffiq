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
- `SUPABASE_KEY` (service key)

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
5. Deploy.

Railway start command used:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

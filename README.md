# AI Call Server (Render)

## Overview
This Flask app handles AI call logic: register customers, simulate calls, receive incoming call webhooks, produce AI responses and TTS audio.

## Prepare
1. Create a GitHub repo and push this folder.
2. In Render create a new Web Service (Python):
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2`
3. In Render dashboard, set environment variables (Settings -> Environment):
   - OPENAI_API_KEY = sk-...
   - SERVER_API_KEY = a-strong-random-string
   - OPENAI_CHAT_MODEL = gpt-4o-mini
   - OPENAI_TTS_MODEL = tts-1
   - DATABASE_FILE = ai_calls.db

## Endpoints
- `GET /` health
- `GET /status` (requires API key)
- `POST /register-customer` (JSON {username, phone, agent_prompt}) — require API key
- `POST /simulate-call` (JSON {username, from, message}) — require API key
- `POST /call` (incoming webhook) — require API key
- `GET /calls?username=...` — require API key

## Dashboard Integration (example)
Use `Authorization: Bearer <SERVER_API_KEY>` header in all requests from your dashboard on VPS.

Example:

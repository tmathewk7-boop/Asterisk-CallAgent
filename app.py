import os
import sqlite3
import json
import tempfile
import requests
from flask import Flask, request, jsonify, send_file, abort
from dotenv import load_dotenv
from datetime import datetime

# Load env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1")
SERVER_API_KEY = os.getenv("SERVER_API_KEY", None)
DB_FILE = os.getenv("DATABASE_FILE", "ai_calls.db")

if SERVER_API_KEY is None:
    raise RuntimeError("SERVER_API_KEY must be set in environment")

app = Flask(__name__)

# ----------------------------
# DB helpers
# ----------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    # customers table: username, phone, agent_prompt (JSON/text)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        phone TEXT,
        agent_prompt TEXT,
        created_at TEXT
    )
    """)
    # calls table to store call logs
    cur.execute("""
    CREATE TABLE IF NOT EXISTS calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        caller TEXT,
        direction TEXT,
        transcription TEXT,
        ai_response TEXT,
        summary TEXT,
        raw_json TEXT,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()

def db_conn():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

init_db()

# ----------------------------
# Auth decorator (API key)
# ----------------------------
from functools import wraps
def require_api_key(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        key = None
        # prefer Authorization header Bearer <key>
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            key = auth.split(" ", 1)[1].strip()
        if not key:
            key = request.headers.get("X-API-KEY") or request.args.get("api_key")
        if key != SERVER_API_KEY:
            return jsonify({"error": "unauthorized"}), 401
        return fn(*a, **kw)
    return wrapper

# ----------------------------
# OpenAI helpers
# ----------------------------
def openai_chat(prompt, system=None):
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        messages = []
        if system:
            messages.append({"role":"system", "content": system})
        messages.append({"role":"user", "content": prompt})
        payload = {
            "model": OPENAI_CHAT_MODEL,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 512
        }
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"], data
    except Exception as e:
        return f"ERROR: {e}", {"error": str(e)}

def openai_tts(text):
    """
    Returns path to MP3 temporary file, or None on error.
    """
    try:
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": OPENAI_TTS_MODEL,
            "voice": "alloy",
            "input": text,
            "response_format": "mp3"
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(r.content)
        tmp.close()
        return tmp.name
    except Exception as e:
        app.logger.exception("TTS failed")
        return None

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return {"status":"ai-call-server", "time": datetime.utcnow().isoformat()}

@app.route("/status")
@require_api_key
def status():
    # lightweight status
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as c FROM calls")
    calls_count = cur.fetchone()["c"]
    conn.close()
    return {"status":"ok","calls_stored": calls_count}

# Register or update a customer
@app.route("/register-customer", methods=["POST"])
@require_api_key
def register_customer():
    payload = request.json or {}
    username = payload.get("username")
    phone = payload.get("phone")
    agent_prompt = payload.get("agent_prompt", "")
    if not username:
        return jsonify({"error":"username required"}), 400

    conn = db_conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    try:
        cur.execute("INSERT OR REPLACE INTO customers (username, phone, agent_prompt, created_at) VALUES ((SELECT username FROM customers WHERE username=?), ?, ?, ?)",
                    (username, phone, agent_prompt, now))
        conn.commit()
    except Exception as e:
        # fallback insert/update manual
        cur.execute("SELECT id FROM customers WHERE username=?", (username,))
        row = cur.fetchone()
        if row:
            cur.execute("UPDATE customers SET phone=?, agent_prompt=?, created_at=? WHERE username=?", (phone, agent_prompt, now, username))
        else:
            cur.execute("INSERT INTO customers (username, phone, agent_prompt, created_at) VALUES (?, ?, ?, ?)", (username, phone, agent_prompt, now))
        conn.commit()
    conn.close()
    return jsonify({"status":"ok", "username": username})

# List calls (filter by username optional)
@app.route("/calls", methods=["GET"])
@require_api_key
def list_calls():
    username = request.args.get("username")
    conn = db_conn()
    cur = conn.cursor()
    if username:
        cur.execute("SELECT * FROM calls WHERE username=? ORDER BY id DESC LIMIT 200", (username,))
    else:
        cur.execute("SELECT * FROM calls ORDER BY id DESC LIMIT 200")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify({"calls": rows})

# Simulate a call (useful for testing before buying DID)
@app.route("/simulate-call", methods=["POST"])
@require_api_key
def simulate_call():
    payload = request.json or {}
    username = payload.get("username")
    caller = payload.get("from", "test-caller")
    message = payload.get("message", "")
    # lookup prompt for customer
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT agent_prompt FROM customers WHERE username=?", (username,))
    row = cur.fetchone()
    agent_prompt = row["agent_prompt"] if row else ""
    # simple system prompt fallback
    system_prompt = agent_prompt or "You are a helpful customer-facing AI assistant. Keep replies short and professional."
    ai_text, raw = openai_chat(message, system=system_prompt)
    # persist call
    now = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO calls (username, caller, direction, transcription, ai_response, raw_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (username, caller, "inbound", message, ai_text, json.dumps(raw), now))
    conn.commit()
    call_id = cur.lastrowid
    conn.close()
    return jsonify({"call_id": call_id, "ai_response": ai_text})

# Webhook endpoint for voice providers (Twilio/SignalWire)
@app.route("/call", methods=["POST"])
@require_api_key
def incoming_call():
    """
    This endpoint expects provider to POST JSON with:
    {
      "username": "<customer assigned to DID>",
      "from": "<caller number>",
      "audio_url": "<optional: recorded audio url or streaming token>",
      "transcription": "<optional: pre-transcribed text>",
      ...
    }
    If transcription provided we use it; otherwise you can implement logic to download audio and send to Whisper (TODO).
    """
    payload = request.json or {}
    username = payload.get("username")
    caller = payload.get("from")
    transcription = payload.get("transcription")
    message = transcription or payload.get("message") or ""
    # get agent prompt
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT agent_prompt FROM customers WHERE username=?", (username,))
    row = cur.fetchone()
    agent_prompt = row["agent_prompt"] if row else ""
    system_prompt = agent_prompt or "You are a friendly, concise assistant."
    ai_text, raw = openai_chat(message, system=system_prompt)
    now = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO calls (username, caller, direction, transcription, ai_response, raw_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (username, caller, "inbound", message, ai_text, json.dumps(raw), now))
    conn.commit()
    call_id = cur.lastrowid
    conn.close()
    # Optionally generate TTS and return URL or serve audio stream
    tts_path = openai_tts(ai_text)
    if tts_path:
        # send file back in response as attachment (provider may expect URL instead)
        return send_file(tts_path, mimetype="audio/mpeg", as_attachment=False, download_name=f"response_{call_id}.mp3")
    else:
        return jsonify({"ai_response": ai_text})

# Simple healthcheck
@app.route("/health")
def health():
    return jsonify({"status":"ok", "time": datetime.utcnow().isoformat()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

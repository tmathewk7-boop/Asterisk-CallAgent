import os
import base64
import json
import asyncio
import audioop
import httpx
import datetime
from collections import defaultdict
from typing import Optional
from pydantic import BaseModel

# --- IMPORTS ---
from groq import Groq
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ---------- Configuration ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
PORT = int(os.getenv("PORT", 8000))
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://your-app.onrender.com")
SETTINGS_FILE = "user_settings.json"

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ---------- DATABASE (In-Memory + File Persistence) ----------
call_db = {}      
transcripts = defaultdict(list) 
media_ws_map = {}
silence_counter = defaultdict(int)
full_sentence_buffer = defaultdict(bytearray)
active_call_config = {} 

# --- PERSISTENT SETTINGS STORAGE ---
# Structure: { "+13065551234": { "system_prompt": "...", "greeting": "..." } }
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_settings(data):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)

# Load on startup
USER_CONFIGS = load_settings()

# Default if user hasn't customized yet
DEFAULT_CONFIG = {
    "system_prompt": "You are a helpful assistant. Be concise.",
    "greeting": "Hello! I am your AI assistant."
}

# ---------- API MODELS ----------
class AgentSettings(BaseModel):
    phone_number: str
    system_prompt: str
    greeting: str

# ---------------- API FOR DASHBOARD ----------------

@app.get("/api/calls/{target_number}")
async def get_calls_for_client(target_number: str):
    """
    Returns calls specifically for the logged-in user's phone number.
    """
    filtered_calls = []
    # Normalize formatting if needed, strictly matching for now
    for call in call_db.values():
        if call.get("system_number") == target_number:
            filtered_calls.append(call)
    filtered_calls.reverse()
    return filtered_calls

@app.get("/api/settings/{target_number}")
async def get_settings(target_number: str):
    """
    Returns the current AI settings for a user.
    """
    return USER_CONFIGS.get(target_number, DEFAULT_CONFIG)

@app.post("/api/settings")
async def update_settings(settings: AgentSettings):
    """
    Updates the AI Prompt and Greeting for a specific user.
    """
    USER_CONFIGS[settings.phone_number] = {
        "system_prompt": settings.system_prompt,
        "greeting": settings.greeting
    }
    save_settings(USER_CONFIGS)
    return {"status": "success", "message": "Settings saved"}

# ---------------- TwiML endpoint ----------------
@app.post("/twilio/incoming")
async def twilio_incoming(request: Request):
    form = await request.form()
    
    call_sid = form.get("CallSid")
    caller_number = form.get("From", "Unknown")
    system_number = form.get("To", "") 
    
    # 1. LOAD USER CONFIG
    # Check if this number has custom settings, otherwise use default
    config = USER_CONFIGS.get(system_number, DEFAULT_CONFIG)
    
    # Save config for this specific call session
    active_call_config[call_sid] = config

    # 2. LOG THE CALL
    city = form.get("FromCity", "")
    location = f"{city}, {form.get('FromState', '')}" if city else "Unknown"
    
    call_db[call_sid] = {
        "sid": call_sid,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "number": caller_number,
        "system_number": system_number,
        "location": location,
        "status": "Live",
        "summary": None
    }
    
    print(f"New Call: {caller_number} -> {system_number}")

    host = PUBLIC_URL
    if host.startswith("https://"): host = host.replace("https://", "wss://")
    elif host.startswith("http://"): host = host.replace("http://", "ws://")
    stream_url = f"{host}/media-ws"
    
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{stream_url}" />
    </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")

# ---------------- Media WebSocket ----------------
@app.websocket("/media-ws")
async def media_ws_endpoint(ws: WebSocket):
    await ws.accept()
    call_sid = None
    stream_sid = None

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            event = data.get("event")

            if event == "start":
                call_sid = data["start"].get("callSid")
                stream_sid = data["start"].get("streamSid")
                media_ws_map[call_sid] = ws
                
                # GET CUSTOM GREETING
                config = active_call_config.get(call_sid, DEFAULT_CONFIG)
                greeting = config.get("greeting", "Hello.")
                
                await send_deepgram_tts(ws, stream_sid, greeting)
                continue

            if event == "media":
                payload_b64 = data["media"]["payload"]
                chunk = base64.b64decode(payload_b64)
                if call_sid:
                    await process_audio_stream(call_sid, stream_sid, chunk)
                continue

            if event == "stop":
                break
    except Exception as e:
        pass
    finally:
        if call_sid:
            if call_sid in call_db: call_db[call_sid]["status"] = "Ended"
            asyncio.create_task(generate_call_summary(call_sid))
            if call_sid in media_ws_map: del media_ws_map[call_sid]
            if call_sid in full_sentence_buffer: del full_sentence_buffer[call_sid]
            if call_sid in active_call_config: del active_call_config[call_sid]

# ---------------- LOGIC ----------------
async def process_audio_stream(call_sid: str, stream_sid: str, audio_ulaw: bytes):
    pcm16 = audioop.ulaw2lin(audio_ulaw, 2)
    rms = audioop.rms(pcm16, 2)
    
    if rms > 600:
        silence_counter[call_sid] = 0
        full_sentence_buffer[call_sid].extend(audio_ulaw)
    else:
        silence_counter[call_sid] += 1

    if silence_counter[call_sid] >= 20: 
        if len(full_sentence_buffer[call_sid]) > 2000: 
            complete_audio = bytes(full_sentence_buffer[call_sid])
            full_sentence_buffer[call_sid].clear()
            silence_counter[call_sid] = 0
            asyncio.create_task(handle_complete_sentence(call_sid, stream_sid, complete_audio))
        else:
            if len(full_sentence_buffer[call_sid]) > 0: full_sentence_buffer[call_sid].clear()
            silence_counter[call_sid] = 0

async def handle_complete_sentence(call_sid: str, stream_sid: str, raw_ulaw: bytes):
    try:
        transcript = await transcribe_raw_audio(raw_ulaw)
        if not transcript: return

        print(f"[{call_sid}] User: {transcript}")
        transcripts[call_sid].append(f"User: {transcript}")

        # GET CUSTOM PROMPT
        config = active_call_config.get(call_sid, DEFAULT_CONFIG)
        custom_prompt = config.get("system_prompt", "You are a helpful assistant.")

        response_text = await generate_smart_response(transcript, custom_prompt)
        transcripts[call_sid].append(f"AI: {response_text}")
        
        ws = media_ws_map.get(call_sid)
        if ws:
            await send_deepgram_tts(ws, stream_sid, response_text)

    except Exception as e:
        print(f"Error: {e}")

# ---------------- SUMMARIZER ----------------
async def generate_call_summary(call_sid: str):
    if not groq_client or call_sid not in transcripts: return
    full_text = "\n".join(transcripts[call_sid])
    if not full_text: return
    try:
        loop = asyncio.get_running_loop()
        completion = await loop.run_in_executor(None, lambda: groq_client.chat.completions.create(
            messages=[{"role": "system", "content": "Summarize this call in 6 words."}, {"role": "user", "content": full_text}],
            model="llama-3.1-8b-instant", max_tokens=20
        ))
        summary = completion.choices[0].message.content.strip()
        if call_sid in call_db: call_db[call_sid]["summary"] = summary
    except Exception: pass

# ---------------- HELPERS ----------------
async def transcribe_raw_audio(raw_ulaw):
    if not DEEPGRAM_API_KEY: return None
    try:
        url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&encoding=mulaw&sample_rate=8000"
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/basic"}
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, content=raw_ulaw)
        data = response.json()
        if 'results' in data: return data['results']['channels'][0]['alternatives'][0]['transcript']
        return None
    except Exception: return None

async def generate_smart_response(user_text, system_prompt):
    if not groq_client: return "Error."
    try:
        # Force conciseness on top of user prompt
        full_prompt = f"{system_prompt} Keep your answer to 1 short sentence (max 15 words)."
        loop = asyncio.get_running_loop()
        completion = await loop.run_in_executor(None, lambda: groq_client.chat.completions.create(
            messages=[{"role": "system", "content": full_prompt}, {"role": "user", "content": user_text}],
            model="llama-3.1-8b-instant", max_tokens=60
        ))
        return completion.choices[0].message.content.strip()
    except Exception: return "I didn't catch that."

async def send_deepgram_tts(ws: WebSocket, stream_sid: str, text: str):
    if not DEEPGRAM_API_KEY: return
    url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=mulaw&sample_rate=8000&container=none"
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "application/json"}
    payload = {"text": text}
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                async for chunk in response.aiter_bytes():
                    if chunk:
                        payload = base64.b64encode(chunk).decode("ascii")
                        await ws.send_json({"event": "media", "streamSid": stream_sid, "media": {"payload": payload}})
                        await asyncio.sleep(0.001)
    except Exception: pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)

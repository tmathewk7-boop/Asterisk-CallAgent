import os
import base64
import json
import asyncio
import audioop
import httpx
import datetime
from collections import defaultdict

# --- IMPORTS ---
from groq import Groq
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ==========================================
#       📝 CLIENT CONFIGURATION AREA
# ==========================================
# This is where you register your 3 phone numbers.
# The server will automatically switch personalities based on which number is dialed.

CLIENT_CONFIG = {
    # CLIENT 1: Example - Dental Clinic
    "+188254352488": {
        "name": "Thomas Mathew",
        "prompt": "You are a receptionist for Thomas Mathew the Lawyer. Be friendly. Try to schedule an appointment.",
    },
    
    # CLIENT 2: Example - Pizza Place
    "+16475550200": {
        "name": "Tony's Pizza",
        "prompt": "You are a pizza order taker. The special today is pepperoni. Keep it short.",
    },

    # CLIENT 3: Example - Tech Support
    "+15551234567": {
        "name": "IT Support",
        "prompt": "You are a tired IT support agent. Ask if they have tried turning it off and on again.",
    }
}

# Fallback (If a number calls that isn't in the list)
DEFAULT_CLIENT = {
    "name": "General AI",
    "prompt": "You are a helpful assistant. Be concise."
}
# ==========================================

# ---------- Env Vars ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
PORT = int(os.getenv("PORT", 8000))
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://your-app.onrender.com")

if GROQ_API_KEY: groq_client = Groq(api_key=GROQ_API_KEY)
else: groq_client = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ---------- STATE & DATABASE ----------
call_db = {}      
transcripts = defaultdict(list) 
media_ws_map = {}
silence_counter = defaultdict(int)
full_sentence_buffer = defaultdict(bytearray)

# Tracks which client config belongs to which active call
active_call_sessions = {} 

# ---------------- API FOR DASHBOARD ----------------
@app.get("/api/calls/{target_number}")
async def get_calls_for_client(target_number: str):
    """
    Your Dashboard .exe calls this.
    It filters the database and returns ONLY calls made to 'target_number'.
    """
    filtered_calls = []
    for call in call_db.values():
        # We check the 'system_number' (the number the user dialed)
        if call.get("system_number") == target_number:
            filtered_calls.append(call)
            
    # Sort newest first
    filtered_calls.reverse()
    return filtered_calls

# ---------------- TwiML endpoint ----------------
@app.post("/twilio/incoming")
async def twilio_incoming(request: Request):
    form = await request.form()
    
    call_sid = form.get("CallSid")
    caller_number = form.get("From", "Unknown")
    
    # 1. IDENTIFY THE CLIENT
    # "To" is the number the person dialed (Your Twilio Number)
    dialed_number = form.get("To", "") 
    
    # 2. LOAD THE PROFILE
    if dialed_number in CLIENT_CONFIG:
        client_profile = CLIENT_CONFIG[dialed_number]
    else:
        client_profile = DEFAULT_CLIENT
    
    # Save this session so the AI knows who to be
    active_call_sessions[call_sid] = client_profile

    # 3. LOG THE CALL
    city = form.get("FromCity", "")
    location = f"{city}, {form.get('FromState', '')}" if city else "Unknown"
    
    call_db[call_sid] = {
        "sid": call_sid,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "number": caller_number,       # Who called
        "system_number": dialed_number, # Which of your clients they called
        "client_name": client_profile["name"],
        "location": location,
        "status": "Live",
        "summary": None
    }
    
    print(f"New Call: {caller_number} -> {client_profile['name']} ({dialed_number})")

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
                
                # Send welcome
                await send_deepgram_tts(ws, stream_sid, "Hello! I'm listening.")
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
            if call_sid in active_call_sessions: del active_call_sessions[call_sid]

# ---------------- LOGIC ----------------
async def process_audio_stream(call_sid: str, stream_sid: str, audio_ulaw: bytes):
    pcm16 = audioop.ulaw2lin(audio_ulaw, 2)
    rms = audioop.rms(pcm16, 2)
    
    # VAD Threshold (Low for sensitivity)
    if rms > 600:
        silence_counter[call_sid] = 0
        full_sentence_buffer[call_sid].extend(audio_ulaw)
    else:
        silence_counter[call_sid] += 1

    # Fast Pause (0.4s)
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

        # --- MULTI-USER PROMPT SWITCHING ---
        # 1. Get the profile for this specific call
        profile = active_call_sessions.get(call_sid, DEFAULT_CLIENT)
        # 2. Get the custom prompt
        custom_system_prompt = profile["prompt"]

        response_text = await generate_smart_response(transcript, custom_system_prompt)
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

# Updated to accept 'system_prompt'
async def generate_smart_response(user_text, system_prompt):
    if not groq_client: return "Error."
    try:
        # We append instructions to keep it short
        full_prompt = f"{system_prompt} Keep your answer to 1 short sentence."
        
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

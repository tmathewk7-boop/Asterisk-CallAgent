import os
import base64
import json
import asyncio
import audioop
import httpx # pip install httpx
from collections import defaultdict

# --- IMPORTS ---
from groq import Groq
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ---------- Configuration ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
PORT = int(os.getenv("PORT", 8000))
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://your-app.onrender.com")

# Initialize Groq
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# State
media_ws_map = {}
silence_counter = defaultdict(int)
full_sentence_buffer = defaultdict(bytearray)

# ---------------- TwiML endpoint ----------------
@app.post("/twilio/incoming")
async def twilio_incoming(request: Request):
    form = await request.form()
    host = PUBLIC_URL
    if host.startswith("https://"):
        host = host.replace("https://", "wss://")
    elif host.startswith("http://"):
        host = host.replace("http://", "ws://")
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
    print("Media WS connected")

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            event = data.get("event")

            if event == "start":
                call_sid = data["start"].get("callSid")
                stream_sid = data["start"].get("streamSid")
                media_ws_map[call_sid] = ws
                print(f"[{call_sid}] Stream started")
                
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

    except WebSocketDisconnect:
        print("Media WS disconnected")
    except Exception as e:
        print("Media WS error:", e)
    finally:
        if call_sid in media_ws_map: del media_ws_map[call_sid]
        if call_sid in full_sentence_buffer: del full_sentence_buffer[call_sid]

# ---------------- VAD & Listener (RAW MODE) ----------------
async def process_audio_stream(call_sid: str, stream_sid: str, audio_ulaw: bytes):
    # We only convert to PCM for the volume check (VAD)
    # We keep the original 'audio_ulaw' for sending to Deepgram (Faster)
    pcm16 = audioop.ulaw2lin(audio_ulaw, 2)
    rms = audioop.rms(pcm16, 2)
    
    SILENCE_THRESHOLD = 600 

    if rms > SILENCE_THRESHOLD:
        silence_counter[call_sid] = 0
        # Append RAW UL-AW bytes directly
        full_sentence_buffer[call_sid].extend(audio_ulaw)
    else:
        silence_counter[call_sid] += 1

    # SPEED TWEAK: 20 chunks = 0.4 seconds. 
    # This makes it interrupt you faster.
    PAUSE_LIMIT = 20
    
    if silence_counter[call_sid] >= PAUSE_LIMIT:
        # 2000 bytes of ulaw = 0.25s of audio
        if len(full_sentence_buffer[call_sid]) > 2000: 
            complete_audio = bytes(full_sentence_buffer[call_sid])
            full_sentence_buffer[call_sid].clear()
            silence_counter[call_sid] = 0
            
            # Run in background
            asyncio.create_task(handle_complete_sentence(call_sid, stream_sid, complete_audio))
        else:
            if len(full_sentence_buffer[call_sid]) > 0:
                full_sentence_buffer[call_sid].clear()
            silence_counter[call_sid] = 0

async def handle_complete_sentence(call_sid: str, stream_sid: str, raw_ulaw: bytes):
    print(f"[{call_sid}] Processing...")

    try:
        # 1. Transcribe (Direct Raw Send)
        transcript = await transcribe_raw_audio(raw_ulaw)
        
        if not transcript:
            print(f"[{call_sid}] No speech detected.")
            return

        print(f"[{call_sid}] User said: '{transcript}'")
        
        # 2. Brain (Groq)
        response_text = await generate_smart_response(transcript)
        
        # 3. Speak (Deepgram Aura)
        ws = media_ws_map.get(call_sid)
        if ws:
            await send_deepgram_tts(ws, stream_sid, response_text)

    except Exception as e:
        print(f"Processing Error: {e}")

async def transcribe_raw_audio(raw_ulaw):
    if not DEEPGRAM_API_KEY: return None
    try:
        # Nova-2 supports raw mulaw (Twilio format) directly!
        # This saves us the time of converting to WAV.
        url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&encoding=mulaw&sample_rate=8000"
        
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/basic" # Standard for raw audio
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, content=raw_ulaw)
            
        data = response.json()
        
        if 'results' in data and 'channels' in data['results']:
             return data['results']['channels'][0]['alternatives'][0]['transcript']
        return None
    except Exception as e:
        print(f"Transcribe Error: {e}")
        return None

# ---------------- The Brain (Async) ----------------
async def generate_smart_response(user_text):
    if not groq_client: return "No brain found."
    try:
        print(f"Asking Llama: {user_text}")
        system_prompt = "You are a conversational phone assistant. Be extremely concise. Answer in 1 sentence (max 15 words)."
        
        # Run Groq in thread (it's sync) or use AsyncGroq if available. 
        # For now, simple executor is fine.
        loop = asyncio.get_running_loop()
        completion = await loop.run_in_executor(None, lambda: groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            model="llama-3.1-8b-instant",
            max_tokens=50,
        ))
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq Error: {e}")
        return "I didn't catch that."

# ---------------- DEEPGRAM TTS (Streaming + Async) ----------------
async def send_deepgram_tts(ws: WebSocket, stream_sid: str, text: str):
    if not DEEPGRAM_API_KEY: return
    print(f"Deepgram Streaming: {text}")
    
    url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=mulaw&sample_rate=8000&container=none"
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "application/json"}
    payload = {"text": text}

    try:
        # Use httpx for async streaming
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                async for chunk in response.aiter_bytes():
                    if chunk:
                        payload = base64.b64encode(chunk).decode("ascii")
                        await ws.send_json({
                            "event": "media", 
                            "streamSid": stream_sid, 
                            "media": {"payload": payload}
                        })
                        # Tiny sleep is often not needed with real async streaming
                        # But we keep a micro-sleep to be safe
                        await asyncio.sleep(0.001)

    except Exception as e:
        print(f"TTS Error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)

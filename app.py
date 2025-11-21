import os
import base64
import json
import tempfile
import asyncio
import audioop
import requests  # pip install requests
from pathlib import Path
from collections import defaultdict

# --- IMPORTS ---
from groq import Groq
import speech_recognition as sr
from pydub import AudioSegment

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ---------- Configuration ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY") # <--- NEW KEY
PORT = int(os.getenv("PORT", 8000))
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://your-app.onrender.com")

# Initialize Groq
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None

# Setup Folders
STATIC_DIR = Path("./static")
TTS_DIR = STATIC_DIR / "tts"
TTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

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
                
                # Send welcome using Deepgram (Human Voice)
                await send_deepgram_tts(ws, stream_sid, "Hello! I am online.")
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

# ---------------- VAD & Listener ----------------
async def process_audio_stream(call_sid: str, stream_sid: str, audio_ulaw: bytes):
    pcm16 = audioop.ulaw2lin(audio_ulaw, 2)
    rms = audioop.rms(pcm16, 2)
    
    # High threshold to ignore static
    SILENCE_THRESHOLD = 2000 

    if rms > SILENCE_THRESHOLD:
        silence_counter[call_sid] = 0
        full_sentence_buffer[call_sid].extend(pcm16)
    else:
        silence_counter[call_sid] += 1

    # Wait for 1 second of silence
    if silence_counter[call_sid] >= 50:
        if len(full_sentence_buffer[call_sid]) > 4000: 
            complete_audio = bytes(full_sentence_buffer[call_sid])
            full_sentence_buffer[call_sid].clear()
            silence_counter[call_sid] = 0
            asyncio.create_task(handle_complete_sentence(call_sid, stream_sid, complete_audio))
        else:
            if len(full_sentence_buffer[call_sid]) > 0:
                full_sentence_buffer[call_sid].clear()
            silence_counter[call_sid] = 0

async def handle_complete_sentence(call_sid: str, stream_sid: str, pcm_bytes: bytes):
    print(f"[{call_sid}] Processing speech...")
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = tf.name

    try:
        loop = asyncio.get_running_loop()
        # 1. Save WAV
        await loop.run_in_executor(None, lambda: save_optimized_wav(pcm_bytes, wav_path))

        # 2. Transcribe (Google)
        transcript = await loop.run_in_executor(None, lambda: transcribe_audio(wav_path))
        
        if not transcript:
            print(f"[{call_sid}] No speech detected.")
            return

        print(f"[{call_sid}] User said: '{transcript}'")

        # 3. Brain (Groq)
        response_text = generate_smart_response(transcript)
        
        # 4. Speak (Deepgram Aura)
        ws = media_ws_map.get(call_sid)
        if ws:
            await send_deepgram_tts(ws, stream_sid, response_text)

    except Exception as e:
        print(f"Processing Error: {e}")
    finally:
        try: os.unlink(wav_path)
        except: pass

def save_optimized_wav(pcm_bytes, path):
    audio = AudioSegment(data=pcm_bytes, sample_width=2, frame_rate=8000, channels=1)
    audio = audio.normalize()
    audio = audio.set_frame_rate(16000)
    audio.export(path, format="wav")

def transcribe_audio(wav_path):
    """
    Uses Deepgram Nova-2 for lightning fast Speech-to-Text.
    Replaces the flaky Google Free API.
    """
    if not DEEPGRAM_API_KEY:
        print("Error: DEEPGRAM_API_KEY missing for STT.")
        return None

    try:
        # Deepgram "Nova-2" is their fastest, most accurate model for phone calls
        url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true"
        
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/wav"
        }

        with open(wav_path, "rb") as audio_file:
            response = requests.post(url, headers=headers, data=audio_file)

        if response.status_code != 200:
            print(f"Deepgram STT Error: {response.text}")
            return None

        data = response.json()
        # Extract the transcript text
        transcript = data['results']['channels'][0]['alternatives'][0]['transcript']
        
        if not transcript:
            return None
            
        return transcript

    except Exception as e:
        print(f"Transcribe Error: {e}")
        return None

# ---------------- The Brain ----------------
def generate_smart_response(user_text):
    if not groq_client: return "No brain found."
    try:
        print(f"Asking Llama: {user_text}")
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful phone assistant. Keep answers short and conversational."},
                {"role": "user", "content": user_text},
            ],
            model="llama-3.1-8b-instant",
            max_tokens=60,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq Error: {e}")
        return "I am confused."

# ---------------- DEEPGRAM TTS (The Vapi Killer) ----------------
async def send_deepgram_tts(ws: WebSocket, stream_sid: str, text: str):
    """
    Uses Deepgram Aura. 
    - Human-like breathing and intonation.
    - Ultra low latency.
    - Returns raw µ-law (no conversion needed!).
    """
    if not DEEPGRAM_API_KEY:
        print("Error: DEEPGRAM_API_KEY missing.")
        return

    print(f"Deepgram Generating: {text}")
    
    # We use the REST API directly because it's faster/simpler than the SDK for this
    url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=mulaw&sample_rate=8000&container=none"
    
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"text": text}

    try:
        # Run request in background thread
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: requests.post(url, headers=headers, json=payload))

        if response.status_code != 200:
            print(f"Deepgram Error: {response.text}")
            return

        # Deepgram returns raw audio bytes in the exact format Twilio needs!
        audio_data = response.content

        # Send to Twilio
        chunk_size = 1600 
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            payload = base64.b64encode(chunk).decode("ascii")
            await ws.send_json({
                "event": "media", "streamSid": stream_sid, "media": {"payload": payload}
            })

    except Exception as e:
        print(f"TTS Error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)

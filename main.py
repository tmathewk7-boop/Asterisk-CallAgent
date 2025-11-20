import os
import base64
import asyncio
import json
import tempfile
import time
import sys
import math
import struct
from pathlib import Path
from collections import defaultdict
from functools import partial

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Pydub for audio manipulation
from pydub import AudioSegment
from twilio.rest import Client as TwilioClient
from gtts import gTTS

# ========== MU-LAW HELPER FUNCTIONS (Python 3.13 Safe) ==========
# Twilio uses G.711 Mu-law. We need to convert between Linear PCM (WAV) and Mu-law.
def lin2ulaw(sample):
    """Convert a 16-bit linear PCM sample to 8-bit mu-law."""
    sign = 1
    if sample < 0:
        sample = -sample
        sign = -1
    sample = sample + 132
    if sample > 32767:
        sample = 32767
    exponent = int(math.log(sample) / math.log(2)) - 7
    mantissa = (sample >> (exponent + 3)) & 0x0F
    ulaw = ~(sign * (exponent << 4 | mantissa))
    return ulaw & 0xFF

def audio_to_ulaw(audio_bytes):
    """Convert 16-bit PCM bytes to mu-law bytes."""
    pcm_samples = struct.unpack(f"<{len(audio_bytes)//2}h", audio_bytes)
    return bytes([lin2ulaw(s) for s in pcm_samples])

# ========== ENVIRONMENT VARIABLES ==========
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://your-app.onrender.com") # CHANGE THIS
PORT = int(os.getenv("PORT", 8000))

# ========== GLOBAL STATE ==========
# Buffer incoming audio to prevent processing every 20ms packet
call_audio_buffer = defaultdict(bytearray) 
media_ws_map = {}
last_ai_reply_time = defaultdict(lambda: 0.0)

twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

STATIC_DIR = Path("./static")
TTS_DIR = STATIC_DIR / "tts"
TTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ====================== Twilio Incoming Call Handler ======================
@app.post("/twilio/incoming")
async def twilio_incoming(request: Request):
    # We do NOT use <Play> here for the greeting. We stream it via WebSocket 
    # to ensure the connection is established first.
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{PUBLIC_URL}/media-ws" />
  </Connect>
</Response>"""
    return PlainTextResponse(content=twiml, media_type="text/xml")

# ====================== Twilio Media Streams WS ======================
@app.websocket("/media-ws")
async def media_ws_endpoint(ws: WebSocket):
    await ws.accept()
    call_sid = None
    stream_sid = None
    
    print("Media WS Connected")

    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            event = data.get("event")

            if event == "start":
                call_sid = data["start"]["callSid"]
                stream_sid = data["start"]["streamSid"]
                media_ws_map[call_sid] = ws
                print(f"Stream started. CallSid: {call_sid}")
                
                # Send initial greeting immediately
                await send_tts_to_call(ws, stream_sid, "Hello! I am your AI assistant. How can I help?")

            elif event == "media":
                # 1. Accumulate Audio
                chunk = base64.b64decode(data["media"]["payload"])
                call_audio_buffer[call_sid].extend(chunk)

                # 2. Simple VAD (Voice Activity Detection) Logic
                # Wait until we have roughly 1 second of audio (8000 bytes for mulaw 8khz)
                # In a real app, you'd use a library like 'webrtcvad' or endpoints from Deepgram
                if len(call_audio_buffer[call_sid]) > 8000:
                    audio_data = bytes(call_audio_buffer[call_sid])
                    call_audio_buffer[call_sid] = bytearray() # Clear buffer
                    
                    # 3. Process Logic
                    # We run this in background so we don't block the websocket loop
                    asyncio.create_task(process_conversation(call_sid, stream_sid, audio_data, ws))

            elif event == "stop":
                print(f"Call stopped: {call_sid}")
                break

    except WebSocketDisconnect:
        print("WS Disconnected")
    finally:
        if call_sid in media_ws_map:
            del media_ws_map[call_sid]
        if call_sid in call_audio_buffer:
            del call_audio_buffer[call_sid]

# ====================== AI Processing ======================
async def process_conversation(call_sid, stream_sid, audio_data, ws):
    # Check logic to prevent AI talking over itself too much
    now = time.time()
    if now - last_ai_reply_time[call_sid] < 3: 
        return
    
    # --- STUB: SPEECH TO TEXT WOULD GO HERE ---
    # In a real app, you send 'audio_data' to Deepgram/OpenAI here.
    # Since we don't have an API key, we assume the user said "Hello".
    user_text = "Hello" 
    print(f"Processing audio... Assumed text: {user_text}")

    # --- AI LOGIC ---
    ai_reply = "I heard you speak. This is the Python server responding."
    
    # Update lock
    last_ai_reply_time[call_sid] = time.time()

    # --- TEXT TO SPEECH & SEND ---
    await send_tts_to_call(ws, stream_sid, ai_reply)


async def send_tts_to_call(ws: WebSocket, stream_sid: str, text: str):
    print(f"Generating TTS: {text}")
    
    # 1. Generate MP3 with gTTS
    # Using a temporary file so we don't fill up the disk
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
        tts = gTTS(text=text, lang='en')
        tts.save(tf.name)
        tf.close()
        
        try:
            # 2. Convert MP3 -> PCM 8khz 16-bit Mono
            audio = AudioSegment.from_file(tf.name)
            audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)
            pcm_data = audio.raw_data
            
            # 3. Convert PCM -> Mu-law (Essential for Twilio!)
            ulaw_data = audio_to_ulaw(pcm_data)
            
            # 4. Send in chunks
            # Twilio handles 20ms chunks best, but we can send slightly larger ones
            chunk_size = 1000 # bytes
            
            for i in range(0, len(ulaw_data), chunk_size):
                chunk = ulaw_data[i:i+chunk_size]
                payload = base64.b64encode(chunk).decode("utf-8")
                
                await ws.send_json({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {
                        "payload": payload
                    }
                })
                # Small sleep to not overwhelm the socket buffer
                await asyncio.sleep(0.02)
                
            # Send a "mark" event so we know when audio finishes (optional but good practice)
            await ws.send_json({
                "event": "mark",
                "streamSid": stream_sid,
                "mark": {"name": "reply_complete"}
            })
            
        except Exception as e:
            print(f"Error in TTS sending: {e}")
        finally:
            os.unlink(tf.name)

# ====================== Health Check ======================
@app.get("/")
async def index():
    return HTMLResponse("<h3>Voice Engine Online</h3>")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)

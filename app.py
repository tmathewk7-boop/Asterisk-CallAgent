# main.py
import os
import base64
import asyncio
import json
import tempfile
import time
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import soundfile as sf
import numpy as np

# Audio processing

import pyttsx3  # local TTS fallback

# Twilio
from twilio.rest import Client as TwilioClient

# ========== ENVIRONMENT VARIABLES ==========
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "+1XXXX")
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://example.onrender.com")
PORT = int(os.getenv("PORT", 8000))
TWILIO_SAMPLE_RATE = int(os.getenv("TWILIO_SAMPLE_RATE", 8000))

STATIC_DIR = Path("./static")
TTS_DIR = STATIC_DIR / "tts"
TTS_DIR.mkdir(parents=True, exist_ok=True)

# ========== GLOBAL STATE ==========
audio_buffers = defaultdict(bytearray)
last_activity = {}
call_history = defaultdict(list)  # short-term local conversation memory
last_ai_reply_time = defaultdict(lambda: 0.0)

twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# dashboard websocket manager
dashboard_ws: WebSocket | None = None
dashboard_lock = asyncio.Lock()

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ====================== Twilio Incoming Call Handler ======================
@app.post("/twilio/incoming")
async def twilio_incoming(request: Request):
    form = await request.form()
    from_number = form.get("From")
    call_sid = form.get("CallSid")
    print("Incoming call from", from_number, "CallSid", call_sid)

    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">Connecting you to the AI assistant.</Say>
  <Start>
    <Stream url="wss://asterisk-callagent.onrender.com/media-ws"/>
  </Start>
  <Pause length="3600"/>
</Response>"""

    await notify_dashboard({
        "event": "call_started",
        "from": from_number,
        "callSid": call_sid,
        "timestamp": int(time.time())
    })

    return PlainTextResponse(content=twiml, media_type="text/xml")

# ====================== Twilio Media Streams WS ======================
@app.websocket("/media-ws")
async def media_ws_endpoint(ws: WebSocket):
    await ws.accept()
    print("Media WebSocket connected")
    call_sid = None
    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            event = data.get("event")
            if event == "start":
                call_sid = data.get("start", {}).get("callSid")
                print("Stream started callSid", call_sid)
                last_activity[call_sid] = time.time()
            elif event == "media":
                payload_b64 = data["media"]["payload"]
                pcm = base64.b64decode(payload_b64)
                audio_buffers[call_sid] += pcm
                last_activity[call_sid] = time.time()

                if len(audio_buffers[call_sid]) > (TWILIO_SAMPLE_RATE * 2 * 5):  # ~5s audio
                    pcm_chunk = bytes(audio_buffers[call_sid])
                    audio_buffers[call_sid].clear()
                    asyncio.create_task(process_and_transcribe(call_sid, pcm_chunk))

            elif event == "stop":
                if call_sid and len(audio_buffers[call_sid]) > 0:
                    pcm_chunk = bytes(audio_buffers[call_sid])
                    audio_buffers[call_sid].clear()
                    await process_and_transcribe(call_sid, pcm_chunk)
                await notify_dashboard({"event": "call_ended", "callSid": call_sid, "timestamp": int(time.time())})
            else:
                pass
    except WebSocketDisconnect:
        print("Media WebSocket disconnected.")

# ====================== Dashboard WS ======================
@app.websocket("/dashboard-ws")
async def dashboard_ws_endpoint(ws: WebSocket):
    global dashboard_ws
    await ws.accept()
    print("Dashboard connected.")
    async with dashboard_lock:
        dashboard_ws = ws
    try:
        while True:
            msg = await ws.receive_text()
            print("Dashboard -> backend:", msg)
            try:
                obj = json.loads(msg)
                if obj.get("cmd") == "play_tts":
                    asyncio.create_task(handle_play_tts_cmd(obj.get("callSid"), obj.get("text")))
            except Exception:
                pass
    except WebSocketDisconnect:
        print("Dashboard disconnected.")
    finally:
        async with dashboard_lock:
            if dashboard_ws == ws:
                dashboard_ws = None

async def notify_dashboard(payload: dict):
    global dashboard_ws
    if dashboard_ws is None:
        return
    try:
        await dashboard_ws.send_text(json.dumps(payload))
    except Exception as e:
        print("Error sending to dashboard:", e)

# ====================== Transcription & AI ======================
async def process_and_transcribe(call_sid: str, pcm_bytes: bytes):
    channels = 1
    sampwidth = 2
    try:
        audio = AudioSegment(
            data=pcm_bytes,
            sample_width=sampwidth,
            frame_rate=TWILIO_SAMPLE_RATE,
            channels=channels
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            wav_path = tf.name
            audio.export(wav_path, format="wav")
    except Exception as e:
        print("Audio chunk assembly failed:", e)
        return

    # For testing, simulate transcription as "user said something"
    transcript = "User spoke."
    await notify_dashboard({
        "event": "transcript",
        "callSid": call_sid,
        "text": transcript,
        "timestamp": int(time.time())
    })

    ai_reply = await generate_ai_response(call_sid, transcript)
    tts_url = await synthesize_tts_and_get_url(call_sid, ai_reply)
    if tts_url and twilio_client:
        try:
            twiml_play = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response><Play>{tts_url}</Play></Response>"""
            twilio_client.calls(call_sid).update(twiml=twiml_play)
            await notify_dashboard({"event":"ai_play_started","callSid":call_sid,"tts_url":tts_url})
        except Exception as e:
            print("Error playing TTS in call:", e)


    try:
        Path(wav_path).unlink(missing_ok=True)
    except Exception:
        pass

# ====================== Local Rule-Based AI ======================
async def generate_ai_response(call_sid: str, user_text: str) -> str:
    now = time.time()
    if now - last_ai_reply_time[call_sid] < 2:  # avoid spam
        return "..."
    last_ai_reply_time[call_sid] = now

    # Simple rules for testing
    text = user_text.lower()
    if "hello" in text:
        return "Hello! This is a test AI."
    elif "help" in text:
        return "I am here to assist you."
    else:
        return "Thanks for speaking. This is a demo reply."

# ====================== TTS via pyttsx3 ======================
def safe_filename(call_sid: str, text: str) -> str:
    import hashlib
    h = hashlib.sha256(text.encode()).hexdigest()[:12]
    ts = int(time.time())
    return f"{call_sid}_{ts}_{h}.wav"

async def synthesize_tts_and_get_url(call_sid: str, text: str) -> str | None:
    fname = safe_filename(call_sid, text)
    out_path = TTS_DIR / fname
    try:
        engine = pyttsx3.init()
        engine.save_to_file(text, str(out_path))
        engine.runAndWait()
        return f"{PUBLIC_URL}/static/tts/{fname}"
    except Exception as e:
        print("pyttsx3 TTS failed:", e)
        return None

async def handle_play_tts_cmd(call_sid: str, text: str):
    tts_url = await synthesize_tts_and_get_url(call_sid, text)
    if not tts_url:
        await notify_dashboard({"event":"ai_play_failed","callSid":call_sid})
        return
    twiml_play = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response><Play>{tts_url}</Play></Response>"""
    try:
        twilio_client.calls(call_sid).update(twiml=twiml_play)
        await notify_dashboard({"event":"ai_play_started","callSid":call_sid,"tts_url":tts_url})
    except Exception as e:
        print("play_tts error", e)
        await notify_dashboard({"event":"ai_play_failed","callSid":call_sid,"error":str(e)})

# ====================== Health Route ======================
@app.get("/")
async def index():
    return HTMLResponse("<h3>Voice Engine running. Connect your dashboard to /dashboard-ws</h3>")

# ====================== Run Server ======================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")



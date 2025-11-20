# main.py
media_ws_map = {}
import os
import base64
import asyncio
import json
import tempfile
import time
import sys
import base64
# Mock audioop for Render (Python 3.13)
try:
    import audioop
except ModuleNotFoundError:
    import numpy as np

    class audioop:
        @staticmethod
        def lin2lin(fragment, width_from, width_to):
            return np.frombuffer(fragment, dtype=np.int16).tobytes()
        @staticmethod
        def max(fragment, width):
            return max(fragment) if fragment else 0

    import sys
    sys.modules['audioop'] = audioop

# Patch pyaudioop for pydub
import sys
sys.modules['pyaudioop'] = sys.modules['audioop']

# Now safe to import pydub
from pydub import AudioSegment


import pydub
sys.modules['pyaudioop'] = audioop
from pathlib import Path
from collections import defaultdict

STATIC_DIR = Path("./static")
TTS_DIR = STATIC_DIR / "tts"
TTS_DIR.mkdir(parents=True, exist_ok=True)

welcome_path = TTS_DIR / "welcome.mp3"
if not welcome_path.exists():
    from gtts import gTTS
    tts = gTTS("Connecting you to the AI assistant.", lang="en")
    tts.save(str(welcome_path))

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

from gtts import gTTS
from functools import partial
from asyncio import get_running_loop

# ========== ENVIRONMENT VARIABLES ==========
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "+1XXXX")
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://example.onrender.com")
PORT = int(os.getenv("PORT", 8000))
TWILIO_SAMPLE_RATE = int(os.getenv("TWILIO_SAMPLE_RATE", 8000))



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

    # Play the pre-generated welcome TTS
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Play>{PUBLIC_URL}/static/tts/welcome.mp3</Play>
  <Start>
    <Stream url="{PUBLIC_URL}/media-ws"/>
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
    welcome_sent = set()  # Track which call SIDs have received the welcome

    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            event = data.get("event")

            if event == "start":
                call_sid = data.get("start", {}).get("callSid")
                media_ws_map[call_sid] = ws

                # Stream welcome TTS only once per call
                if call_sid not in welcome_sent:
                    welcome_sent.add(call_sid)
                    welcome_path = TTS_DIR / "welcome.mp3"
                    asyncio.create_task(send_tts_to_call(ws, welcome_path))

            elif event == "media":
                payload_b64 = data["media"]["payload"]
                pcm_bytes = base64.b64decode(payload_b64)
                asyncio.create_task(process_and_transcribe(call_sid, pcm_bytes))
                pass

            elif event == "stop":
                if call_sid and audio_buffers[call_sid]:
                    pcm_chunk = bytes(audio_buffers[call_sid])
                    audio_buffers[call_sid].clear()
                    await process_and_transcribe(call_sid, pcm_chunk)
                pass

    except WebSocketDisconnect:
        print("Media WebSocket disconnected.")
    finally:
        if call_sid and call_sid in media_ws_map:
            del media_ws_map[call_sid]
            print(f"Cleaned up media_ws_map for {call_sid}")


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
    # 1️⃣ Assemble the incoming audio chunk (existing code)
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

    # 2️⃣ Notify dashboard (simulate transcription)
    transcript = "User spoke."
    await notify_dashboard({
        "event": "transcript",
        "callSid": call_sid,
        "text": transcript,
        "timestamp": int(time.time())
    })

    # 3️⃣ Generate AI reply
    ai_reply = await generate_ai_response(call_sid, transcript)

    # 4️⃣ Generate TTS locally (MP3)
    tts_path = TTS_DIR / f"{call_sid}_ai.mp3"
    from gtts import gTTS
    gTTS(ai_reply).save(tts_path)

    # 5️⃣ Stream TTS over the call's media WebSocket (real-time)
    ws = await wait_for_media_ws(call_sid)
    if ws:
        await send_tts_to_call(ws, tts_path)
    else:
        print("Media WebSocket not ready for TTS streaming")


    # 6️⃣ Optional: update Twilio TTS fallback (less reliable)
    if tts_path.exists() and twilio_client:
        try:
            twiml_play = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response><Play>{PUBLIC_URL}/static/tts/{tts_path.name}</Play></Response>"""
            twilio_client.calls(call_sid).update(twiml=twiml_play)
            await notify_dashboard({
                "event":"ai_play_started",
                "callSid":call_sid,
                "tts_url":f"{PUBLIC_URL}/static/tts/{tts_path.name}"
            })
        except Exception as e:
            print("Error playing TTS in call via Twilio:", e)

    # 7️⃣ Cleanup
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
    fname = safe_filename(call_sid, text).replace(".wav", ".mp3")
    out_path = TTS_DIR / fname

    loop = get_running_loop()
    try:
        await loop.run_in_executor(None, partial(gTTS(text=text, lang='en').save, str(out_path)))
        return f"{PUBLIC_URL}/static/tts/{fname}"
    except Exception as e:
        print("gTTS TTS failed:", e)
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

def mp3_to_pcm8khz_bytes(mp3_path):
    audio = AudioSegment.from_file(mp3_path)
    audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)
    return audio.raw_data

async def send_tts_to_call(ws: WebSocket, tts_path):
    pcm = mp3_to_pcm8khz_bytes(tts_path)
    print(f"Sending TTS {tts_path} to call, total bytes={len(pcm)}")
    frame_size = 320
    for i in range(0, len(pcm), frame_size):
        chunk = pcm[i:i+frame_size]
        await ws.send_json({
            "event": "media",
            "media": {"payload": base64.b64encode(chunk).decode()}
        })
        await asyncio.sleep(0.02)
    print("TTS streaming finished")

async def wait_for_media_ws(call_sid, timeout=3):
    start = time.time()
    while time.time() - start < timeout:
        ws = media_ws_map.get(call_sid)
        if ws:
            return ws
        await asyncio.sleep(0.1)
    return None

# ====================== Run Server ======================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")



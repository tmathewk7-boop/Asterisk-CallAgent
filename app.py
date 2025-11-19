# main.py
import os
import base64
import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

# ASR: whisper
import whisper
from pydub import AudioSegment
import numpy as np

import hashlib
import requests
from pathlib import Path
from twilio.rest import Client as TwilioClient
import pyttsx3   # fallback local TTS

load_dotenv()

TWILIO_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "+1XXXX")
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://example.ngrok.io")  # your webhook base URL
PORT = int(os.getenv("PORT", 8000))
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")  # optional
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID")      # optional
STATIC_DIR = Path("./static")
TTS_DIR = STATIC_DIR / "tts"
TTS_DIR.mkdir(parents=True, exist_ok=True)

twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# In-memory buffers keyed by callSid
# Each buffer will be list of raw PCM frames (bytes)
audio_buffers: Dict[str, bytearray] = defaultdict(bytearray)
last_activity: Dict[str, float] = {}

# dashboard websocket manager (single dashboard connection assumed for simplicity)
dashboard_ws: WebSocket | None = None
dashboard_lock = asyncio.Lock()

# load whisper model (choose "small" for demo; change to "tiny"/"base" for speed)
print("Loading Whisper model (this can take time)...")
whisper_model = whisper.load_model(os.getenv("WHISPER_MODEL", "small"))
print("Whisper loaded.")

# ========== Twilio webhook to answer incoming DID call ==========
# Configure your Twilio phone number webhook (Voice -> Webhook) to POST to PUBLIC_URL + /twilio/incoming
@app.post("/twilio/incoming")
async def twilio_incoming(request: Request):
    """
    Twilio will POST here when a call arrives on your DID.
    We return TwiML that instructs Twilio to <Stream> the call audio to our WebSocket /media-ws.
    """
    form = await request.form()
    from_number = form.get("From")
    call_sid = form.get("CallSid")
    print("Incoming call from", from_number, "CallSid", call_sid)

    # TwiML response to start a Media Stream to our websocket server
    # Twilio expects text/xml
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">Please hold while we connect to the AI assistant.</Say>
  <Start>
    <Stream url="wss://{(PUBLIC_URL).replace('https://','').replace('http://','')}/media-ws"/>
  </Start>
  <Pause length="3600"/>
</Response>"""
    # notify dashboard
    await notify_dashboard({
        "event": "call_started",
        "from": from_number,
        "callSid": call_sid,
        "timestamp": int(time.time())
    })
    return PlainTextResponse(content=twiml, media_type="text/xml")


# ========== WebSocket endpoint Twilio Media Streams will connect to ==========
@app.websocket("/media-ws")
async def media_ws_endpoint(ws: WebSocket):
    """
    Accept Twilio Media Streams WebSocket connection.
    Twilio sends control messages (connected, start, media frames).
    See Twilio Media Streams docs for the JSON message formats. :contentReference[oaicite:7]{index=7}
    """
    await ws.accept()
    print("Media WebSocket connected (Twilio Media Streams).")
    call_sid = None
    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            # Twilio sends event 'start' with details
            event = data.get("event")
            if event == "start":
                # connection start metadata
                stream_sid = data.get("start", {}).get("streamSid")
                call_sid = data.get("start", {}).get("callSid")
                print("Stream started", stream_sid, "callSid", call_sid)
                last_activity[call_sid] = time.time()
            elif event == "media":
                # media.payload is base64 of raw audio (encoding: base64-encoded 16-bit signed PCM little-endian, 8kHz or 16kHz depending on Twilio settings)
                payload_b64 = data["media"]["payload"]
                pcm = base64.b64decode(payload_b64)
                # append to buffer for this call
                if call_sid is None:
                    # try callSid in top-level if provided
                    call_sid = data.get("callSid", call_sid)
                audio_buffers[call_sid] += pcm
                last_activity[call_sid] = time.time()

                # simple heuristic: when enough audio accumulated, trigger async transcription
                if len(audio_buffers[call_sid]) > (16000 * 2 * 5):  # e.g., ~5s at 16kHz, 16-bit
                    # grab buffer and empty
                    pcm_chunk = bytes(audio_buffers[call_sid])
                    audio_buffers[call_sid].clear()
                    asyncio.create_task(process_and_transcribe(call_sid, pcm_chunk))
            elif event == "stop":
                # Twilio indicates stream stopped — flush remaining audio
                print("Stream stopped for callSid", call_sid)
                if call_sid and len(audio_buffers[call_sid]) > 0:
                    pcm_chunk = bytes(audio_buffers[call_sid])
                    audio_buffers[call_sid].clear()
                    await process_and_transcribe(call_sid, pcm_chunk)
                await notify_dashboard({"event": "call_ended", "callSid": call_sid, "timestamp": int(time.time())})
            else:
                # other events
                pass
    except WebSocketDisconnect:
        print("Media WebSocket disconnected.")


# ========== Dashboard WebSocket endpoint ==========
@app.websocket("/dashboard-ws")
async def dashboard_ws_endpoint(ws: WebSocket):
    """
    Your .exe dashboard connects here (ws://yourserver:8000/dashboard-ws).
    It receives JSON messages like: call_started, transcript, call_ended.
    It can also send control messages back (not implemented fully here).
    """
    global dashboard_ws
    await ws.accept()
    print("Dashboard connected.")
    async with dashboard_lock:
        dashboard_ws = ws
    try:
        while True:
            msg = await ws.receive_text()
            print("Dashboard -> backend:", msg)
            # you might accept controls like {"cmd":"hangup","callSid":"..."}
            try:
                obj = json.loads(msg)
                if obj.get("cmd") == "hangup":
                    # implement hangup via Twilio REST if needed
                    pass
            except Exception:
                pass
    except WebSocketDisconnect:
        print("Dashboard disconnected.")
    finally:
        async with dashboard_lock:
            if dashboard_ws == ws:
                dashboard_ws = None


async def notify_dashboard(payload: Dict[str, Any]):
    """
    Send JSON payload to dashboard if connected.
    """
    global dashboard_ws
    if dashboard_ws is None:
        return
    try:
        await dashboard_ws.send_text(json.dumps(payload))
    except Exception as e:
        print("Error sending to dashboard:", e)


# ========== Transcription helper ==========
async def process_and_transcribe(call_sid: str, pcm_bytes: bytes):
    """
    Convert raw PCM bytes to a temporary WAV file and run Whisper to transcribe.
    Then send transcript to dashboard.
    """
    # Twilio sends 16-bit signed PCM LE. We don't always know sample rate: common is 8kHz for telephony.
    # For higher accuracy use 16kHz if your Twilio MediaStream is configured to 16kHz.
    sample_rate = int(os.getenv("TWILIO_SAMPLE_RATE", "8000"))
    channels = 1
    sampwidth = 2  # bytes per sample for 16-bit audio

    # write raw PCM to a temporary WAV using pydub
    try:
        # build AudioSegment from raw PCM
        audio = AudioSegment(
            data=pcm_bytes,
            sample_width=sampwidth,
            frame_rate=sample_rate,
            channels=channels
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            wav_path = tf.name
            audio.export(wav_path, format="wav")
    except Exception as e:
        print("Failed to assemble audio chunk:", e)
        return

    # run whisper transcription (blocking, so run in thread via asyncio.to_thread)
    try:
        print(f"Transcribing chunk for {call_sid}, file={wav_path}")
        result = await asyncio.to_thread(whisper_model.transcribe, wav_path, {"language": "en"})
        text = result.get("text", "").strip()
        print("Transcript:", text)
        await notify_dashboard({
            "event": "transcript",
            "callSid": call_sid,
            "text": text,
            "timestamp": int(time.time())
        })
    except Exception as e:
        print("Whisper transcription error:", e)
    finally:
        try:
            Path(wav_path).unlink(missing_ok=True)
        except Exception:
            pass

# ========== Basic health route & quick test page ==========
@app.get("/")
async def index():
    return HTMLResponse("<h3>Voice Engine running. Connect your dashboard to /dashboard-ws</h3>")

# ---------------------------
# TTS helpers
# ---------------------------
def safe_filename_for_text(call_sid: str, text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    ts = int(time.time())
    return f"{call_sid}_{ts}_{h}.wav"

def synthesize_tts_elevenlabs(text: str, out_path: Path) -> bool:
    """
    Use ElevenLabs API to synthesize WAV. Return True on success.
    Requires ELEVENLABS_API_KEY and ELEVEN_VOICE_ID env vars.
    """
    if not ELEVENLABS_API_KEY or not ELEVEN_VOICE_ID:
        return False
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {"text": text, "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
    try:
        r = requests.post(url, json=payload, headers=headers, stream=True, timeout=30)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print("ElevenLabs TTS failed:", e)
        return False

def synthesize_tts_pyttsx3(text: str, out_path: Path) -> bool:
    """
    Local TTS fallback using pyttsx3. Saves a WAV file.
    Note: pyttsx3 may require specific drivers/platform support.
    """
    try:
        engine = pyttsx3.init()
        # Optionally tweak voice/rate here:
        # engine.setProperty('rate', 150)
        engine.save_to_file(text, str(out_path))
        engine.runAndWait()
        return out_path.exists()
    except Exception as e:
        print("pyttsx3 TTS failed:", e)
        return False

async def synthesize_tts_and_get_url(call_sid: str, text: str) -> str | None:
    """
    Create TTS audio for given text, return the public HTTPS URL Twilio can fetch.
    """
    fname = safe_filename_for_text(call_sid, text)
    out_path = TTS_DIR / fname

    # Try ElevenLabs first, then fallback to local
    ok = False
    if ELEVENLABS_API_KEY and ELEVEN_VOICE_ID:
        ok = synthesize_tts_elevenlabs(text, out_path)
    if not ok:
        ok = synthesize_tts_pyttsx3(text, out_path)
    if not ok:
        print("TTS synthesis failed for both ElevenLabs and local fallback.")
        return None

    # Ensure PUBLIC_URL is https and points to your server that serves static files
    public_url = os.getenv("PUBLIC_URL", "").rstrip("/")
    if not public_url:
        print("PUBLIC_URL not configured; Twilio needs an HTTPS URL for audio files.")
        return None

    # Compose URL where Twilio can GET the audio
    tts_url = f"{public_url}/static/tts/{fname}"
    return tts_url

# ---------------------------
# Endpoint for dashboard to request AI -> TTS playback
# ---------------------------
@app.post("/api/ai/play")
async def api_ai_play(request: Request):
    """
    Dashboard (or your backend AI) calls this to play `text` into an active call.
    JSON body: {"callSid":"CAXXX", "text":"Hello customer..."}
    The server synthesizes TTS, hosts it under /static/tts/* and instructs Twilio
    to redirect the call to TwiML that plays the audio file.
    """
    body = await request.json()
    call_sid = body.get("callSid")
    text = body.get("text")
    if not call_sid or not text:
        return {"ok": False, "error": "callSid and text required"}

    tts_url = await synthesize_tts_and_get_url(call_sid, text)
    if not tts_url:
        return {"ok": False, "error": "tts_failed"}

    # Build TwiML to play the audio file and then return to original flow (or hangup)
    twiml_play = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Play>{tts_url}</Play>
  <!-- after play, resume previous flow; could also <Redirect> or <Gather> -->
</Response>"""

    # Use Twilio REST API to update the live call to new TwiML
    if not twilio_client:
        return {"ok": False, "error": "twilio not configured on server"}

    try:
        call = twilio_client.calls(call_sid).update(twiml=twiml_play)
        # notify dashboard
        await notify_dashboard({"event": "ai_play_started", "callSid": call_sid, "tts_url": tts_url})
        return {"ok": True, "tts_url": tts_url, "twilio_sid": call.sid}
    except Exception as e:
        print("Error instructing Twilio to play TTS:", e)
        return {"ok": False, "error": "twilio_update_failed", "detail": str(e)}

# ---------------------------
# Optional: accept dashboard WS commands to play TTS
# (Add within dashboard_ws_endpoint message handling)
# ---------------------------
# In the dashboard WS handler, inside message loop add:
#
#    obj = json.loads(msg)
#    if obj.get("cmd") == "play_tts":
#        # {"cmd":"play_tts","callSid":"CAXX","text":"Respond with this"}
#        asyncio.create_task(handle_play_tts_cmd(obj.get("callSid"), obj.get("text")))
#
# And define:
async def handle_play_tts_cmd(call_sid: str, text: str):
    resp = await api_ai_play.__wrapped__(Request) if False else None
    # simpler: call synthesize + twilio update directly
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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")


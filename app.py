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

load_dotenv()

TWILIO_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "+1XXXX")
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://example.ngrok.io")  # your webhook base URL
PORT = int(os.getenv("PORT", 8000))

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")


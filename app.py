# main.py
"""
Working Twilio Media Streams example (FastAPI).
- Accepts incoming Twilio call webhook at /twilio/incoming (returns TwiML that opens a media stream to /media-ws)
- Handles Twilio media websocket messages at /media-ws
- Buffers incoming audio (µ-law 8kHz) and processes ~1s chunks
- Produces a text reply (stubbed) and streams it back to the call as µ-law frames
Notes:
- Do NOT call Twilio REST to change TwiML mid-call (that will tear down streaming).
- Make sure PUBLIC_URL is reachable by Twilio and uses wss/http/https accordingly.
"""
import os
import base64
import asyncio
import json
import tempfile
import time
import math
import struct
from pathlib import Path
from collections import defaultdict
from functools import partial

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from pydub import AudioSegment
from gtts import gTTS

# ---------- Configuration ----------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")  # optional, not used to change call midstream
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://your-app.onrender.com")  # <-- set this in Render
PORT = int(os.getenv("PORT", 8000))
SAMPLE_RATE = 8000  # Twilio stream sample rate
# -----------------------------------

STATIC_DIR = Path("./static")
TTS_DIR = STATIC_DIR / "tts"
TTS_DIR.mkdir(parents=True, exist_ok=True)

# Prepare a welcome file (will be created if missing)
welcome_path = TTS_DIR / "welcome.mp3"
if not welcome_path.exists():
    gTTS("Connecting you to the AI assistant.", lang="en").save(str(welcome_path))

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# State
call_audio_buffer = defaultdict(bytearray)   # incoming µ-law bytes per callSid
media_ws_map = {}                            # callSid -> websocket
last_ai_reply_time = defaultdict(lambda: 0.0)

# ---------------- µ-law helper functions ----------------
# Standard µ-law encoding/decoding (8-bit µ-law <-> signed 16-bit PCM)
BIAS = 0x84
CLIP = 32635

def ulaw2lin(u_val):
    """Convert a single 8-bit mu-law value to 16-bit signed PCM."""
    u_val = ~u_val & 0xFF
    sign = (u_val & 0x80)
    exponent = (u_val >> 4) & 0x07
    mantissa = u_val & 0x0F
    t = ((mantissa << 3) + 0x84) << exponent
    sample = t - 0x84
    if sign != 0:
        return -sample
    else:
        return sample

def lin2ulaw(sample):
    """Convert a single 16-bit PCM sample to 8-bit µ-law."""
    # Clamp
    if sample > CLIP:
        sample = CLIP
    if sample < -CLIP:
        sample = -CLIP

    sign = 0
    if sample < 0:
        sample = -sample
        sign = 0x80
    sample = sample + BIAS
    exponent = 7
    mask = 0x4000
    for i in range(7):
        if sample & mask:
            exponent = i
            break
        mask >>= 1
    mantissa = (sample >> (exponent + 3)) & 0x0F
    u_val = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return u_val

def ulaw_bytes_to_pcm16_bytes(ulaw_bytes: bytes) -> bytes:
    """Convert bytes of µ-law to bytes of 16-bit LE PCM."""
    out = bytearray()
    for b in ulaw_bytes:
        s = ulaw2lin(b)
        out += struct.pack("<h", int(s))
    return bytes(out)

def pcm16_bytes_to_ulaw_bytes(pcm16_bytes: bytes) -> bytes:
    """Convert 16-bit LE PCM bytes to µ-law bytes."""
    samples = struct.unpack("<{}h".format(len(pcm16_bytes)//2), pcm16_bytes)
    return bytes([lin2ulaw(int(s)) for s in samples])

# ---------------- End µ-law helpers ----------------

# ---------------- TwiML endpoint ----------------
@app.post("/twilio/incoming")
async def twilio_incoming(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid")
    print(f"Incoming call CallSid: {call_sid}")

    # FIX 1: Ensure the URL uses wss:// (Twilio requires wss for media streams)
    host = PUBLIC_URL
    if host.startswith("https://"):
        host = host.replace("https://", "wss://")
    elif host.startswith("http://"):
        host = host.replace("http://", "ws://")
    
    stream_url = f"{host}/media-ws"

    # FIX 2: Use <Connect><Stream> for bidirectional (conversational) audio.
    # <Start> is for background listening (forking); <Connect> is for talking bots.
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
                stream_sid = data["start"].get("streamSid", None)
                media_ws_map[call_sid] = ws
                print(f"[{call_sid}] stream START (streamSid={stream_sid})")
                # Immediately stream the welcome message back (as µ-law)
                await send_text_tts_over_ws(ws, stream_sid, "Connecting you to the AI assistant. Say something and I'll reply.")
                continue

            if event == "media":
                # Twilio sends base64 encoded payload of µ-law bytes (one byte per sample at 8kHz)
                payload_b64 = data["media"]["payload"]
                chunk = base64.b64decode(payload_b64)
                if call_sid is None:
                    # if no call_sid (rare), try to find inside start object or ignore
                    print("Received media before start? ignoring")
                    continue

                # Append to buffer for that call
                call_audio_buffer[call_sid].extend(chunk)

                # If we've accumulated ~1 second (8000 samples => 8000 bytes) then process
                if len(call_audio_buffer[call_sid]) >= 8000:
                    audio_ulaw = bytes(call_audio_buffer[call_sid])
                    call_audio_buffer[call_sid].clear()
                    # spawn background processing (won't block websocket)
                    asyncio.create_task(handle_audio_chunk(call_sid, stream_sid, audio_ulaw))
                continue

            if event == "stop":
                print(f"[{call_sid}] stream STOP")
                break

    except WebSocketDisconnect:
        print("Media WS disconnected")
    except Exception as e:
        print("Media WS error:", e)
    finally:
        if call_sid and call_sid in media_ws_map:
            del media_ws_map[call_sid]
        if call_sid and call_sid in call_audio_buffer:
            del call_audio_buffer[call_sid]
        print(f"[{call_sid}] cleaned up")

# ---------------- Audio processing pipeline ----------------
async def handle_audio_chunk(call_sid: str, stream_sid: str, audio_ulaw: bytes):
    """
    audio_ulaw: raw µ-law bytes (8kHz mono). Convert to PCM for STT or debugging.
    This is a stubbed pipeline that pretends it recognized speech and replies.
    Replace the STT stub with a real STT call (Deepgram, OpenAI, etc.).
    """
    print(f"[{call_sid}] handling audio chunk, {len(audio_ulaw)} bytes")

    # Convert µ-law to 16-bit PCM
    pcm16 = ulaw_bytes_to_pcm16_bytes(audio_ulaw)

    # Optional: write to temp WAV for debugging
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_wav = tf.name
        # Use pydub to build a wav (frame_rate=8000, sample_width=2)
        seg = AudioSegment(
            data=pcm16,
            sample_width=2,
            frame_rate=SAMPLE_RATE,
            channels=1
        )
        seg.export(tmp_wav, format="wav")
        # (You can inspect tmp_wav if needed)
    except Exception as e:
        print(f"[{call_sid}] failed writing debug wav: {e}")

    # --- STT stub ----
    # In production: send `pcm16` to your ASR provider and get user_text.
    user_text = "hello"  # placeholder
    print(f"[{call_sid}] (stub) recognized: {user_text}")

    # Simple AI reply logic — avoid spamming replies too quickly
    now = time.time()
    if now - last_ai_reply_time[call_sid] < 1.5:
        print(f"[{call_sid}] skipping reply to avoid overlap")
        return
    last_ai_reply_time[call_sid] = now

    ai_reply = f"I heard you: \"{user_text}\". This is a test reply."

    # send TTS back
    ws = media_ws_map.get(call_sid)
    if ws:
        await send_text_tts_over_ws(ws, stream_sid, ai_reply)
    else:
        print(f"[{call_sid}] websocket disappeared before reply")

# ---------------- TTS and streaming back as µ-law ----------------
async def send_text_tts_over_ws(ws: WebSocket, stream_sid: str, text: str):
    """
    Generate TTS (gTTS -> mp3), convert to PCM 16-bit @8kHz mono,
    convert to µ-law bytes, and stream them to Twilio over the media websocket.
    """
    print("Generating TTS:", text)
    # create temp mp3
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
        mp3_path = tf.name
    try:
        gTTS(text=text, lang="en").save(mp3_path)

        # load via pydub and convert to PCM16 @8kHz mono
        audio = AudioSegment.from_file(mp3_path)
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
        pcm16 = audio.raw_data  # little-endian 16-bit PCM

        # convert to µ-law bytes
        ulaw_bytes = pcm16_bytes_to_ulaw_bytes(pcm16)

        # send in small chunks (keep under ~1 KB; Twilio handles small packets)
        chunk_size = 160  # 20 ms at 8kHz -> 160 samples -> 160 bytes of µ-law
        for i in range(0, len(ulaw_bytes), chunk_size):
            chunk = ulaw_bytes[i:i + chunk_size]
            payload = base64.b64encode(chunk).decode("ascii")
            msg = {
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": payload}
            }
            await ws.send_json(msg)
            await asyncio.sleep(0.02)  # pace the stream

        # optional mark event
        try:
            await ws.send_json({
                "event": "mark",
                "streamSid": stream_sid,
                "mark": {"name": "tts_done"}
            })
        except Exception:
            pass

        print("TTS streaming finished")
    except Exception as e:
        print("TTS generation/streaming error:", e)
    finally:
        try:
            os.unlink(mp3_path)
        except Exception:
            pass

# ---------------- Health ----------------
@app.get("/")
async def index():
    return HTMLResponse("<h3>Voice Engine Online</h3>")

# ---------------- Run ----------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")


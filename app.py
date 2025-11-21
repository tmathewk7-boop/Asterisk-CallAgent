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
import wave
import functools
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
import audioop

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from pydub import AudioSegment
from gtts import gTTS
import speech_recognition as sr
import datetime
# Initialize the free recognizer
recognizer = sr.Recognizer()

# Tracks the audio for the current sentence
full_sentence_buffer = defaultdict(bytearray) 
# Tracks silence duration
silence_counter = defaultdict(int)

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
    Robust Processing:
    1. Boosts volume so Google can hear you.
    2. Upsamples to 16kHz for better accuracy.
    3. SPEAKS BACK errors instead of staying silent.
    """
    try:
        # 1. Decode and check for silence
        pcm16 = audioop.ulaw2lin(audio_ulaw, 2)
        rms = audioop.rms(pcm16, 2)
        
        SILENCE_THRESHOLD = 500 

        if rms > SILENCE_THRESHOLD:
            silence_counter[call_sid] = 0
            full_sentence_buffer[call_sid].extend(pcm16)
            # print(f"[{call_sid}] ... speaking (Vol: {rms})")
            return
        else:
            silence_counter[call_sid] += 1

        # 2. Process if silence persists for ~1 second
        if silence_counter[call_sid] >= 3 and len(full_sentence_buffer[call_sid]) > 0:
            print(f"[{call_sid}] Processing sentence...")
            
            complete_audio = bytes(full_sentence_buffer[call_sid])
            full_sentence_buffer[call_sid].clear()
            silence_counter[call_sid] = 0

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                wav_path = tf.name

            try:
                # --- FIX 1: PRE-PROCESS AUDIO (Boost Volume + Upsample) ---
                # Load raw PCM into Pydub
                seg = AudioSegment(
                    data=complete_audio, 
                    sample_width=2, 
                    frame_rate=SAMPLE_RATE, # 8000 incoming
                    channels=1
                )
                # Normalize (Boost volume to max) - helps Google hear you
                seg = seg.normalize()
                # Upsample to 16kHz - Google prefers this
                seg = seg.set_frame_rate(16000)
                
                # Export as clean WAV for Google
                seg.export(wav_path, format="wav")
                
                # 3. Send to Google STT
                print(f"[{call_sid}] Sending to Google...")
                loop = asyncio.get_running_loop()
                
                def recognize_speech():
                    r = sr.Recognizer()
                    # Adjust for ambient noise helps if there's static
                    with sr.AudioFile(wav_path) as source:
                        # r.adjust_for_ambient_noise(source, duration=0.5) # Optional if static is bad
                        audio_data = r.record(source)
                        return r.recognize_google(audio_data)

                transcript = await loop.run_in_executor(None, recognize_speech)
                print(f"[{call_sid}] You said: '{transcript}'")

                # 4. Brain Logic
                text = transcript.lower()
                response_text = f"You said: {text}" # Default fallback

                if "hello" in text or "hi" in text:
                    response_text = "Hello! The audio is working perfectly now."
                elif "time" in text:
                    now = datetime.datetime.now().strftime("%I:%M %p")
                    response_text = f"It is currently {now}."
                elif "weather" in text:
                    response_text = "I cannot check the weather, but I hope it is nice."

                # Speak valid response
                ws = media_ws_map.get(call_sid)
                if ws:
                    await send_text_tts_over_ws(ws, stream_sid, response_text)

            except sr.UnknownValueError:
                # --- FIX 2: SPEAK THE ERROR ---
                print(f"[{call_sid}] Google couldn't understand.")
                ws = media_ws_map.get(call_sid)
                if ws:
                    # Tell the user we couldn't hear them
                    await send_text_tts_over_ws(ws, stream_sid, "I heard sound, but I could not understand the words.")
            
            except sr.RequestError:
                print(f"[{call_sid}] Google Connection Error.")
                ws = media_ws_map.get(call_sid)
                if ws:
                    await send_text_tts_over_ws(ws, stream_sid, "I am having trouble connecting to the internet.")
            
            finally:
                try:
                    os.unlink(wav_path)
                except:
                    pass

    except Exception as e:
        print(f"[{call_sid}] Critical Error: {e}")

# ---------------- TTS and streaming back as µ-law ----------------
async def send_text_tts_over_ws(ws: WebSocket, stream_sid: str, text: str):
    """
    Final Optimized Sender:
    1. Generates in background (No server freeze).
    2. Filters audio (No static).
    3. Sends LARGE chunks (No cutting out).
    """
    print(f"Generating TTS: {text}")
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
        mp3_path = tf.name

    try:
        # 1. Generate in background thread (Prevents lag)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, 
            lambda: gTTS(text=text, lang="en").save(mp3_path)
        )

        # 2. Process Audio
        audio = AudioSegment.from_file(mp3_path)
        
        # Filter high freq static
        audio = audio.low_pass_filter(3000) 
        # Reduce volume to prevent clipping
        audio = audio - 14 
        
        # Convert to PCM 16-bit @8kHz mono
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
        pcm16 = audio.raw_data
        ulaw_bytes = audioop.lin2ulaw(pcm16, 2)

        # 3. Stream to Twilio
        # FIX: Send LARGER chunks (1600 bytes = 0.2s of audio)
        # Previously 160 bytes caused network congestion ("cutting out").
        chunk_size = 1600 
        
        for i in range(0, len(ulaw_bytes), chunk_size):
            chunk = ulaw_bytes[i:i + chunk_size]
            payload = base64.b64encode(chunk).decode("ascii")
            
            msg = {
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": payload}
            }
            await ws.send_json(msg)
            
            # Remove sleep entirely. 
            # We trust Twilio's jitter buffer to handle 0.2s chunks smoothly.
            # This ensures the audio arrives as fast as possible.

        # Mark event
        try:
            await ws.send_json({
                "event": "mark",
                "streamSid": stream_sid,
                "mark": {"name": "tts_done"}
            })
        except Exception:
            pass

    except Exception as e:
        print("TTS error:", e)
    finally:
        try:
            if os.path.exists(mp3_path):
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


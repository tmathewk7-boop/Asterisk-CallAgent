import os
import base64
import json
import tempfile
import asyncio
import audioop
import datetime
import io
from pathlib import Path
from collections import defaultdict
from groq import Groq

# --- NEW IMPORTS ---
import edge_tts
import speech_recognition as sr

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydub import AudioSegment

# ---------- Configuration ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
PORT = int(os.getenv("PORT", 8000))
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://your-app.onrender.com")
SAMPLE_RATE = 8000 

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
call_audio_buffer = defaultdict(bytearray)
media_ws_map = {}
silence_counter = defaultdict(int)
full_sentence_buffer = defaultdict(bytearray)

# ---------------- TwiML endpoint ----------------
@app.post("/twilio/incoming")
async def twilio_incoming(request: Request):
    form = await request.form()
    host = PUBLIC_URL
    # Ensure wss:// for websocket
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
                # Send welcome using the FAST TTS
                await send_fast_tts(ws, stream_sid, "Hello! I am online and ready.")
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
        if call_sid in call_audio_buffer: del call_audio_buffer[call_sid]
        if call_sid in full_sentence_buffer: del full_sentence_buffer[call_sid]

# ---------------- VAD & Listener ----------------
async def process_audio_stream(call_sid: str, stream_sid: str, audio_ulaw: bytes):
    """Decodes audio, checks for silence, and triggers processing."""
    pcm16 = audioop.ulaw2lin(audio_ulaw, 2)
    rms = audioop.rms(pcm16, 2)
    
    # THRESHOLD: Lower = more sensitive. Higher = ignores noise.
    SILENCE_THRESHOLD = 500 

    if rms > SILENCE_THRESHOLD:
        silence_counter[call_sid] = 0
        full_sentence_buffer[call_sid].extend(pcm16)
    else:
        silence_counter[call_sid] += 1

    # If we have audio and ~0.5s (2 chunks) of silence, process it
    if silence_counter[call_sid] >= 2 and len(full_sentence_buffer[call_sid]) > 0:
        complete_audio = bytes(full_sentence_buffer[call_sid])
        full_sentence_buffer[call_sid].clear()
        silence_counter[call_sid] = 0
        
        # Run in background to avoid blocking websocket
        asyncio.create_task(handle_complete_sentence(call_sid, stream_sid, complete_audio))

async def handle_complete_sentence(call_sid: str, stream_sid: str, pcm_bytes: bytes):
    print(f"[{call_sid}] Processing speech...")
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = tf.name

    try:
        # 1. Normalize & Convert to WAV for Google (Fixes "Silence" issue)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: save_optimized_wav(pcm_bytes, wav_path))

        # 2. Speech to Text (Google Free)
        transcript = await loop.run_in_executor(None, lambda: transcribe_audio(wav_path))
        
        if not transcript:
            print(f"[{call_sid}] Google detected no speech.")
            return

        print(f"[{call_sid}] User said: '{transcript}'")

        # 3. THE SMART BRAIN (New Logic)
        response_text = generate_smart_response(transcript)
        
        # 4. FAST SPEAK BACK (Edge TTS)
        ws = media_ws_map.get(call_sid)
        if ws:
            await send_fast_tts(ws, stream_sid, response_text)

    except Exception as e:
        print(f"Processing Error: {e}")
    finally:
        try: os.unlink(wav_path)
        except: pass

def save_optimized_wav(pcm_bytes, path):
    # Normalize volume and upsample to 16kHz so Google hears clearly
    audio = AudioSegment(data=pcm_bytes, sample_width=2, frame_rate=8000, channels=1)
    audio = audio.normalize()
    audio = audio.set_frame_rate(16000)
    audio.export(path, format="wav")

def transcribe_audio(path):
    r = sr.Recognizer()
    with sr.AudioFile(path) as source:
        audio = r.record(source)
        try:
            return r.recognize_google(audio)
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            return None

# ---------------- The Smart Brain ----------------
def generate_smart_response(user_text):
    """
    Uses Groq (Llama 3) for extremely fast inference.
    """
    try:
        print(f"Asking Llama 3 (Groq): {user_text}")
        
        system_prompt = """
        You are a helpful, witty, and concise phone assistant. 
        Your replies are converted to audio, so do not use special characters (*, #, -).
        Keep answers short (1-2 sentences max) and conversational.
        """
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            model="llama3-8b-8192", # Extremely fast model
            temperature=0.6,
            max_tokens=100,
        )
        
        ai_text = chat_completion.choices[0].message.content.strip()
        print(f"Llama replied: {ai_text}")
        return ai_text

    except Exception as e:
        print(f"Groq Error: {e}")
        return "I'm having a brain freeze, could you ask that again?"

# ---------------- FAST TTS (Edge-TTS) ----------------
async def send_fast_tts(ws: WebSocket, stream_sid: str, text: str):
    """
    Uses Edge-TTS (Microsoft) which is INSTANT compared to gTTS.
    """
    print(f"Generating Fast TTS: {text}")
    
    # Voice options: 'en-US-AriaNeural' (Female), 'en-US-GuyNeural' (Male)
    VOICE = "en-US-AriaNeural"
    
    try:
        # 1. Generate Audio in Memory (No disk I/O = Faster)
        communicate = edge_tts.Communicate(text, VOICE)
        
        mp3_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_data += chunk["data"]

        # 2. Convert MP3 -> PCM -> u-law
        loop = asyncio.get_running_loop()
        ulaw_bytes = await loop.run_in_executor(None, lambda: convert_mp3_to_ulaw(mp3_data))

        # 3. Stream to Twilio (Large chunks for smoothness)
        chunk_size = 1600 
        for i in range(0, len(ulaw_bytes), chunk_size):
            chunk = ulaw_bytes[i:i + chunk_size]
            payload = base64.b64encode(chunk).decode("ascii")
            
            await ws.send_json({
                "event": "media", 
                "streamSid": stream_sid, 
                "media": {"payload": payload}
            })

    except Exception as e:
        print(f"TTS Error: {e}")

def convert_mp3_to_ulaw(mp3_data):
    # Load from memory
    audio = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
    
    # Filter high freq noise & Format for Phone
    audio = audio.low_pass_filter(3000)
    audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)
    
    # Convert to u-law
    return audioop.lin2ulaw(audio.raw_data, 2)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)


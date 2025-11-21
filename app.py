import os
import base64
import json
import asyncio
import audioop
import httpx
import datetime
from collections import defaultdict

# --- IMPORTS ---
from groq import Groq
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, HTMLResponse, JSONResponse
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

# ---------- DATABASE (In-Memory) ----------
# In a real app, you would use SQLite or Postgres. 
# For now, we store calls in a list.
call_db = {}      # Stores metadata: {call_sid: {number: "...", summary: "..."}}
transcripts = defaultdict(list) # Stores chat history: {call_sid: ["User: hi", "AI: hello"]}
media_ws_map = {}
silence_counter = defaultdict(int)
full_sentence_buffer = defaultdict(bytearray)

# ---------------- DASHBOARD HTML ----------------
@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Call Dashboard</title>
        <meta http-equiv="refresh" content="5"> <style>
            body { font-family: sans-serif; background: #f4f4f9; padding: 20px; }
            h1 { color: #333; }
            table { width: 100%; border-collapse: collapse; background: white; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #007bff; color: white; }
            tr:hover { background-color: #f1f1f1; }
            .status-live { color: green; font-weight: bold; }
            .status-ended { color: #666; }
        </style>
    </head>
    <body>
        <h1>📞 Live Call Dashboard</h1>
        <table id="callTable">
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Caller Number</th>
                    <th>Location</th>
                    <th>Status</th>
                    <th>Summary / Transcript</th>
                </tr>
            </thead>
            <tbody id="tableBody">
                </tbody>
        </table>

        <script>
            async def fetchCalls() {
                const response = await fetch('/api/calls');
                const calls = await response.json();
                const tbody = document.getElementById('tableBody');
                tbody.innerHTML = '';
                
                // Sort by newest first
                calls.reverse();

                calls.forEach(call => {
                    const row = `<tr>
                        <td>${call.timestamp}</td>
                        <td>${call.number}</td>
                        <td>${call.location}</td>
                        <td class="${call.status === 'Live' ? 'status-live' : 'status-ended'}">${call.status}</td>
                        <td>${call.summary || "<i>Listening...</i>"}</td>
                    </tr>`;
                    tbody.innerHTML += row;
                });
            }
            fetchCalls();
            setInterval(fetchCalls, 2000); // Update every 2 seconds
        </script>
    </body>
    </html>
    """

@app.get("/api/calls")
async def get_calls_json():
    # Convert dict to list for the frontend
    return list(call_db.values())

# ---------------- TwiML endpoint ----------------
@app.post("/twilio/incoming")
async def twilio_incoming(request: Request):
    form = await request.form()
    
    # 1. CAPTURE CALL DETAILS
    call_sid = form.get("CallSid")
    caller_number = form.get("From", "Unknown")
    city = form.get("FromCity", "")
    state = form.get("FromState", "")
    location = f"{city}, {state}" if city else "Unknown"
    
    # Save to DB
    call_db[call_sid] = {
        "sid": call_sid,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "number": caller_number,
        "location": location,
        "status": "Live",
        "summary": None
    }
    print(f"New Call: {caller_number} from {location}")

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

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            event = data.get("event")

            if event == "start":
                call_sid = data["start"].get("callSid")
                stream_sid = data["start"].get("streamSid")
                media_ws_map[call_sid] = ws
                
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
        print("Client disconnected")
    except Exception as e:
        print("Error:", e)
    finally:
        if call_sid:
            # 2. HANDLE DISCONNECT & SUMMARIZE
            print(f"Call Ended: {call_sid}")
            if call_sid in call_db:
                call_db[call_sid]["status"] = "Ended"
            
            # Generate Summary in background
            asyncio.create_task(generate_call_summary(call_sid))

            if call_sid in media_ws_map: del media_ws_map[call_sid]
            if call_sid in full_sentence_buffer: del full_sentence_buffer[call_sid]

# ---------------- LOGIC ----------------
async def process_audio_stream(call_sid: str, stream_sid: str, audio_ulaw: bytes):
    pcm16 = audioop.ulaw2lin(audio_ulaw, 2)
    rms = audioop.rms(pcm16, 2)
    
    if rms > 600:
        silence_counter[call_sid] = 0
        full_sentence_buffer[call_sid].extend(audio_ulaw)
    else:
        silence_counter[call_sid] += 1

    if silence_counter[call_sid] >= 20: # 0.4s pause
        if len(full_sentence_buffer[call_sid]) > 2000: 
            complete_audio = bytes(full_sentence_buffer[call_sid])
            full_sentence_buffer[call_sid].clear()
            silence_counter[call_sid] = 0
            asyncio.create_task(handle_complete_sentence(call_sid, stream_sid, complete_audio))
        else:
            if len(full_sentence_buffer[call_sid]) > 0:
                full_sentence_buffer[call_sid].clear()
            silence_counter[call_sid] = 0

async def handle_complete_sentence(call_sid: str, stream_sid: str, raw_ulaw: bytes):
    try:
        transcript = await transcribe_raw_audio(raw_ulaw)
        if not transcript: return

        print(f"[{call_sid}] User: {transcript}")
        
        # 3. LOG USER SPEECH
        transcripts[call_sid].append(f"User: {transcript}")

        response_text = await generate_smart_response(transcript)
        
        # 4. LOG AI RESPONSE
        transcripts[call_sid].append(f"AI: {response_text}")
        
        ws = media_ws_map.get(call_sid)
        if ws:
            await send_deepgram_tts(ws, stream_sid, response_text)

    except Exception as e:
        print(f"Error: {e}")

# ---------------- SUMMARIZER (Llama 3) ----------------
async def generate_call_summary(call_sid: str):
    """
    When call ends, send full transcript to Groq to summarize.
    """
    if not groq_client or call_sid not in transcripts:
        return

    full_text = "\n".join(transcripts[call_sid])
    if not full_text: return

    print(f"Summarizing call {call_sid}...")

    try:
        system_prompt = "You are a secretary. Summarize this phone call in 1 short sentence."
        
        # Run Groq (Sync in thread)
        loop = asyncio.get_running_loop()
        completion = await loop.run_in_executor(None, lambda: groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_text},
            ],
            model="llama-3.1-8b-instant",
            max_tokens=50,
        ))
        
        summary = completion.choices[0].message.content.strip()
        
        # Update DB
        if call_sid in call_db:
            call_db[call_sid]["summary"] = summary
        
        print(f"Summary: {summary}")

    except Exception as e:
        print(f"Summary failed: {e}")

# ---------------- HELPERS ----------------
async def transcribe_raw_audio(raw_ulaw):
    if not DEEPGRAM_API_KEY: return None
    try:
        url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&encoding=mulaw&sample_rate=8000"
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/basic"}
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, content=raw_ulaw)
        data = response.json()
        if 'results' in data and 'channels' in data['results']:
             return data['results']['channels'][0]['alternatives'][0]['transcript']
        return None
    except Exception:
        return None

async def generate_smart_response(user_text):
    if not groq_client: return "Error."
    try:
        system_prompt = "You are a phone assistant. Be concise (1 sentence)."
        loop = asyncio.get_running_loop()
        completion = await loop.run_in_executor(None, lambda: groq_client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}],
            model="llama-3.1-8b-instant", max_tokens=50
        ))
        return completion.choices[0].message.content.strip()
    except Exception:
        return "I didn't catch that."

async def send_deepgram_tts(ws: WebSocket, stream_sid: str, text: str):
    if not DEEPGRAM_API_KEY: return
    url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=mulaw&sample_rate=8000&container=none"
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "application/json"}
    payload = {"text": text}
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                async for chunk in response.aiter_bytes():
                    if chunk:
                        payload = base64.b64encode(chunk).decode("ascii")
                        await ws.send_json({"event": "media", "streamSid": stream_sid, "media": {"payload": payload}})
                        await asyncio.sleep(0.001)
    except Exception:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)

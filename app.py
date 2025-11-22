import os
import pymysql
import base64
import re
import json
import asyncio
import audioop
import httpx
import datetime
from collections import defaultdict
from pydantic import BaseModel
from typing import Optional, Dict, Any

# --- IMPORTS ---
from groq import Groq
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- TWILIO IMPORTS (Required for Forwarding) ---
from twilio.twiml.voice_response import VoiceResponse, Dial

# ---------- Configuration ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
PORT = int(os.getenv("PORT", 8000))
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://your-app.onrender.com")

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ---------- DATABASE (Memory + File) ----------
call_db = {}      
transcripts = defaultdict(list) 
media_ws_map = {}
silence_counter = defaultdict(int)
full_sentence_buffer = defaultdict(bytearray)
active_call_config = {} 
tts_task_map = {}

# --- PERSISTENT SETTINGS ---
def get_db_connection() -> Optional[pymysql.Connection]:
    try:
        conn = pymysql.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            connect_timeout=5,
            cursorclass=pymysql.cursors.DictCursor
        )
        return conn
    except Exception as e:
        print(f"DB Connection error: {e}")
        return None

def get_user_settings(phone_number: str) -> Dict[str, Any]:
    conn = get_db_connection()
    if not conn:
        return {}
    try:
        with conn.cursor() as cursor:
            sql = """
                SELECT system_prompt, greeting, personal_phone, ai_active 
                FROM users 
                WHERE phone_number = %s
            """
            cursor.execute(sql, (phone_number,))
            settings = cursor.fetchone()
            return settings if settings else {}
    finally:
        if conn:
            conn.close()

DEFAULT_CONFIG = {
    "system_prompt": "You are a helpful assistant. Be concise.",
    "greeting": "Hello! I am your AI assistant.",
    "active": True,
    "personal_phone": "" # You can manually set this in user_settings.json
}

# ---------- API MODELS ----------
class AgentSettings(BaseModel):
    phone_number: str
    system_prompt: Optional[str] = None
    greeting: Optional[str] = None
    personal_phone: Optional[str] = None

class ToggleRequest(BaseModel):
    phone_number: str
    active: bool

class DeleteCallsRequest(BaseModel):
    call_sids: list[str]

# ---------------- API ENDPOINTS ----------------

@app.get("/api/calls/{target_number}")
async def get_calls_for_client(target_number: str):
    """Returns calls only for the specific phone number."""
    filtered_calls = []
    for call in call_db.values():
        call_sys_num = call.get("system_number", "").replace(" ", "")
        target = target_number.replace(" ", "")
        
        if call_sys_num == target:
            filtered_calls.append(call)
    filtered_calls.reverse()
    return filtered_calls

# Add this endpoint to handle the deletion
@app.post("/api/calls/delete")
async def delete_calls(req: DeleteCallsRequest):
    """Deletes a list of calls from memory."""
    deleted_count = 0
    for sid in req.call_sids:
        if sid in call_db:
            del call_db[sid]
            # Also clean up transcripts if you want
            if sid in transcripts: del transcripts[sid]
            deleted_count += 1
            
    print(f"Deleted {deleted_count} calls.")
    return {"status": "success", "deleted": deleted_count}

@app.get("/api/settings/{target_number}")
async def get_settings(target_number: str):
    """Get AI settings for a specific user from the Database."""
    settings = get_user_settings(target_number)
    
    # Fallback to defaults if user is not in the DB or has no settings saved
    default_config = {
        "system_prompt": "You are a helpful assistant. Be concise.",
        "greeting": "Hello! I am your AI assistant.",
        "ai_active": True,
        "personal_phone": ""
    }
    
    # Merge DB settings with defaults
    return {**default_config, **settings}

@app.post("/api/settings")
async def update_settings(settings: AgentSettings):
    """Update AI settings in the Database."""
    conn = get_db_connection()
    if not conn:
        return JSONResponse({"status": "error", "message": "Database unavailable"}, status_code=500)
    try:
        with conn.cursor() as cursor:
            sql = """
                UPDATE users SET 
                    system_prompt = %s, 
                    greeting = %s, 
                    personal_phone = %s
                WHERE phone_number = %s
            """
            cursor.execute(sql, (
                settings.system_prompt,
                settings.greeting,
                settings.personal_phone,
                settings.phone_number
            ))
        conn.commit()
        return {"status": "success"}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        if conn:
            conn.close()

@app.post("/toggle")
async def toggle_agent(req: ToggleRequest):
    """Turn the AI ON or OFF in the Database."""
    conn = get_db_connection()
    if not conn:
        return JSONResponse({"status": "error", "message": "Database unavailable"}, status_code=500)
    try:
        with conn.cursor() as cursor:
            sql = "UPDATE users SET ai_active = %s WHERE phone_number = %s"
            cursor.execute(sql, (req.active, req.phone_number))
        conn.commit()
        return {"status": "success", "active": req.active}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        if conn:
            conn.close()

@app.get("/status")
async def server_status():
    return {"status": "online"}

# ---------------- TwiML (UPDATED FOR TOGGLE) ----------------
# --- REPLACEMENT FOR /twilio/incoming ---
@app.post("/twilio/incoming")
async def twilio_incoming(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid")
    caller_number = form.get("From", "Unknown")
    system_number = form.get("To", "") # The number they dialed
    
    # 1. Load User Config from the DATABASE
    config = get_user_settings(system_number) 

    # Fallback to default configuration if DB returned no specific settings
    # Note: ai_active, system_prompt, greeting are all now fields in the DB
    default_config = {
        "system_prompt": "You are a helpful assistant. Be concise.",
        "greeting": "Hello! I am your AI assistant.",
        "ai_active": True,
        "personal_phone": ""
    }
    
    # Merge DB settings with defaults
    config = {**default_config, **config}

    # --- CHECK IF AI IS ACTIVE ---
    is_active = config.get("ai_active", True)
    
    if not is_active:
        # === AI OFF: CALL FORWARDING LOGIC ===
        print(f"AI is OFF for {system_number}. Forwarding call...")
        response = VoiceResponse()
        
        personal_phone = config.get("personal_phone")
        
        if personal_phone:
            response.say("Connecting you to the user.")
            response.dial(personal_phone)
        else:
            response.say("The person you are calling is unavailable and has not set a forwarding number.")
            
        return Response(content=str(response), media_type="application/xml")

    # === AI ON: WEBSOCKET LOGIC ===
    active_call_config[call_sid] = config

    # Log Call
    city = form.get("FromCity", "")
    location = f"{city}, {form.get('FromState', '')}" if city else "Unknown"
    
    call_db[call_sid] = {
        "sid": call_sid,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "number": caller_number,
        "system_number": system_number,
        "client_name": caller_number,
        "location": location,
        "status": "Live",
        "summary": None
    }
    print(f"New Call (AI Active): {caller_number} -> {system_number}")

    host = PUBLIC_URL
    if host.startswith("https://"): host = host.replace("https://", "wss://")
    elif host.startswith("http://"): host = host.replace("http://", "ws://")
    stream_url = f"{host}/media-ws"
    
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{stream_url}" />
    </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")

# ---------------- WebSocket ----------------
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
                
                # Use Custom Greeting
                config = active_call_config.get(call_sid, DEFAULT_CONFIG)
                greeting = config.get("greeting", "Hello.")
                
                # --- FIX: Pass call_sid for logging/optional interruption handling ---
                await send_deepgram_tts(ws, stream_sid, greeting, call_sid)
                
                continue

            if event == "media":
                payload_b64 = data["media"]["payload"]
                chunk = base64.b64decode(payload_b64)
                if call_sid:
                    await process_audio_stream(call_sid, stream_sid, chunk)
                continue

            if event == "stop":
                break
    except Exception:
        pass
    finally:
        if call_sid:
            if call_sid in call_db: call_db[call_sid]["status"] = "Ended"
            asyncio.create_task(generate_call_summary(call_sid))
            if call_sid in media_ws_map: del media_ws_map[call_sid]
            if call_sid in full_sentence_buffer: del full_sentence_buffer[call_sid]
            if call_sid in active_call_config: del active_call_config[call_sid]

# ---------------- Logic ----------------
async def process_audio_stream(call_sid: str, stream_sid: str, audio_ulaw: bytes):
    pcm16 = audioop.ulaw2lin(audio_ulaw, 2)
    rms = audioop.rms(pcm16, 2)
    
    # --- INTERRUPTION CHECK ---
    if rms > 600: # User voice is loud enough to be an active speaker
        silence_counter[call_sid] = 0
        
        # Check if the AI is currently speaking (task is in the map)
        if call_sid in tts_task_map and tts_task_map[call_sid] is not None:
            print(f"[{call_sid}] USER INTERRUPT DETECTED. Canceling AI speech.")
            
            # Cancel the running TTS stream task immediately
            tts_task_map[call_sid].cancel() 
            tts_task_map[call_sid] = None
            
            # The current user speech is a new sentence, so clear the previous buffer
            full_sentence_buffer[call_sid].clear()
            
        full_sentence_buffer[call_sid].extend(audio_ulaw)
        
    else:
        # If the AI was just interrupted, or if no one is talking
        silence_counter[call_sid] += 1

    # --- END-OF-SENTENCE VAD LOGIC ---
    if silence_counter[call_sid] >= 30: 
        if len(full_sentence_buffer[call_sid]) > 2000: 
            complete_audio = bytes(full_sentence_buffer[call_sid])
            full_sentence_buffer[call_sid].clear()
            silence_counter[call_sid] = 0
            
            # Trigger the LLM response after the user finishes speaking
            asyncio.create_task(handle_complete_sentence(call_sid, stream_sid, complete_audio))
        else:
            if len(full_sentence_buffer[call_sid]) > 0: full_sentence_buffer[call_sid].clear()
            silence_counter[call_sid] = 0

async def handle_complete_sentence(call_sid: str, stream_sid: str, raw_ulaw: bytes):
    try:
        transcript = await transcribe_raw_audio(raw_ulaw)
        if not transcript: return

        print(f"[{call_sid}] User: {transcript}")
        transcripts[call_sid].append(f"User: {transcript}")

        asyncio.create_task(extract_client_name(transcript, call_sid))

        MAX_HISTORY_LINES = 10 
        context_history = transcripts[call_sid][-MAX_HISTORY_LINES:]
        
        config = active_call_config.get(call_sid, DEFAULT_CONFIG)
        custom_prompt = config.get("system_prompt", "You are a helpful assistant.")

        response_text = await generate_smart_response(transcript, custom_prompt, context_history)
        transcripts[call_sid].append(f"AI: {response_text}")
        
        ws = media_ws_map.get(call_sid)
        if ws:
            # 1. Create the task, passing the call_sid to the TTS function
            tts_task = asyncio.create_task(send_deepgram_tts(ws, stream_sid, response_text, call_sid))
            
            # 2. Store the task object for immediate interruption
            tts_task_map[call_sid] = tts_task
            
            # 3. Wait for the task to finish (or be cancelled)
            await tts_task
            
            # 4. Clear the task map once speech is finished
            tts_task_map[call_sid] = None 

    except Exception as e:
        print(f"Error in handle_complete_sentence: {e}")

# ---------------- Helpers ----------------
async def generate_call_summary(call_sid: str):
    """Generates a short summary of the conversation history."""
    if not groq_client or call_sid not in transcripts: return
    full_text = "\n".join(transcripts[call_sid])
    if not full_text: return
    try:
        loop = asyncio.get_running_loop()
        completion = await loop.run_in_executor(None, lambda: groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Summarize call in 2 words. Be extremely brief."},
                {"role": "user", "content": full_text}
            ],
            model="llama-3.1-8b-instant", max_tokens=10
        ))
        if call_sid in call_db:
            call_db[call_sid]["summary"] = completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Summary generation failed: {e}")
        pass

async def transcribe_raw_audio(raw_ulaw):
    if not DEEPGRAM_API_KEY: return None
    try:
        url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&encoding=mulaw&sample_rate=8000"
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/basic"}
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, content=raw_ulaw)
        data = response.json()
        if 'results' in data: return data['results']['channels'][0]['alternatives'][0]['transcript']
        return None
    except Exception: return None

async def generate_smart_response(user_text: str, system_prompt: str, context_history: list):
    if not groq_client: return "I apologize, I experienced a brief issue. Could you repeat that?"
    try:
        ssml_prompt = (
            f"{system_prompt} You must respond in a single SSML `<speak>` tag. "
            f"Keep your answer to one short sentence (max 20 words). "
            f"Use SSML tags like `<break time='300ms'/>` for natural pauses, and "
            f"`<say-as interpret-as='filler'>um</say-as>` or `<say-as interpret-as='filler'>uh</say-as>` "
            f"for human-like conversational fluidity. Do not include the initial greeting."
        )

        # --- CONSTRUCT MESSAGES ARRAY FROM HISTORY ---
        messages = [{"role": "system", "content": ssml_prompt}]
        
        # Parse the context_history (e.g., "User: text" or "AI: text")
        for line in context_history:
            if line.startswith("User:"):
                role = "user"
                content = line[5:].strip()
            elif line.startswith("AI:"):
                role = "assistant"
                content = line[3:].strip()
            else:
                continue
                
            # Skip the final message, which is passed separately as user_text
            if content == user_text and role == "user": continue
                
            messages.append({"role": role, "content": content})

        # Add the current user input as the final message
        messages.append({"role": "user", "content": user_text})

        # --- CALL GROQ ---
        loop = asyncio.get_running_loop()
        completion = await loop.run_in_executor(None, lambda: groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            max_tokens=100
        ))
        
        # Extract response text
        raw_response = completion.choices[0].message.content
        
        # --- CRITICAL FIX: REGEX CLEANING ---
        # 1. Replace any sequence of whitespace (tabs, newlines, multiple spaces) with a single space.
        # 2. Strip any leading/trailing spaces.
        cleaned_response = re.sub(r'\s+', ' ', raw_response).strip()
        
        # Ensure the response is wrapped in <speak> tags if Groq lost them
        if not cleaned_response.startswith("<speak>"):
             return f"<speak>{cleaned_response}</speak>" 
        
        return cleaned_response
        
    except Exception as e:
        print(f"Groq generation failed: {e}")
        return "I apologize, I experienced a brief issue. Could you repeat that?"


# Updated send_deepgram_tts signature
async def send_deepgram_tts(ws: WebSocket, stream_sid: str, text: str, call_sid: Optional[str] = None):
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
    except asyncio.CancelledError:
        if call_sid:
            print(f"[{call_sid}] TTS stream CANCELLED by user interrupt.")
        raise 
    except Exception: 
        pass

async def extract_client_name(transcript: str, call_sid: str):
    """Uses Groq's low-latency API to extract the caller's name from a transcript segment."""
    if not groq_client or not transcript: return

    try:
        loop = asyncio.get_running_loop()
        # Zero-shot prompt to extract the name
        completion = await loop.run_in_executor(None, lambda: groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Analyze the following transcript. If the caller explicitly stated their full name, extract ONLY the name (first and last). If no name is given, return the word 'None'."}, 
                {"role": "user", "content": transcript}
            ],
            model="llama-3.1-8b-instant", max_tokens=15
        ))
        
        extracted_name = completion.choices[0].message.content.strip()

        if extracted_name.lower() not in ["none", ""]:
            # Update the call_db dictionary, which is reflected in the dashboard
            if call_sid in call_db:
                call_db[call_sid]["client_name"] = extracted_name
                print(f"[{call_sid}] Name Extracted and Updated: {extracted_name}")
                
    except Exception as e:
        print(f"Error during name extraction: {e}")
        
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)

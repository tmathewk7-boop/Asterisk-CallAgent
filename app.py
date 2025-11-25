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
from twilio.rest import Client

# --- IMPORTS ---
from groq import Groq
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- TWILIO IMPORTS (Required for Forwarding) ---
from twilio.twiml.voice_response import VoiceResponse, Dial

# ---------- Configuration ----------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
PORT = int(os.getenv("PORT", 8000))
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://your-app.onrender.com")
MS_CLIENT_ID = os.getenv("MS_CLIENT_ID", "YOUR_AZURE_CLIENT_ID")
MS_CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET", "YOUR_AZURE_CLIENT_SECRET")
MS_REDIRECT_URI = f"{PUBLIC_URL}/outlook/auth-callback" 
MS_AUTH_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/authorize"
MS_TOKEN_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
MS_GRAPH_URL = "https://graph.microsoft.com/v1.0"
SCHEDULE_REQUEST_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "notify_user_of_schedule_request",
        "description": "Collects the caller's full name, phone number, and requested meeting time/reason when they ask to schedule an appointment. This triggers a notification for the lawyer.",
        "parameters": {
            "type": "object",
            "properties": {
                "caller_name": {"type": "string", "description": "The caller's full name."},
                "caller_phone": {"type": "string", "description": "The caller's phone number, including country code."},
                "requested_time_str": {"type": "string", "description": "The specific time/date the caller requested, kept as a raw string (e.g., 'next Tuesday at 3 PM')."},
                "reason": {"type": "string", "description": "A brief reason for scheduling (e.g., 'case review', 'client intake')."}
            },
            "required": ["caller_name", "caller_phone", "requested_time_str"]
        }
    }
}

TRANSFER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "transfer_call",
        "description": "Transfers the active phone call to a specific lawyer or employee at the firm.",
        "parameters": {
            "type": "object",
            "properties": {
                "person_name": {
                    "type": "string", 
                    "description": "The first name of the person to transfer to (e.g. 'Chris', 'James')."
                }
            },
            "required": ["person_name"]
        }
    }
}

if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_rest_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
else:
    twilio_rest_client = None

# --- FIRM DIRECTORY (Who can receive calls?) ---
# Update these with REAL numbers (E.164 format: +1...)
FIRM_DIRECTORY = {
    "Lawyer": "+14037757197",  
}

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

@app.post("/twilio/transfer-failed")
async def transfer_failed(request: Request):
    """
    Handles failed transfers (busy/no-answer).
    Automatically logs a 'Callback Request' to the dashboard and ends the call.
    """
    form = await request.form()
    call_sid = form.get("CallSid")
    dial_status = form.get("DialCallStatus") # 'busy', 'no-answer', 'failed', 'completed'
    target_name = request.query_params.get("target", "the lawyer")

    print(f"[{call_sid}] Transfer Status: {dial_status}")

    # 1. If the call was answered, we are done.
    if dial_status == "completed":
        return Response(content="<Response><Hangup/></Response>", media_type="application/xml")

    # 2. If Busy/No-Answer, LOG IT TO DASHBOARD
    # Retrieve caller details from active memory
    call_data = call_db.get(call_sid)
    
    if call_data:
        # The lawyer's main account number (to link to dashboard)
        lawyer_main_phone = call_data.get("system_number") 
        
        # Construct the request data
        request_args = {
            "caller_name": call_data.get("client_name", "Unknown Caller"),
            "caller_phone": call_data.get("number", "Unknown"),
            "requested_time_str": "ASAP (Missed Transfer)",
            "reason": f"Missed call transfer to {target_name.capitalize()}"
        }
        
        # Save to the 'schedule_requests' table so it appears on the Booking Page
        save_schedule_request_to_db(lawyer_main_phone, request_args)
        print(f"[{call_sid}] Auto-logged missed transfer for {target_name}")

    # 3. Play a polite message and Hang Up
    # The caller hears this and the call ends.
    twiml = f"""
    <Response>
        <Say>I apologize, but {target_name} is currently unavailable.</Say>
        <Say>I have notified them that you called, and they will return your call shortly. Goodbye.</Say>
        <Hangup/>
    </Response>
    """
    return Response(content=twiml, media_type="application/xml")

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

# --- ADD THIS TO YOUR FASTAPI BACKEND (app.py) ---

@app.get("/api/schedule/requests/{user_phone}")
async def get_schedule_requests(user_phone: str):
    """Retrieves all PENDING schedule requests for the given lawyer."""
    conn = get_db_connection()
    if not conn:
        return JSONResponse({"status": "error", "message": "Database unavailable"}, status_code=500)
    try:
        with conn.cursor() as cursor:
            # Note: The status is hardcoded to 'PENDING' to show only new requests
            sql = """
                SELECT request_id, caller_name, caller_phone, requested_time_str, reason, timestamp
                FROM schedule_requests 
                WHERE user_phone = %s AND status = 'PENDING'
                ORDER BY timestamp DESC
            """
            cursor.execute(sql, (user_phone,))
            requests = cursor.fetchall()
            return requests
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
        # === AI OFF: SIMULTANEOUS RING LOGIC ===
        print(f"AI is OFF for {system_number}. Rejecting call so Desk Phone keeps ringing.")
        response = VoiceResponse()
        
        # <Reject> tells the carrier "I'm busy, try the other phone."
        # This allows the Simultaneous Ring to continue on the Desk Phone.
        response.reject() 
            
        return Response(content=str(response), media_type="application/xml")

    # === AI ON: WEBSOCKET LOGIC ===
    active_call_config[call_sid] = config
    active_call_config[call_sid]['caller_id'] = caller_number

    # Log Call
    city = form.get("FromCity", "")
    location = f"{city}, {form.get('FromState', '')}" if city else "Unknown"
    now_utc = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")
    
    call_db[call_sid] = {
        "sid": call_sid,
        "timestamp": now_utc,
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



# ---------------- Booker ----------------
@app.get("/outlook/connect")
async def outlook_connect():
    """Redirects the user to the Microsoft login page to grant calendar permission."""
    scopes = "openid offline_access Calendars.ReadWrite"
    
    redirect_uri = (
        f"{MS_AUTH_URL}?"
        f"client_id={MS_CLIENT_ID}&"
        f"response_type=code&"
        f"redirect_uri={MS_REDIRECT_URI}&"
        f"scope={scopes}"
    )
    return RedirectResponse(url=redirect_uri)

@app.get("/outlook/auth-callback")
async def outlook_auth_callback(code: Optional[str] = None):
    """Handles the redirect from Microsoft, exchanges the code for tokens, and saves them."""
    if not code:
        return HTMLResponse("<h1>Outlook connection failed. No code received.</h1>", status_code=400)

    # 1. Exchange Code for Tokens
    token_data = {
        "client_id": MS_CLIENT_ID,
        "client_secret": MS_CLIENT_SECRET,
        "code": code,
        "redirect_uri": MS_REDIRECT_URI,
        "grant_type": "authorization_code"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(MS_TOKEN_URL, data=token_data)
        
    if not response.is_success:
        print(f"Token exchange failed: {response.text}")
        return HTMLResponse("<h1>Token exchange failed.</h1>", status_code=500)

    tokens = response.json()
    refresh_token = tokens.get("refresh_token")
    access_token = tokens.get("access_token")

    # 2. Get User Email (Need to use the Access Token)
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {access_token}"}
        user_info_response = await client.get(f"{MS_GRAPH_URL}/me?$select=mail", headers=headers)
        
    if not user_info_response.is_success:
        return HTMLResponse("<h1>Could not retrieve user email.</h1>", status_code=500)
    
    outlook_email = user_info_response.json().get("mail")

    # 3. Save Tokens (Need an endpoint/function to update the users table)
    if refresh_token and outlook_email:
        success = save_outlook_tokens(outlook_email, refresh_token)
        if success:
            return HTMLResponse(f"<h1>Outlook Connected!</h1><p>Email: {outlook_email}</p><p>You can now close this window.</p>")
        else:
            return HTMLResponse("<h1>Failed to save tokens to database.</h1>", status_code=500)
    
    return HTMLResponse("<h1>Connection failed due to missing tokens.</h1>", status_code=500)

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
    
    # --- INTERRUPT/VOICE CHECK ---
    # Lowered from 300 to 200 for maximum sensitivity.
    if rms > 200: # User voice is loud enough to be an active speaker
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
        # If the user is silent
        silence_counter[call_sid] += 1

    # --- END-OF-SENTENCE VAD LOGIC (40 chunks = 800ms) ---
    if silence_counter[call_sid] >= 40: # 40 chunks = 800ms of silence
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

        MAX_HISTORY_LINES = 10 
        context_history = transcripts[call_sid][-MAX_HISTORY_LINES:]
        
        config = active_call_config.get(call_sid, DEFAULT_CONFIG)
        custom_prompt = config.get("system_prompt", "You are a helpful assistant.")
        fixed_caller_id = config.get("caller_id", "N/A") 

        # --- FIX: Run extraction, but don't assign to a variable that doesn't exist ---
        # The function 'extract_client_name' saves directly to 'call_db', 
        # so we don't need a return value here immediately.
        asyncio.create_task(extract_client_name(transcript, call_sid))
        # -----------------------------------------------------------------------------
        
        response_text = await generate_smart_response(transcript, custom_prompt, context_history, fixed_caller_id)
        transcripts[call_sid].append(f"AI: {response_text}")
        
        ws = media_ws_map.get(call_sid)
        if ws:
            tts_task = asyncio.create_task(send_deepgram_tts(ws, stream_sid, response_text, call_sid))
            tts_task_map[call_sid] = tts_task
            
            try:
                await tts_task 
            except asyncio.CancelledError:
                print(f"[{call_sid}] TTS task successfully intercepted and cancelled.")
                pass 
            
            tts_task_map[call_sid] = None 

    except Exception as e:
        print(f"Error in handle_complete_sentence: {e}")
# ---------------- Helpers ----------------
async def generate_call_summary(call_sid: str):
    """Generates a short summary of the conversation history."""
    if not groq_client or call_sid not in transcripts: return
    full_text = "\n".join(transcripts[call_sid])
    call_data = call_db.get(call_sid, {})
    if not call_data: return
    
    call_data["full_transcript"] = full_text
    save_call_log_to_db(call_data)
    
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
        # KEYWORDS parameter biases the AI to hear legal terms instead of random words
        url = (
            "https://api.deepgram.com/v1/listen"
            "?model=nova-2"
            "&smart_format=true"
            "&encoding=mulaw"
            "&sample_rate=8000"
            "&keywords=divorce:2"
            "&keywords=lawyer:2"
            "&keywords=legal:2"
            "&keywords=custody:2"
            "&keywords=court:2"
        )
        
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/basic"}
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, content=raw_ulaw)
        
        data = response.json()
        if 'results' in data: 
            return data['results']['channels'][0]['alternatives'][0]['transcript']
        return None
    except Exception: return None

async def generate_smart_response(user_text: str, system_prompt: str, context_history: list, fixed_caller_id: str, call_sid: str):
    if not groq_client: return "I apologize, I experienced a brief issue."
    
    try:
        # ... (Previous SSML and Message setup remains the same) ...
        cleaned_digits = fixed_caller_id.lstrip('+').lstrip('1') 
        ssml_fixed_caller_id = f'<say-as interpret-as="characters">{cleaned_digits}</say-as>'
        ssml_prompt = (
            f"{system_prompt} The client is calling from: {ssml_fixed_caller_id}. "
            f"Respond in a single SSML `<speak>` tag. Keep it short (max 20 words)."
        )

        messages = [{"role": "system", "content": ssml_prompt}]
        # ... (Add history loop here as before) ...
        messages.append({"role": "user", "content": user_text})

        # --- CALL GROQ (With Transfer Tool Enabled) ---
        loop = asyncio.get_running_loop()
        completion = await loop.run_in_executor(None, lambda: groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            # ENABLE THE TRANSFER TOOL HERE
            tools=[TRANSFER_TOOL_SCHEMA], 
            max_tokens=150
        ))

        # --- TOOL HANDLING LOGIC ---
        if completion.choices[0].message.tool_calls:
            tool_call = completion.choices[0].message.tool_calls[0]
            func_name = tool_call.function.name
            
            if func_name == "transfer_call":
                args = json.loads(tool_call.function.arguments)
                target_name = args.get("person_name", "").lower()
                target_number = FIRM_DIRECTORY.get(target_name)
                
                if target_number and twilio_rest_client:
                    print(f"[{call_sid}] TRANSFERRING -> {target_name}")
                    
                    # --- THE BOOMERANG FIX ---
                    # We add 'action' to the Dial verb.
                    # If the call is NOT completed (busy/no-answer), Twilio hits this URL.
                    # We pass the target_name in the query string so we know who missed the call.
                    callback_url = f"{PUBLIC_URL}/twilio/transfer-failed?target={target_name}"
                    
                    transfer_twiml = f"""
                        <Response>
                            <Say>Please hold while I connect you to {target_name.capitalize()}.</Say>
                            <Dial action="{callback_url}" timeout="20">{target_number}</Dial>
                        </Response>
                    """
                    twilio_rest_client.calls(call_sid).update(twiml=transfer_twiml)
                    return "<speak>Transferring you now.</speak>"
                else:
                    # Fallback if name not found or Client missing
                    return "<speak>I apologize, but I don't have a number for that person. Is there someone else?</speak>"

        # Standard Text Extraction (Fallback if no tool called)
        raw_response = completion.choices[0].message.content
        cleaned_response = re.sub(r'\s+', ' ', raw_response).strip()
        if "<speak>" in cleaned_response:
            cleaned_response = cleaned_response.replace("<speak>", "").replace("</speak>", "")
        return f"<speak>{cleaned_response}</speak>"
            
    except Exception as e:
        print(f"Groq generation failed: {e}")
        return "I apologize, I experienced a brief issue."

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

# --- REPLACE THIS FUNCTION IN app.py ---

async def extract_client_name(transcript: str, call_sid: str):
    """Extracts the caller's name with strict filtering to prevent AI chat."""
    if not groq_client or not transcript: return

    LAWYER_NAME = "Chris"  # Update this to the actual lawyer's name
    
    try:
        loop = asyncio.get_running_loop()
        
        # STRICT PROMPT: Force JSON-like brevity
        system_instruction = (
            "You are a precise data extraction engine. You are NOT a chatbot. "
            "Your ONLY task is to output the caller's full name from the transcript. "
            "Rules:\n"
            f"1. If the name is '{LAWYER_NAME}' or refers to the lawyer, return 'None'.\n"
            "2. If the name is not clearly stated by the caller, return 'None'.\n"
            "3. Do NOT output sentences like 'Here is the name'. Output ONLY the name.\n"
            "4. Ignore phrases like 'My name is'. Just return the name itself."
        )

        completion = await loop.run_in_executor(None, lambda: groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instruction}, 
                {"role": "user", "content": transcript}
            ],
            model="llama-3.1-8b-instant", 
            max_tokens=10, # Extremely low token limit cuts off long hallucinations
            temperature=0.1 # Low temperature for deterministic output
        ))
        
        extracted_name = completion.choices[0].message.content.strip()

        # --- VALIDATION FILTERS ---
        # 1. Reject if it looks like a sentence
        if len(extracted_name.split()) > 4 or "transcript" in extracted_name.lower():
            print(f"[{call_sid}] REJECTED BAD NAME: '{extracted_name}'")
            return

        # 2. Cleaning
        cleaned_name = extracted_name.replace('"', '').replace('.', '').strip()
        cleaned_name = ' '.join(word.capitalize() for word in cleaned_name.split())

        # 3. Final Checks
        if LAWYER_NAME.lower() in cleaned_name.lower():
            return 

        if cleaned_name.lower() != "none" and len(cleaned_name) > 2:
            if call_sid in call_db:
                call_db[call_sid]["client_name"] = cleaned_name
                print(f"[{call_sid}] SUCCESS: Name Updated to '{cleaned_name}'")

    except Exception as e:
        print(f"Error during name extraction: {e}")

def save_call_log_to_db(call_data: dict):
    conn = get_db_connection()
    if not conn: return
    try:
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO calls (call_sid, phone_number, system_number, timestamp, client_name, summary, full_transcript)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                call_data.get('sid'),
                call_data.get('number'),
                call_data.get('system_number'),
                datetime.datetime.utcnow(),
                call_data.get('client_name'),
                call_data.get('summary'),
                call_data.get('full_transcript')
            ))
        conn.commit()
    except Exception as e:
        print(f"Error saving call log to DB: {e}")
    finally:
        if conn: conn.close()

# --- ADD THIS NEW DATABASE FUNCTION ---
def save_schedule_request_to_db(user_phone: str, args: dict) -> bool:
    conn = get_db_connection()
    if not conn:
        print("DB Connection failed for schedule request.")
        return False
    try:
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO schedule_requests 
                (user_phone, caller_name, caller_phone, requested_time_str, reason, status)
                VALUES (%s, %s, %s, %s, %s, 'PENDING')
            """
            cursor.execute(sql, (
                user_phone,
                args.get('caller_name'),
                args.get('caller_phone'),
                args.get('requested_time_str'),
                args.get('reason', 'Scheduling Request')
            ))
        conn.commit()
        print(f"Schedule request saved for {user_phone}.")
        return True
    except Exception as e:
        print(f"Error saving schedule request: {e}")
        conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


async def call_tool_function(tool_call, lawyer_phone: str) -> Optional[str]:
    """Routes the LLM's requested tool call to the corresponding Python function."""
    func_name = tool_call.function.name
    
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        print("Error parsing tool arguments.")
        return None

    if func_name == "notify_user_of_schedule_request":
        print(f"Executing tool: notify_user_of_schedule_request for {lawyer_phone}")
        success = save_schedule_request_to_db(lawyer_phone, args)
        
        # Return the AI's final spoken confirmation immediately
        if success:
            requested_time = args.get('requested_time_str', 'the time you specified')
            reason = args.get('reason', 'your matter')
            
            # This is the final spoken response
            return f"<speak>Thank you. I have successfully logged your request for a meeting regarding <say-as interpret-as='reason'>{reason}</say-as> at {requested_time}. The lawyer will follow up with you shortly.</speak>"
        else:
            return "<speak>I apologize, I was unable to log the request at this moment. Please call back soon.</speak>"
            
    return None

# --- END NEW HELPER FUNCTION ---

# --- END NEW DATABASE FUNCTION ---

# --- NEW ENDPOINT FOR TRANSCRIPT RETRIEVAL ---
@app.get("/api/transcripts/{call_sid}")
async def get_full_transcript(call_sid: str):
    """Retrieves the full conversation history for a given Call SID."""
    transcript_data = transcripts.get(call_sid)
    
    if transcript_data:
        # Return the data as a clean list of strings
        return {"call_sid": call_sid, "transcript": transcript_data}
    else:
        # Return summary if transcript is no longer in memory
        summary = call_db.get(call_sid, {}).get("summary", "Transcript not found in active memory.")
        return JSONResponse(
            {"message": "Transcript not found in active memory. Call may have ended.", 
             "summary": summary}, 
            status_code=404
        )


        
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)

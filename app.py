import os
import pymysql
import base64
import re
import json
import asyncio
import audioop
import httpx
import html
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
from twilio.twiml.voice_response import VoiceResponse, Dial

# ---------- Configuration ----------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
PORT = int(os.getenv("PORT", 8000))
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://your-app.onrender.com")

# Microsoft / Outlook Config (Placeholders)
MS_CLIENT_ID = os.getenv("MS_CLIENT_ID", "YOUR_AZURE_CLIENT_ID")
MS_CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET", "YOUR_AZURE_CLIENT_SECRET")
MS_REDIRECT_URI = f"{PUBLIC_URL}/outlook/auth-callback"
MS_AUTH_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/authorize"
MS_TOKEN_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
MS_GRAPH_URL = "https://graph.microsoft.com/v1.0"

# --- FIRM DIRECTORY ---
# Keys must be lowercase single names to match AI output
FIRM_DIRECTORY = {
    "james": "+13065183350"
}

# --- TOOLS ---
TRANSFER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "transfer_call",
        "description": "Transfers the active phone call to a specific lawyer or employee.",
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

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ---------- DATABASE & STATE ----------
call_db = {}       
transcripts = defaultdict(list) 
media_ws_map = {}
silence_counter = defaultdict(int)
full_sentence_buffer = defaultdict(bytearray)
active_call_config = {} 
tts_task_map = {}

# --- DB FUNCTIONS ---
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
    if not conn: return {}
    try:
        with conn.cursor() as cursor:
            sql = "SELECT system_prompt, greeting, personal_phone, ai_active FROM users WHERE phone_number = %s"
            cursor.execute(sql, (phone_number,))
            return cursor.fetchone() or {}
    finally:
        if conn: conn.close()

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

def save_schedule_request_to_db(user_phone: str, args: dict) -> bool:
    conn = get_db_connection()
    if not conn: return False
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
                args.get('reason', 'Request'),
            ))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving schedule request: {e}")
        return False
    finally:
        if conn: conn.close()

DEFAULT_CONFIG = {
    "system_prompt": "You are a helpful assistant.",
    "greeting": "Hello! I am your AI assistant.",
    "ai_active": True,
    "personal_phone": ""
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

# ---------------- ENDPOINTS ----------------

@app.post("/twilio/transfer-failed")
async def transfer_failed(request: Request):
    """
    Handles failed transfers (busy/no-answer).
    Reconnects the caller to the AI so it can take a message.
    """
    form = await request.form()
    call_sid = form.get("CallSid")
    dial_status = form.get("DialCallStatus")
    target_name = request.query_params.get("target", "the lawyer").capitalize()

    print(f"[{call_sid}] Transfer Status: {dial_status}")

    # If successful, hang up (call is done)
    if dial_status == "completed":
        return Response(content="<Response><Hangup/></Response>", media_type="application/xml")

    # --- BOOMERANG LOGIC (Reconnect to AI) ---
    # We update the AI's config to force a specific apology message upon reconnection
    if call_sid in active_call_config:
        apology_msg = f"I apologize, but {target_name} is currently unavailable. May I have your name and phone number so they can return your call?"
        active_call_config[call_sid]["greeting"] = apology_msg
    
    # Generate Websocket URL
    host = PUBLIC_URL.replace("https://", "wss://").replace("http://", "ws://")
    stream_url = f"{host}/media-ws"
    
    # TwiML to reconnect the stream
    twiml = f"""
    <Response>
        <Connect>
            <Stream url="{stream_url}" />
        </Connect>
    </Response>
    """
    return Response(content=twiml, media_type="application/xml")

@app.post("/api/schedule/reject/{request_id}")
async def reject_schedule_request(request_id: int, payload: dict):
    conn = get_db_connection()
    if not conn: return JSONResponse({"status": "error"}, status_code=500)
    try:
        with conn.cursor() as cursor:
            sql = "DELETE FROM schedule_requests WHERE request_id = %s"
            cursor.execute(sql, (request_id,))
        conn.commit()
        return {"status": "success"}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        if conn: conn.close()

@app.get("/api/calls/{target_number}")
async def get_calls_for_client(target_number: str):
    filtered_calls = []
    for call in call_db.values():
        if call.get("system_number", "").replace(" ", "") == target_number.replace(" ", ""):
            filtered_calls.append(call)
    filtered_calls.reverse()
    return filtered_calls

@app.post("/api/calls/delete")
async def delete_calls(req: DeleteCallsRequest):
    count = 0
    for sid in req.call_sids:
        if sid in call_db:
            del call_db[sid]
            if sid in transcripts: del transcripts[sid]
            count += 1
    return {"status": "success", "deleted": count}

@app.get("/api/settings/{target_number}")
async def get_settings(target_number: str):
    settings = get_user_settings(target_number)
    return {**DEFAULT_CONFIG, **settings}

@app.post("/api/settings")
async def update_settings(settings: AgentSettings):
    conn = get_db_connection()
    if not conn: return JSONResponse({"status": "error"}, status_code=500)
    try:
        with conn.cursor() as cursor:
            sql = "UPDATE users SET system_prompt=%s, greeting=%s, personal_phone=%s WHERE phone_number=%s"
            cursor.execute(sql, (settings.system_prompt, settings.greeting, settings.personal_phone, settings.phone_number))
        conn.commit()
        return {"status": "success"}
    finally:
        if conn: conn.close()

@app.post("/toggle")
async def toggle_agent(req: ToggleRequest):
    conn = get_db_connection()
    if not conn: return JSONResponse({"status": "error"}, status_code=500)
    try:
        with conn.cursor() as cursor:
            sql = "UPDATE users SET ai_active = %s WHERE phone_number = %s"
            cursor.execute(sql, (req.active, req.phone_number))
        conn.commit()
        return {"status": "success", "active": req.active}
    finally:
        if conn: conn.close()

@app.get("/api/schedule/requests/{user_phone}")
async def get_schedule_requests(user_phone: str):
    conn = get_db_connection()
    if not conn: return JSONResponse({"status": "error"}, status_code=500)
    try:
        with conn.cursor() as cursor:
            sql = """
                SELECT request_id, caller_name, caller_phone, requested_time_str, reason, timestamp
                FROM schedule_requests 
                WHERE user_phone = %s AND status = 'PENDING'
                ORDER BY timestamp DESC
            """
            cursor.execute(sql, (user_phone,))
            return cursor.fetchall()
    finally:
        if conn: conn.close()

@app.get("/api/transcripts/{call_sid}")
async def get_full_transcript(call_sid: str):
    transcript_data = transcripts.get(call_sid)
    if transcript_data:
        return {"call_sid": call_sid, "transcript": transcript_data}
    return JSONResponse({"message": "Not found"}, status_code=404)

@app.get("/status")
async def server_status():
    return {"status": "online"}

# ---------------- CALL LOGIC ----------------

@app.post("/twilio/incoming")
async def twilio_incoming(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid")
    caller_number = form.get("From", "Unknown")
    system_number = form.get("To", "")
    
    config = get_user_settings(system_number)
    config = {**DEFAULT_CONFIG, **config}

    is_active = config.get("ai_active", True)
    if not is_active:
        print(f"AI OFF: Rejecting {caller_number} for SimRing.")
        response = VoiceResponse()
        response.reject()
        return Response(content=str(response), media_type="application/xml")

    active_call_config[call_sid] = config
    active_call_config[call_sid]['caller_id'] = caller_number

    call_db[call_sid] = {
        "sid": call_sid,
        "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "number": caller_number,
        "system_number": system_number,
        "client_name": "Unknown", # Placeholder
        "location": f"{form.get('FromCity','')}, {form.get('FromState','')}",
        "status": "Live",
        "summary": None
    }
    
    host = PUBLIC_URL.replace("https://", "wss://").replace("http://", "ws://")
    stream_url = f"{host}/media-ws"
    
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response><Connect><Stream url="{stream_url}" /></Connect></Response>"""
    return Response(content=twiml, media_type="application/xml")

@app.websocket("/media-ws")
async def media_ws_endpoint(ws: WebSocket):
    await ws.accept()
    call_sid, stream_sid = None, None
    try:
        while True:
            data = json.loads(await ws.receive_text())
            event = data.get("event")

            if event == "start":
                call_sid = data["start"].get("callSid")
                stream_sid = data["start"].get("streamSid")
                media_ws_map[call_sid] = ws
                
                config = active_call_config.get(call_sid, DEFAULT_CONFIG)
                # This greeting might be the "Apology" if coming from a failed transfer
                greeting = config.get("greeting", "Hello.")
                await send_deepgram_tts(ws, stream_sid, greeting, call_sid)

            elif event == "media" and call_sid:
                chunk = base64.b64decode(data["media"]["payload"])
                await process_audio_stream(call_sid, stream_sid, chunk)

            elif event == "stop":
                break
    except Exception: pass
    finally:
        if call_sid:
            if call_sid in call_db: call_db[call_sid]["status"] = "Ended"
            asyncio.create_task(generate_call_summary(call_sid))
            if call_sid in media_ws_map: del media_ws_map[call_sid]
            if call_sid in full_sentence_buffer: del full_sentence_buffer[call_sid]
            if call_sid in active_call_config: del active_call_config[call_sid]

async def process_audio_stream(call_sid, stream_sid, audio_bytes):
    rms = audioop.rms(audioop.ulaw2lin(audio_bytes, 2), 2)
    if rms > 200:
        silence_counter[call_sid] = 0
        if tts_task_map.get(call_sid):
            tts_task_map[call_sid].cancel()
            tts_task_map[call_sid] = None
            full_sentence_buffer[call_sid].clear()
        full_sentence_buffer[call_sid].extend(audio_bytes)
    else:
        silence_counter[call_sid] += 1

    if silence_counter[call_sid] >= 40: # Silence detected
        if len(full_sentence_buffer[call_sid]) > 2000:
            audio = bytes(full_sentence_buffer[call_sid])
            full_sentence_buffer[call_sid].clear()
            silence_counter[call_sid] = 0
            asyncio.create_task(handle_complete_sentence(call_sid, stream_sid, audio))

async def handle_complete_sentence(call_sid, stream_sid, audio):
    try:
        transcript = await transcribe_raw_audio(audio)
        if not transcript: return
        
        transcripts[call_sid].append(f"User: {transcript}")
        print(f"[{call_sid}] User: {transcript}")

        # Extract Name (Updates DB directly)
        asyncio.create_task(extract_client_name(transcript, call_sid))
        
        config = active_call_config.get(call_sid, DEFAULT_CONFIG)
        
        response_text = await generate_smart_response(
            transcript, 
            config.get("system_prompt", "You are a helpful assistant."), 
            transcripts[call_sid][-10:], 
            config.get("caller_id", "N/A"),
            call_sid
        )

        transcripts[call_sid].append(f"AI: {response_text}")
        
        ws = media_ws_map.get(call_sid)
        if ws:
            task = asyncio.create_task(send_deepgram_tts(ws, stream_sid, response_text, call_sid))
            tts_task_map[call_sid] = task
            try: await task 
            except asyncio.CancelledError: pass
            tts_task_map[call_sid] = None

    except Exception as e:
        print(f"Error handling sentence: {e}")

async def execute_transfer(json_args, call_sid):
    try:
        clean_json = html.unescape(json_args).replace("'", '"')
        args = json.loads(clean_json)
        target_name = args.get("person_name", "").lower()
        target_number = FIRM_DIRECTORY.get(target_name)
        
        if target_number and twilio_rest_client:
            print(f"[{call_sid}] EXECUTING TRANSFER -> {target_name}")
            callback_url = f"{PUBLIC_URL}/twilio/transfer-failed?target={target_name}"
            
            # SILENT TRANSFER (No Robot Voice)
            # We do NOT use <Say>. We just <Dial>. The caller hears ringing immediately.
            twiml = f"""
                <Response>
                    <Dial action="{callback_url}" timeout="20" answerOnBridge="true" machineDetection="Enable">
                        {target_number}
                    </Dial>
                </Response>
            """
            twilio_rest_client.calls(call_sid).update(twiml=twiml)
            
            # The AI says "Transferring" before the switch happens (using Deepgram voice)
            return "<speak>Transferring you now.</speak>"
        else:
            return "<speak>I apologize, but I cannot connect you at this time.</speak>"
    except Exception as e:
        print(f"Transfer Error: {e}")
        return "<speak>I am having trouble transferring you.</speak>"

async def generate_smart_response(user_text, system_prompt, history, caller_id, call_sid):
    if not groq_client: return "I apologize, I am having trouble."
    try:
        cleaned_id = caller_id.lstrip('+').lstrip('1')
        ssml_prompt = f"{system_prompt} Client Caller ID: <say-as interpret-as='characters'>{cleaned_id}</say-as>. Respond in a single short SSML <speak> tag."
        
        messages = [{"role": "system", "content": ssml_prompt}]
        for line in history:
            role = "user" if line.startswith("User:") else "assistant"
            messages.append({"role": role, "content": line.split(":", 1)[1].strip()})
        messages.append({"role": "user", "content": user_text})

        loop = asyncio.get_running_loop()
        completion = await loop.run_in_executor(None, lambda: groq_client.chat.completions.create(
            messages=messages, model="llama-3.1-8b-instant", tools=[TRANSFER_TOOL_SCHEMA], max_tokens=150
        ))

        # 1. Check for Tool Call
        if completion.choices[0].message.tool_calls:
            return await execute_transfer(completion.choices[0].message.tool_calls[0].function.arguments, call_sid)

        # 2. Check for Hallucinated Tool
        raw = completion.choices[0].message.content
        match = re.search(r"<function=transfer_call>(.*?)</function>", raw, re.DOTALL)
        if match:
            return await execute_transfer(match.group(1).strip(), call_sid)

        # 3. Standard Text
        clean = re.sub(r"<function.*?>.*?</function>", "", raw).strip()
        if "<speak>" in clean: clean = clean.replace("<speak>", "").replace("</speak>", "")
        return f"<speak>{clean}</speak>"

    except Exception as e:
        print(f"Groq Error: {e}")
        return "I apologize, I am having trouble answering."

# --- EXTRACT & SUMMARIZE ---
async def extract_client_name(transcript, call_sid):
    if not groq_client: return
    try:
        loop = asyncio.get_running_loop()
        sys_prompt = "Extract ONLY the caller's name. If not found or refers to lawyer/firm, return 'None'. No sentences."
        res = await loop.run_in_executor(None, lambda: groq_client.chat.completions.create(
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": transcript}],
            model="llama-3.1-8b-instant", max_tokens=10, temperature=0.1
        ))
        name = res.choices[0].message.content.strip().replace('.', '')
        if len(name.split()) < 5 and name.lower() != "none" and "lawyer" not in name.lower():
            if call_sid in call_db:
                call_db[call_sid]["client_name"] = name
                print(f"[{call_sid}] Name set: {name}")
    except Exception: pass

async def generate_call_summary(call_sid):
    if not groq_client or call_sid not in transcripts: return
    full_text = "\n".join(transcripts[call_sid])
    call_db[call_sid]["full_transcript"] = full_text
    save_call_log_to_db(call_db[call_sid])
    
    try:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, lambda: groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Summarize the caller's intent in 3-5 words."},
                {"role": "user", "content": full_text}
            ],
            model="llama-3.1-8b-instant", max_tokens=15
        ))
        call_db[call_sid]["summary"] = res.choices[0].message.content.strip()
    except Exception: pass

async def transcribe_raw_audio(raw):
    if not DEEPGRAM_API_KEY: return None
    url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&encoding=mulaw&sample_rate=8000&keywords=divorce:2&keywords=lawyer:2"
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/basic"}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(url, headers=headers, content=raw)
        return r.json().get('results', {}).get('channels', [{}])[0].get('alternatives', [{}])[0].get('transcript')
    except Exception: return None

async def send_deepgram_tts(ws, stream_sid, text, call_sid):
    if not DEEPGRAM_API_KEY: return
    url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=mulaw&sample_rate=8000&container=none"
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, headers=headers, json={"text": text}) as response:
                async for chunk in response.aiter_bytes():
                    await ws.send_json({"event": "media", "streamSid": stream_sid, "media": {"payload": base64.b64encode(chunk).decode("ascii")}})
    except Exception: pass
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)

import os
import datetime
import pymysql
import json
import urllib.parse
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
from groq import AsyncGroq
import edge_tts
import uvicorn

# ---------------- CONFIG ----------------
PORT = int(os.getenv("PORT", 8000))
PUBLIC_URL = os.getenv("PUBLIC_URL")

TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

twilio_client = Client(TWILIO_SID, TWILIO_AUTH)
groq_client = AsyncGroq(api_key=GROQ_API_KEY)

# ---------------- APP ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for active call transcripts
active_calls: Dict[str, List[dict]] = {}

# ---------------- DB ----------------
def get_db_connection() -> Optional[pymysql.Connection]:
    try:
        return pymysql.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=5
        )
    except Exception as e:
        print("DB error:", e)
        return None

# ---------------- HELPERS ----------------
def normalize_call(row: dict) -> dict:
    appt_time = row.get("appointment_time")
    
    if appt_time:
        display_request = f"Appointment Request: {appt_time}"
    else:
        display_request = row.get("summary", "No summary provided")
        
    return {
        "sid": row["call_sid"], 
        "client_name": row.get("client_name", "Unknown"),
        "phone": row.get("phone_number"), 
        "summary": display_request, 
        "timestamp": row["timestamp"].strftime("%Y-%m-%d %I:%M %p") if isinstance(row.get("timestamp"), datetime.datetime) else str(row.get("timestamp"))
    }

def get_user_settings(system_number: str) -> dict:
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("""
                SELECT system_prompt, greeting, ai_active
                FROM users
                WHERE phone_number=%s
            """, (system_number,))
            return c.fetchone() or {}
    finally:
        if conn:
            conn.close()

# ---------------- THE FREE MICROSOFT NEURAL TTS HACK ----------------
@app.get("/voice/tts")
async def get_edge_tts(text: str):
    """Twilio hits this to get ultra-realistic audio for $0."""
    # We are using Microsoft's 'Ava' voice, which sounds highly realistic
    voice = "en-US-AvaNeural" 
    communicate = edge_tts.Communicate(text, voice)
    
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
            
    return Response(content=audio_data, media_type="audio/mpeg")

# ---------------- 1. INCOMING CALL (TWILIO WEBHOOK) ----------------
@app.post("/voice/incoming")
async def handle_incoming_call(CallSid: str = Form(...), From: str = Form(...), To: str = Form(...)):
    """Triggered by Twilio when someone calls your system number."""
    
    settings = get_user_settings(To)
    
    if settings.get("ai_active") == 0:
        response = VoiceResponse()
        encoded_offline = urllib.parse.quote("The clinic is currently unavailable. Please call back later.")
        response.play(f"{PUBLIC_URL}/voice/tts?text={encoded_offline}")
        response.hangup()
        return HTMLResponse(content=str(response), media_type="application/xml")

    custom_prompt = settings.get("system_prompt") or "You are a helpful receptionist. Keep answers to 1-2 sentences."
    greeting = settings.get("greeting") or "Hello, how can I help you today?"

    full_system_prompt = f"{custom_prompt}\n\nCRITICAL RULE: You are talking on the phone. Do not use bullet points or markdown. Keep your responses conversational, warm, and very brief (1-3 sentences maximum). Ask one question at a time."

    active_calls[CallSid] = [
        {"role": "system", "content": full_system_prompt},
        {"role": "assistant", "content": greeting}
    ]

    print(f"📞 INCOMING CALL: {From} to {To}. Playing greeting.")

    # Convert greeting to realistic human audio
    encoded_greeting = urllib.parse.quote(greeting)
    response = VoiceResponse()
    response.play(f"{PUBLIC_URL}/voice/tts?text={encoded_greeting}")
    response.gather(input="speech", action=f"{PUBLIC_URL}/voice/process", method="POST", speechTimeout="auto")
    
    return HTMLResponse(content=str(response), media_type="application/xml")

# ---------------- 2. CONVERSATION LOOP ----------------
@app.post("/voice/process")
async def process_speech(CallSid: str = Form(...), SpeechResult: str = Form(None)):
    """Twilio sends the transcribed user speech here."""
    response = VoiceResponse()

    if not SpeechResult:
        encoded_retry = urllib.parse.quote("I'm sorry, I didn't catch that. Could you repeat it?")
        response.play(f"{PUBLIC_URL}/voice/tts?text={encoded_retry}")
        response.gather(input="speech", action=f"{PUBLIC_URL}/voice/process", method="POST", speechTimeout="auto")
        return HTMLResponse(content=str(response), media_type="application/xml")

    print(f"🗣️ USER SAID: {SpeechResult}")

    history = active_calls.get(CallSid)
    if not history: 
        response.hangup()
        return HTMLResponse(content=str(response), media_type="application/xml")

    history.append({"role": "user", "content": SpeechResult})

    try:
        ai_completion = await groq_client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=history
        )
        ai_reply = ai_completion.choices[0].message.content
        history.append({"role": "assistant", "content": ai_reply})
        active_calls[CallSid] = history

        print(f"🤖 AI REPLIED: {ai_reply}")

        # Convert AI reply to realistic human audio
        encoded_reply = urllib.parse.quote(ai_reply)
        response.play(f"{PUBLIC_URL}/voice/tts?text={encoded_reply}")
        response.gather(input="speech", action=f"{PUBLIC_URL}/voice/process", method="POST", speechTimeout="auto")
        
    except Exception as e:
        print(f"❌ Groq Error: {e}")
        encoded_error = urllib.parse.quote("I'm having a little trouble connecting. Please hold.")
        response.play(f"{PUBLIC_URL}/voice/tts?text={encoded_error}")

    return HTMLResponse(content=str(response), media_type="application/xml")

# ---------------- 3. POST-CALL EXTRACTION & SAVING ----------------
@app.post("/voice/status")
async def call_status_update(CallSid: str = Form(...), CallStatus: str = Form(...), From: str = Form(...), To: str = Form(...)):
    """Twilio hits this when the call disconnects. We extract data and save to DB."""
    
    if CallStatus in ["completed", "failed", "busy", "no-answer", "canceled"]:
        history = active_calls.pop(CallSid, None)
        
        if not history or len(history) <= 2:
            print("📞 Call ended before conversation happened.")
            return {"status": "ignored"}

        transcript = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in history if msg['role'] != 'system'])
        print(f"📞 CALL ENDED. Extracting data...\n")

        try:
            extraction_prompt = f"""
            Read the following call transcript. Extract the caller's full name and the appointment time they requested.
            Even if the appointment was not fully confirmed, extract whatever they asked for.
            Return ONLY a valid JSON object with the exact keys: "client_name", "appointment_time", and "summary".
            If a value is not mentioned, return null. Do not add any markdown formatting.
            
            Transcript:
            {transcript}
            """
            
            ext_completion = await groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": extraction_prompt}],
                response_format={"type": "json_object"}
            )
            
            data = json.loads(ext_completion.choices[0].message.content)
            
            client_name = data.get("client_name") or "Unknown"
            appt_time = data.get("appointment_time")
            summary = data.get("summary", "No summary provided")
            safe_summary = summary[:250] + "..." if len(summary) > 250 else summary
            
            appt_status = "pending" if appt_time else "none"

            print(f"🎯 EXTRACTION SUCCESS: Name: {client_name}, Time: {appt_time}")

            conn = get_db_connection()
            if conn:
                try:
                    with conn.cursor() as c:
                        c.execute("""
                            INSERT INTO calls (
                                call_sid, phone_number, system_number,
                                timestamp, client_name, summary, full_transcript,
                                appointment_time, appointment_status
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE
                                summary=VALUES(summary),
                                full_transcript=VALUES(full_transcript),
                                appointment_time=VALUES(appointment_time),
                                appointment_status=VALUES(appointment_status),
                                client_name=VALUES(client_name)
                        """, (
                            CallSid, From, To, datetime.datetime.now(datetime.timezone.utc),
                            client_name, safe_summary, transcript, appt_time, appt_status
                        ))
                    conn.commit()
                    print(f"✅ SUCCESS: Saved to dashboard!")
                finally:
                    conn.close()

        except Exception as e:
            print(f"❌ Extraction/DB Error: {e}")

    return {"status": "received"}

# ---------------- DASHBOARD API ENDPOINTS ----------------
class DeleteCallsRequest(BaseModel):
    call_sids: list[str]

@app.post("/api/calls/delete")
async def delete_calls(req: DeleteCallsRequest):
    if not req.call_sids: return {"ok": True, "message": "No selection"}
    conn = get_db_connection()
    if not conn: raise HTTPException(status_code=500, detail="DB Failed")
    try:
        with conn.cursor() as c:
            placeholders = ', '.join(['%s'] * len(req.call_sids))
            c.execute(f"DELETE FROM calls WHERE call_sid IN ({placeholders})", tuple(req.call_sids))
        conn.commit()
        return {"ok": True, "deleted": len(req.call_sids)}
    finally: conn.close()

@app.get("/api/calls/{system_number}")
async def get_calls(system_number: str):
    conn = get_db_connection()
    if not conn: return []
    try:
        with conn.cursor() as c:
            c.execute("SELECT * FROM calls")
            valid_rows = [r for r in c.fetchall() if isinstance(r.get("timestamp"), datetime.datetime)]
            valid_rows.sort(key=lambda r: r["timestamp"], reverse=True)
            return [normalize_call(r) for r in valid_rows[:50]]
    finally: conn.close()

@app.get("/api/schedule/requests/{system_number}")
async def get_schedule_requests(system_number: str):
    conn = get_db_connection()
    if not conn: return []
    try:
        with conn.cursor() as c:
            c.execute("""
                SELECT call_sid as request_id, phone_number as caller_phone,
                       client_name as caller_name, timestamp,
                       appointment_time as requested_time_str, summary as reason
                FROM calls WHERE appointment_status='pending' ORDER BY timestamp DESC
            """)
            return c.fetchall()
    finally: conn.close()

@app.post("/api/schedule/accept/{call_sid}")
async def accept_schedule(call_sid: str, req: Request):
    data = await req.json()
    caller_phone = data.get("caller_phone")
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("UPDATE calls SET appointment_status='accepted' WHERE call_sid=%s", (call_sid,))
            c.execute("SELECT appointment_time FROM calls WHERE call_sid=%s", (call_sid,))
            appt_time = (c.fetchone() or {}).get('appointment_time', "your requested time")
        conn.commit()
        if caller_phone:
            twilio_client.messages.create(
                body=f"Your appointment with Wellness Partners for {appt_time} is confirmed.",
                from_=TWILIO_NUMBER, to=caller_phone
            )
        return {"ok": True}
    finally: conn.close()

@app.post("/api/schedule/reject/{call_sid}")
async def reject_schedule(call_sid: str):
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("UPDATE calls SET appointment_status='rejected' WHERE call_sid=%s", (call_sid,))
        conn.commit()
        return {"ok": True}
    finally: conn.close()

@app.post("/api/calls/update")
async def update_call(req: Request):
    data = await req.json()
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("UPDATE calls SET appointment_status=%s WHERE call_sid=%s", (data["appointment_status"], data["call_sid"]))
        conn.commit()
        return {"ok": True}
    finally: conn.close()

@app.get("/status")
async def status(): return {"status": "online"}

@app.post("/api/settings")
async def save_customization(req: Request):
    try:
        data = await req.json()
        
        # Default to your system number if the frontend doesn't pass it
        system_number = data.get("system_number", "+18254352488") 
        system_prompt = data.get("system_prompt", "")
        greeting = data.get("greeting", "Hello, how can I help you today?")
        ai_active = data.get("ai_active", 1) 

        conn = get_db_connection()
        if not conn:
            return {"ok": False, "error": "Database connection failed"}
            
        with conn.cursor() as c:
            c.execute("""
                UPDATE users 
                SET system_prompt=%s, greeting=%s, ai_active=%s 
                WHERE phone_number=%s
            """, (system_prompt, greeting, ai_active, system_number))
        conn.commit()
        conn.close()
        
        print(f"✅ CUSTOMIZATION SAVED: {greeting[:30]}...")
        return {"ok": True}
        
    except Exception as e:
        print(f"❌ CUSTOMIZATION SAVE ERROR: {e}")
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)

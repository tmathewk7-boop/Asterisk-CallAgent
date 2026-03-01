import os
import datetime
import pymysql
import json
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
from groq import AsyncGroq
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
# Maps CallSid -> List of message dictionaries
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

# ---------------- 1. INCOMING CALL (TWILIO WEBHOOK) ----------------
@app.post("/voice/incoming")
async def handle_incoming_call(CallSid: str = Form(...), From: str = Form(...), To: str = Form(...)):
    """Triggered by Twilio when someone calls your system number."""
    
    # 🚨 DYNAMIC INJECTION: Fetch from your Dashboard's Customize AI Page
    settings = get_user_settings(To)
    
    # Check Master Switch
    if settings.get("ai_active") == 0:
        response = VoiceResponse()
        response.say("The clinic is currently unavailable. Please call back later.")
        response.hangup()
        return HTMLResponse(content=str(response), media_type="application/xml")

    # Fetch User's Custom Prompt and Greeting
    custom_prompt = settings.get("system_prompt") or "You are a helpful receptionist. Keep answers to 1-2 sentences."
    greeting = settings.get("greeting") or "Hello, how can I help you today?"

    # Force the AI to be conversational over the phone
    full_system_prompt = f"{custom_prompt}\n\nCRITICAL RULE: You are talking on the phone. Do not use bullet points or markdown. Keep your responses conversational, warm, and very brief (1-3 sentences maximum). Ask one question at a time."

    # Initialize memory for this specific call
    active_calls[CallSid] = [
        {"role": "system", "content": full_system_prompt},
        {"role": "assistant", "content": greeting}
    ]

    print(f"📞 INCOMING CALL: {From} to {To}. Playing greeting.")

    # TwiML to speak the custom greeting, then listen for user speech
    response = VoiceResponse()
    response.say(greeting, voice="Polly.Joanna-Neural")
    response.gather(input="speech", action=f"{PUBLIC_URL}/voice/process", method="POST", speechTimeout="auto")
    
    return HTMLResponse(content=str(response), media_type="application/xml")

# ---------------- 2. CONVERSATION LOOP ----------------
@app.post("/voice/process")
async def process_speech(CallSid: str = Form(...), SpeechResult: str = Form(None)):
    """Twilio sends the transcribed user speech here."""
    response = VoiceResponse()

    # If the user was silent, prompt them again
    if not SpeechResult:
        response.say("I'm sorry, I didn't catch that. Could you repeat it?", voice="Polly.Joanna-Neural")
        response.gather(input="speech", action=f"{PUBLIC_URL}/voice/process", method="POST", speechTimeout="auto")
        return HTMLResponse(content=str(response), media_type="application/xml")

    print(f"🗣️ USER SAID: {SpeechResult}")

    history = active_calls.get(CallSid)
    if not history: # Fallback if call drops memory
        response.hangup()
        return HTMLResponse(content=str(response), media_type="application/xml")

    # Add user speech to memory
    history.append({"role": "user", "content": SpeechResult})

    # Ask Groq how to reply
    try:
        ai_completion = await groq_client.chat.completions.create(
            model="llama-3.1-8b-instant", # Free, blazing fast model
            messages=history
        )
        ai_reply = ai_completion.choices[0].message.content
        history.append({"role": "assistant", "content": ai_reply})
        active_calls[CallSid] = history

        print(f"🤖 AI REPLIED: {ai_reply}")

        # Tell Twilio to speak the reply and listen again
        response.say(ai_reply, voice="Polly.Joanna-Neural")
        response.gather(input="speech", action=f"{PUBLIC_URL}/voice/process", method="POST", speechTimeout="auto")
        
    except Exception as e:
        print(f"❌ Groq Error: {e}")
        response.say("I'm having a little trouble connecting. Please hold.", voice="Polly.Joanna-Neural")

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

        # Build raw transcript
        transcript = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in history if msg['role'] != 'system'])
        print(f"📞 CALL ENDED. Extracting data...\n")

        # --- FORCE GROQ TO EXTRACT THE DATA IN JSON MODE ---
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
                model="llama3-8b-8192",
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

            # --- SAVE TO DATABASE ---
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
                body=f"Your appointment with Verity Law for {appt_time} is confirmed.",
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

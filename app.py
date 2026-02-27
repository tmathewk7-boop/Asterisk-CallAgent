import os
import datetime
import pymysql
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from twilio.rest import Client
import uvicorn

# ---------------- CONFIG ----------------
PORT = int(os.getenv("PORT", 8000))
PUBLIC_URL = os.getenv("PUBLIC_URL")

TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")

twilio_client = Client(TWILIO_SID, TWILIO_AUTH)

# ---------------- APP ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    # FIX: In your SQL query 'SELECT *', the column name is 'phone_number'
    # but this function was looking for 'phone_number' and then trying to 
    # map it to a key called 'phone'. We need to make sure the dashboard 
    # receives exactly what it expects.
    return {
        "call_sid": row["call_sid"],
        "client_name": row.get("client_name", "Unknown"),
        "phone_number": row.get("phone_number", "Unknown"), # Changed key from 'phone' to 'phone_number'
        "summary": row.get("summary", ""),
        "timestamp": row["timestamp"].strftime("%Y-%m-%d %I:%M %p") if row.get("timestamp") else "N/A"
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
        conn.close()

def get_lawyer_by_name(system_number: str, name: str) -> Optional[str]:
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("""
                SELECT phone_number
                FROM users
                WHERE LOWER(display_name)=%s
                AND firm_number=%s
            """, (name.lower(), system_number))
            row = c.fetchone()
            return row["phone_number"] if row else None
    finally:
        conn.close()

# ---------------- CALL STORAGE ----------------
def save_call_log(call: dict):
    conn = get_db_connection()
    if not conn: return
    try:
        with conn.cursor() as c:
            # We extract the number from Vapi's specific nested JSON path
            customer_number = call.get("customer", {}).get("number", "Unknown")
            system_number = call.get("phoneNumber", {}).get("number", "Unknown")
            
            c.execute("""
                INSERT INTO calls (
                    call_sid, phone_number, system_number,
                    timestamp, client_name, summary, full_transcript
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                    summary=VALUES(summary),
                    full_transcript=VALUES(full_transcript)
            """, (
                call["id"],
                customer_number,
                system_number,
                datetime.datetime.utcnow(),
                call.get("analysis", {}).get("structuredData", {}).get("client_name", "Unknown"),
                call.get("analysis", {}).get("summary", ""),
                call.get("transcript", "")
            ))
        conn.commit()
    finally:
        conn.close()

# ---------------- APPOINTMENTS ----------------
def save_appointment(call_sid, status, time, attempt, summary):
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("""
                UPDATE calls
                SET appointment_status=%s,
                    appointment_time=%s,
                    appointment_attempt=%s,
                    summary=%s
                WHERE call_sid=%s
            """, (status, time, attempt, summary, call_sid))
        conn.commit()
    finally:
        conn.close()

def trigger_rebooking_call(call_sid, client_phone, system_number, attempt):
    if attempt > 5:
        return

    settings = get_user_settings(system_number)
    greeting = settings.get("greeting", "The lawyer is unavailable.")

    twiml = f"""
    <Response>
        <Say>{greeting}</Say>
        <Say>Please state another time that works.</Say>
        <Record
            maxLength="20"
            action="{PUBLIC_URL}/twilio/rebooking-complete?call_sid={call_sid}&attempt={attempt}&system_number={system_number}"
        />
    </Response>
    """

    twilio_client.calls.create(
        to=client_phone,
        from_=TWILIO_NUMBER,
        twiml=twiml
    )

@app.post("/")  # This stops the 'POST / 404' error
async def handle_vapi_webhook(request: Request):
    data = await request.json()
    
    # Extracting the info Vapi sends at the end of a call
    message = data.get('message', {})
    call_data = message.get('call', {})
    
    call_id = call_data.get('id')
    phone = message.get('customer', {}).get('number')
    status = call_data.get('status')
    
    if call_id:
        # Save to your Oracle DB (40.233.108.163)
        # Ensure you use your db connection logic here
        save_to_db(call_id, phone, status) 
        print(f"Call {call_id} saved to database.")
        return {"status": "success"}, 200
    
    return {"status": "ignored"}, 200

@app.post("/webhook")
async def vapi_webhook(request: Request):
    data = await request.json()
    
    # Extract the call details Vapi sends
    call_id = data.get('message', {}).get('call', {}).get('id')
    phone = data.get('message', {}).get('customer', {}).get('number')
    status = data.get('message', {}).get('call', {}).get('status')
    
    if call_id:
        # Save to your Oracle MySQL database
        save_call_to_db(call_id, phone, status)
        return {"status": "success"}
    
    return {"status": "ignored"}
    
# ---------------- VAPI WEBHOOK ----------------
# Apply these decorators to catch ALL incoming Vapi signals
@app.post("/")
@app.post("/webhook")
@app.post("/vapi/webhook")
async def combined_vapi_webhook(req: Request):
    payload = await req.json()
    msg = payload.get("message", {})
    msg_type = msg.get("type")

    # This is the event that contains the final phone number and summary
    if msg_type == "end-of-call-report":
        call = msg.get("call", {})
        if call:
            # 1. This function extracts 'customer' -> 'number'
            save_call_log(call) 
            
            # 2. Handle appointments if they exist
            appt = call.get("analysis", {}).get("appointment")
            if appt:
                save_appointment(
                    call["id"],
                    appt["status"],
                    appt["proposed_time"],
                    appt.get("attempt", 1),
                    call.get("analysis", {}).get("summary", "")
                )
            return {"status": "success"}

    # Handle live tool calls (like transfers)
    if msg_type == "tool-calls":
        results = []
        # Safely get the system number for lawyer lookup
        phone_data = msg.get("call", {}).get("phoneNumber", {})
        system_number = phone_data.get("number")
        
        for tool in msg.get("toolCalls", []):
            if tool["function"]["name"] == "transfer_call":
                args = tool["function"]["arguments"]
                target = get_lawyer_by_name(system_number, args.get("person_name", ""))
                results.append({
                    "toolCallId": tool["id"], 
                    "result": f"Transfer to {target}" if target else "Lawyer not found"
                })
        return {"results": results}

    return {"ignored": True}
# ---------------- REBOOK CALLBACK ----------------
@app.post("/twilio/rebooking-complete")
async def rebooking_complete(req: Request):
    form = await req.form()
    call_sid = req.query_params["call_sid"]
    attempt = int(req.query_params["attempt"])
    system_number = req.query_params["system_number"]

    transcript = "Client requested new time"  # plug your transcription here
    new_time = "TBD"                            # plug NLP extraction here

    save_appointment(
        call_sid,
        "pending",
        new_time,
        attempt,
        f"Rebooking attempt {attempt}: {transcript}"
    )

    return {"ok": True}

# ---------------- DASHBOARD API ----------------
class DeleteCallsRequest(BaseModel):
    call_sids: list[str]

@app.get("/api/calls/{system_number}")
async def get_calls(system_number: str):
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("""
                SELECT * FROM calls
                WHERE system_number=%s
                ORDER BY timestamp DESC
                LIMIT 50
            """, (system_number,))
            return [normalize_call(r) for r in c.fetchall()]
    finally:
        conn.close()

@app.post("/api/calls/update")
async def update_call(req: Request):
    data = await req.json()
    call_sid = data["call_sid"]
    status = data["appointment_status"]

    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("""
                SELECT phone_number, system_number, appointment_attempt
                FROM calls WHERE call_sid=%s
            """, (call_sid,))
            row = c.fetchone()

            c.execute("""
                UPDATE calls SET appointment_status=%s WHERE call_sid=%s
            """, (status, call_sid))
        conn.commit()
    finally:
        conn.close()

    if status == "rejected":
        trigger_rebooking_call(
            call_sid,
            row["phone_number"],
            row["system_number"],
            row["appointment_attempt"] + 1
        )

    return {"ok": True}

@app.get("/status")
async def status():
    return {"status": "online"}

# ---------------- RUN ----------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)


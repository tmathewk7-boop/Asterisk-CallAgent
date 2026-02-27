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
    return {
        "call_sid": row["call_sid"],
        "client_name": row.get("client_name", "Unknown"),
        "phone": row.get("phone_number"),
        "summary": row.get("summary", ""),
        "timestamp": row["timestamp"].strftime("%Y-%m-%d %I:%M %p")
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
    try:
        with conn.cursor() as c:
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
                call["customer"]["number"],
                call["phoneNumber"]["number"],
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
@app.post("/vapi/webhook")
async def vapi_webhook(req: Request):
    payload = await req.json()
    msg = payload.get("message", {})
    msg_type = msg.get("type")

    if msg_type == "tool-calls":
        results = []
        system_number = msg["call"]["phoneNumber"]["number"]

        for tool in msg.get("toolCalls", []):
            name = tool["function"]["name"]
            args = tool["function"]["arguments"]

            if name == "transfer_call":
                target = get_lawyer_by_name(system_number, args.get("person_name", ""))
                result = (
                    f"Transfer to {target}"
                    if target else "Lawyer not found"
                )
                results.append({"toolCallId": tool["id"], "result": result})

        return {"results": results}

    if msg_type == "end-of-call-report":
        call = msg["call"]
        save_call_log(call)

        appt = call.get("analysis", {}).get("appointment")
        if appt:
            save_appointment(
                call["id"],
                appt["status"],
                appt["proposed_time"],
                appt["attempt"],
                call["analysis"]["summary"]
            )

        return {"ok": True}

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


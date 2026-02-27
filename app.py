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
    """Formats the database row exactly how the Dashboard expects it."""
    return {
        # CRITICAL FIX: Change "call_sid" to "sid" so the dashboard can find the ID to delete it
        "sid": row["call_sid"], 
        "client_name": row.get("client_name", "Unknown"),
        "phone": row.get("phone_number"), 
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
def save_call_log(message: dict):
    # Vapi sends the info in the 'message' object
    call = message.get("call", {})
    customer = message.get("customer", {})
    
    # CRITICAL: This is the exact path to find the number bought from Vapi
    vapi_phone_obj = message.get("phoneNumber", {})
    system_num = vapi_phone_obj.get("number", "Unknown") 
    
    conn = get_db_connection()
    if not conn: return
    try:
        with conn.cursor() as c:
            c.execute("""
                INSERT INTO calls (
                    call_sid, phone_number, system_number,
                    timestamp, client_name, summary, full_transcript
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                call.get("id"),
                customer.get("number"),
                system_num, # This will now be '+18254352488' instead of 'Unknown'
                datetime.datetime.now(datetime.timezone.utc),
                call.get("analysis", {}).get("structuredData", {}).get("client_name", "Unknown"),
                call.get("analysis", {}).get("summary", "No summary"),
                call.get("transcript", "")
            ))
        conn.commit()
        print(f"SUCCESS: Saved call from {customer.get('number')} to {system_num}")
    except Exception as e:
        print(f"DATABASE INSERT ERROR: {e}")
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

@app.post("/")  # This catches the root POSTs from Vapi seen in your logs
async def handle_vapi_webhook(request: Request):
    try:
        data = await request.json()
        
        # Extracting the info Vapi sends at the end of a call
        message = data.get('message', {})
        call_data = message.get('call', {})
        
        # Check if this is the "end-of-call-report" from Vapi
        if message.get("type") == "end-of-call-report":
            # FIX: Use 'save_call_log' because 'save_to_db' is not defined in your script
            save_call_log(call_data)
            print(f"Call {call_data.get('id')} saved to database.")
            return {"status": "success"}
        
        return {"status": "ignored"}
    except Exception as e:
        print(f"Webhook Error: {e}")
        return {"status": "error", "message": str(e)}, 500

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

# ---------------- APPOINTMENT DASHBOARD ENDPOINTS ----------------

@app.get("/api/schedule/requests/{system_number}")
async def get_schedule_requests(system_number: str):
    """Fetches all pending appointments to display on the dashboard calendar."""
    conn = get_db_connection()
    if not conn: return []
    try:
        with conn.cursor() as c:
            # We map the database columns to the keys your EventCard expects
            c.execute("""
                SELECT call_sid as request_id, 
                       phone_number as caller_phone,
                       client_name as caller_name,
                       timestamp,
                       appointment_time as requested_time_str,
                       summary as reason
                FROM calls
                WHERE system_number=%s AND appointment_status='pending'
                ORDER BY timestamp DESC
            """, (system_number,))
            return c.fetchall()
    finally:
        conn.close()

@app.post("/api/schedule/accept/{call_sid}")
async def accept_schedule(call_sid: str, req: Request):
    """Approves the appointment and sends a confirmation SMS."""
    data = await req.json()
    caller_phone = data.get("caller_phone")
    
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("UPDATE calls SET appointment_status='accepted' WHERE call_sid=%s", (call_sid,))
            
            # Fetch the time to include in the text message
            c.execute("SELECT appointment_time FROM calls WHERE call_sid=%s", (call_sid,))
            row = c.fetchone()
            appt_time = row['appointment_time'] if row else "your requested time"
            
        conn.commit()
        
        # Send Confirmation SMS via Twilio
        if caller_phone:
            twilio_client.messages.create(
                body=f"Hello! Your appointment with Wellness Partners for {appt_time} has been confirmed. We look forward to seeing you!",
                from_=TWILIO_NUMBER, 
                to=caller_phone
            )
        return {"ok": True}
    except Exception as e:
        print(f"ACCEPT ERROR: {e}")
        return {"ok": False, "error": str(e)}
    finally:
        conn.close()

@app.post("/api/schedule/reject/{call_sid}")
async def reject_schedule(call_sid: str):
    """Rejects the appointment and triggers Riley to call them back."""
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("""
                SELECT phone_number, system_number, appointment_attempt 
                FROM calls WHERE call_sid=%s
            """, (call_sid,))
            row = c.fetchone()
            
            c.execute("UPDATE calls SET appointment_status='rejected' WHERE call_sid=%s", (call_sid,))
        conn.commit()
        
        # Trigger the AI Callback
        if row:
            trigger_rebooking_call(
                call_sid,
                row["phone_number"],
                row["system_number"],
                row.get("appointment_attempt", 1) + 1
            )
        return {"ok": True}
    except Exception as e:
        print(f"REJECT ERROR: {e}")
        return {"ok": False, "error": str(e)}
    finally:
        conn.close()

# ---------------- VAPI WEBHOOK ----------------
# Apply these decorators to catch ALL incoming Vapi signals
@app.post("/")
@app.post("/webhook")
@app.post("/vapi/webhook")
async def combined_vapi_webhook(req: Request):
    payload = await req.json()
    msg = payload.get("message", {})
    msg_type = msg.get("type", "unknown_type")

    print(f"🔔 VAPI EVENT RECEIVED: {msg_type}")

    # This is the event that contains the final phone number and summary
    if msg_type == "end-of-call-report":
        # CRITICAL FIX: Pass the ENTIRE 'msg' to save_call_log, not just 'call'
        save_call_log(msg) 
        
        # Handle appointments if they exist
        call_obj = msg.get("call", {})
        appt = call_obj.get("analysis", {}).get("appointment")
        if appt:
            save_appointment(
                call_obj.get("id"),
                appt.get("status"),
                appt.get("proposed_time"),
                appt.get("attempt", 1),
                call_obj.get("analysis", {}).get("summary", "")
            )
        return {"status": "success"}

    # Handle live tool calls (like transfers)
    if msg_type == "tool-calls":
        results = []
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

@app.post("/api/calls/delete")
async def delete_calls(req: DeleteCallsRequest):
    # CRITICAL: Prevent the crash if no IDs are sent
    if not req.call_sids:
        return {"ok": True, "message": "No selection to delete"}

    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="DB Connection Failed")
    
    try:
        with conn.cursor() as c:
            # Dynamically build the placeholders for the IN clause
            placeholders = ', '.join(['%s'] * len(req.call_sids))
            sql = f"DELETE FROM calls WHERE call_sid IN ({placeholders})"
            c.execute(sql, tuple(req.call_sids))
        conn.commit()
        return {"ok": True, "deleted": len(req.call_sids)}
    except Exception as e:
        print(f"DELETE ERROR: {e}")
        return {"ok": False, "error": str(e)}, 500
    finally:
        conn.close()

@app.get("/api/calls/{system_number}")
async def get_calls(system_number: str):
    conn = get_db_connection()
    if not conn: return []
    try:
        with conn.cursor() as c:
            # We temporarily removed the 'WHERE system_number=%s' line
            # so the dashboard will show ALL logs while we debug.
            c.execute("""
                SELECT * FROM calls
                ORDER BY timestamp DESC
                LIMIT 50
            """)
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


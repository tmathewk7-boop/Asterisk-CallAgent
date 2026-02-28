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
import json

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
    call = message.get("call", {})
    customer = message.get("customer", {})
    call_id = call.get("id")

    if not call_id:
        print("⚠️ WARNING: Webhook received without a Call ID. Ignoring.")
        return
        
    system_num = call.get("phoneNumber", {}).get("number", "+18254352488")
    customer_num = call.get("customer", {}).get("number") or message.get("customer", {}).get("number", "Unknown Caller")
    
    # 🚨 THE CRITICAL FIX: Looking at the top level of 'message', not 'call'
    analysis = message.get("analysis", {})
    artifact = message.get("artifact", {})
    
    structured = {}
    
    # 1. Check the old Vapi "Analysis" folder
    if analysis.get("structuredData"):
        structured = analysis.get("structuredData")
        
    # 2. Check the new Vapi "Structured Outputs" folder 
    new_outputs = artifact.get("structuredOutputs", {})
    if new_outputs:
        for key, val in new_outputs.items():
            if isinstance(val, dict) and "appointment_time" in val:
                structured = val
                break
        if not structured and isinstance(new_outputs, dict):
            structured = new_outputs

    print(f"🕵️ DATA FOUND: {structured}")
    
    # Extract the final values
    appt_time = structured.get("appointment_time") or structured.get("date")
    client_name = structured.get("client_name") or "Unknown"
    
    appt_status = "pending" if appt_time else "none"
    
    # Grab summary
    summary = analysis.get("summary") or message.get("summary", "No summary provided")
    
    conn = get_db_connection()
    if not conn: 
        print("❌ ERROR: Database connection failed.")
        return
        
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
                call_id,
                customer_num,
                system_num, 
                datetime.datetime.now(datetime.timezone.utc),
                client_name,
                summary,
                call.get("transcript", ""),
                appt_time,
                appt_status
            ))
        conn.commit()
        print(f"✅ SUCCESS: Logged call to DB. Name: {client_name}, Appt: {appt_time}")
    except Exception as e:
        print(f"❌ DATABASE INSERT ERROR: {e}")
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
    greeting = settings.get("greeting", "The clinic is currently unavailable.")

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

# ---------------- MAIN VAPI WEBHOOK ----------------
@app.post("/")
@app.post("/webhook")
@app.post("/vapi/webhook")
async def main_vapi_webhook(req: Request):
    try:
        payload = await req.json()
        msg = payload.get("message", {})
        msg_type = msg.get("type", "unknown_type")

        print(f"🔔 VAPI EVENT RECEIVED: {msg_type}")

        if msg_type == "end-of-call-report":
            # X-RAY DUMP: Print the correct top-level folders
            print("📦 TOP LEVEL FOLDERS:", list(msg.keys()))
            print("📦 RAW ANALYSIS PAYLOAD:")
            print(json.dumps(msg.get("analysis", {}), indent=2))
            print("📦 RAW ARTIFACT PAYLOAD:")
            print(json.dumps(msg.get("artifact", {}), indent=2))
            
            save_call_log(msg) 
            return {"status": "success"}

        if msg_type == "tool-calls":
            results = []
            system_number = msg.get("call", {}).get("phoneNumber", {}).get("number")
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
    except Exception as e:
        print(f"❌ WEBHOOK CRASH: {e}")
        return {"status": "error"}, 500

# ---------------- REBOOK CALLBACK ----------------
@app.post("/twilio/rebooking-complete")
async def rebooking_complete(req: Request):
    form = await req.form()
    call_sid = req.query_params["call_sid"]
    attempt = int(req.query_params["attempt"])
    system_number = req.query_params["system_number"]

    transcript = "Client requested new time" 
    new_time = "TBD" 

    save_appointment(
        call_sid,
        "pending",
        new_time,
        attempt,
        f"Rebooking attempt {attempt}: {transcript}"
    )

    return {"ok": True}

# ---------------- DASHBOARD API ENDPOINTS ----------------
class DeleteCallsRequest(BaseModel):
    call_sids: list[str]

@app.post("/api/calls/delete")
async def delete_calls(req: DeleteCallsRequest):
    if not req.call_sids:
        return {"ok": True, "message": "No selection to delete"}

    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="DB Connection Failed")
    
    try:
        with conn.cursor() as c:
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
            c.execute("""
                SELECT * FROM calls
                ORDER BY timestamp DESC
                LIMIT 50
            """)
            return [normalize_call(r) for r in c.fetchall()]
    finally:
        conn.close()

@app.get("/api/schedule/requests/{system_number}")
async def get_schedule_requests(system_number: str):
    conn = get_db_connection()
    if not conn: return []
    try:
        with conn.cursor() as c:
            c.execute("""
                SELECT call_sid as request_id, 
                       phone_number as caller_phone,
                       client_name as caller_name,
                       timestamp,
                       appointment_time as requested_time_str,
                       summary as reason
                FROM calls
                WHERE appointment_status='pending'
                ORDER BY timestamp DESC
            """)
            return c.fetchall()
    finally:
        conn.close()

@app.post("/api/schedule/accept/{call_sid}")
async def accept_schedule(call_sid: str, req: Request):
    data = await req.json()
    caller_phone = data.get("caller_phone")
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("UPDATE calls SET appointment_status='accepted' WHERE call_sid=%s", (call_sid,))
            c.execute("SELECT appointment_time FROM calls WHERE call_sid=%s", (call_sid,))
            row = c.fetchone()
            appt_time = row['appointment_time'] if row else "your requested time"
        conn.commit()
        
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

@app.post("/api/calls/update")
async def update_call(req: Request):
    data = await req.json()
    call_sid = data["call_sid"]
    status = data["appointment_status"]
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("UPDATE calls SET appointment_status=%s WHERE call_sid=%s", (status, call_sid))
        conn.commit()
    finally:
        conn.close()
    return {"ok": True}

@app.get("/status")
async def status():
    return {"status": "online"}

# ---------------- RUN ----------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)

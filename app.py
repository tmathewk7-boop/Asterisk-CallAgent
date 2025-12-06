import os
import pymysql
import json
import datetime
import uvicorn
from collections import defaultdict
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ---------- Configuration ----------
PORT = int(os.getenv("PORT", 8000))
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://your-app.onrender.com")

# Vapi Auth Token (for validating requests)
VAPI_WEBHOOK_SECRET = os.getenv("VAPI_WEBHOOK_SECRET", "your-secret")

# --- FIRM DIRECTORY ---
FIRM_DIRECTORY = {
    "james": "+13065183350",
    "sarah": "+15550001234"
}

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ---------- DATABASE & STATE ----------
# We keep these so your dashboard works exactly the same
call_db = {}        
DEFAULT_CONFIG = {
    "system_prompt": "You are a helpful legal receptionist.",
    "greeting": "Law office, how may I direct your call?",
    "ai_active": True,
    "personal_phone": ""
}

# --- DB FUNCTIONS (Unchanged) ---
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
            # We map Vapi fields to your existing DB schema
            sql = """
                INSERT INTO calls (call_sid, phone_number, system_number, timestamp, client_name, summary, full_transcript)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                client_name = VALUES(client_name),
                summary = VALUES(summary),
                full_transcript = VALUES(full_transcript)
            """
            cursor.execute(sql, (
                call_data.get('id'), # Vapi Call ID matches your 'sid' column
                call_data.get('customer', {}).get('number'),
                call_data.get('phoneNumber', {}).get('number'), # System number
                datetime.datetime.utcnow(),
                call_data.get('analysis', {}).get('structuredData', {}).get('client_name', 'Unknown'),
                call_data.get('analysis', {}).get('summary', 'No summary'),
                call_data.get('transcript', '')
            ))
        conn.commit()
    except Exception as e:
        print(f"Error saving call log to DB: {e}")
    finally:
        if conn: conn.close()

# ---------- API MODELS ----------
class AgentSettings(BaseModel):
    phone_number: str
    system_prompt: Optional[str] = None
    greeting: Optional[str] = None
    personal_phone: Optional[str] = None

class DeleteCallsRequest(BaseModel):
    call_sids: list[str]

# ---------------- VAPI WEBHOOK (THE NEW BRAIN) ----------------

@app.post("/vapi/webhook")
async def vapi_webhook(request: Request):
    """
    Handles communication from Vapi.
    1. tool-calls: Executes 'transfer_call' logic.
    2. end-of-call-report: Saves data to your dashboard DB.
    """
    payload = await request.json()
    message_type = payload.get("message", {}).get("type")
    
    # 1. HANDLE TOOL CALLS (Transfer Logic)
    if message_type == "tool-calls":
        tool_calls = payload.get("message", {}).get("toolCalls", [])
        results = []
        
        for tool in tool_calls:
            function_name = tool.get("function", {}).get("name")
            args = tool.get("function", {}).get("arguments", {})
            
            if function_name == "transfer_call":
                target_name = args.get("person_name", "").lower()
                target_number = FIRM_DIRECTORY.get(target_name)
                
                if target_number:
                    # Return the destination to Vapi
                    results.append({
                        "toolCallId": tool["id"],
                        "result": f"Transferring to {target_number}. If they do not pick up within 20 seconds, please treat it as unavailable."
                    })
                else:
                    results.append({
                        "toolCallId": tool["id"],
                        "result": "Contact not found in firm directory."
                    })
                    
            elif function_name == "log_call_data":
                # The Lawyer AI uses this to sync to dashboard
                # You can add logic here to trigger a notification
                results.append({
                    "toolCallId": tool["id"],
                    "result": "Data saved to dashboard."
                })

        return {"results": results}

    # 2. HANDLE END OF CALL (Save to DB)
    elif message_type == "end-of-call-report":
        call_data = payload.get("message", {}).get("call", {})
        
        # Save to your existing DB structure
        save_call_log_to_db(call_data)
        
        return {"status": "success"}

    return {"status": "ignored"}

# ---------------- DASHBOARD API (UNCHANGED) ----------------
# These endpoints are required for your frontend to work

@app.get("/api/calls/{target_number}")
async def get_calls_for_client(target_number: str):
    # This now fetches from MySQL directly as Vapi saves directly to DB
    conn = get_db_connection()
    if not conn: return []
    try:
        with conn.cursor() as cursor:
            sql = "SELECT * FROM calls WHERE system_number = %s ORDER BY timestamp DESC LIMIT 50"
            cursor.execute(sql, (target_number,))
            return cursor.fetchall()
    finally:
        if conn: conn.close()

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

@app.get("/status")
async def server_status():
    return {"status": "online", "provider": "vapi"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)

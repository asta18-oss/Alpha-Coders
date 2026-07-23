"""
IntelliFetch
------------
Single-file FastAPI + SQLite backend that matches students to opportunities,
now with a WhatsApp bot powered by Gemini (via the official google-genai SDK).

Setup:
    pip install fastapi uvicorn google-genai twilio python-multipart

Environment:
    export GEMINI_API_KEY="your_gemini_api_key_here"   # required for the webhook route

Run the API:
    python app.py

Seed the database with dummy data:
    python app.py seed

Try it:
    GET  http://127.0.0.1:8000/api/opportunities?student_id=1
    POST http://127.0.0.1:8000/api/whatsapp/webhook   (see Twilio section below)
"""

import json
import sys
from typing import List, Optional
import sqlite3

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Form
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
from twilio.twiml.messaging_response import MessagingResponse

DB_PATH = "intellifetch.db"
GEMINI_MODEL = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the tables if they don't already exist."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT,
            skills TEXT,
            preferred_places TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS opportunities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            company TEXT,
            required_skills TEXT,
            location TEXT,
            deadline TEXT
        )
    """)

    conn.commit()
    conn.close()


def seed_db() -> None:
    """
    Wipe and repopulate the database with:
      - 1 dummy student, explicitly given id=1 (Rahul)
      - 5 dummy opportunities (mix of Remote, Mumbai, Bengaluru)
    """
    init_db()
    conn = get_connection()
    cur = conn.cursor()

    # Clear existing rows so the script is safely re-runnable
    cur.execute("DELETE FROM students")
    cur.execute("DELETE FROM opportunities")
    # Reset autoincrement counters so the explicit id=1 below is guaranteed
    # to stick and future inserts continue predictably after it.
    cur.execute("DELETE FROM sqlite_sequence WHERE name IN ('students', 'opportunities')")

    # Explicitly insert the dummy student profile with id=1, matching the
    # phone number we'll use to test the WhatsApp webhook below.
    cur.execute(
        """
        INSERT INTO students (id, name, phone, skills, preferred_places)
        VALUES (?, ?, ?, ?, ?)
        """,
        (1, "Rahul", "+919876543210", "React, Python", "Mumbai, Bengaluru, Remote"),
    )

    dummy_jobs = [
        ("Backend Intern", "Zenith Tech", "Python, FastAPI", "Remote", "2026-08-15"),
        ("Data Analyst Intern", "Bluepeak Analytics", "SQL, Excel, Python", "Mumbai", "2026-08-20"),
        ("Frontend Developer", "Nimbus Labs", "React, JavaScript", "Bengaluru", "2026-08-25"),
        ("DevOps Intern", "CloudNine Systems", "Docker, Linux, AWS", "Remote", "2026-09-01"),
        ("Full Stack Intern", "OrbitSoft", "React, Node.js, MongoDB", "Bengaluru", "2026-09-05"),
    ]
    cur.executemany(
        """
        INSERT INTO opportunities (title, company, required_skills, location, deadline)
        VALUES (?, ?, ?, ?, ?)
        """,
        dummy_jobs,
    )

    conn.commit()
    conn.close()
    print("Database seeded: student id=1 (Rahul), 5 opportunities.")


def get_student_by_phone(phone: str) -> Optional[sqlite3.Row]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM students WHERE phone = ?", (phone,))
    row = cur.fetchone()
    conn.close()
    return row


def student_preferred_locations(student: sqlite3.Row) -> List[str]:
    """Comma-separated preferred_places -> clean list, always including Remote."""
    raw = student["preferred_places"] or ""
    places = [p.strip() for p in raw.split(",") if p.strip()]
    if "Remote" not in places:
        places.append("Remote")
    return places


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="IntelliFetch",
    description="Matches students to opportunities, in-app and over WhatsApp.",
    lifespan=lifespan,
)
@app.get("/")
def read_index():
    """Serves the index.html file natively from the server root."""
    return FileResponse("index.html")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your index.html file to communicate securely
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Student Signup
# ---------------------------------------------------------------------------

class StudentCreate(BaseModel):
    name: str
    phone: str
    skills: str
    preferred_places: str


class StudentResponse(BaseModel):
    id: int
    name: str
    phone: Optional[str] = None
    skills: Optional[str] = None
    preferred_places: Optional[str] = None


@app.post("/api/students", response_model=StudentResponse)
def create_student(student: StudentCreate):
    conn = get_connection()
    cur = conn.cursor()

    # Check if phone number already exists
    cur.execute(
        "SELECT * FROM students WHERE phone = ?",
        (student.phone,)
    )

    existing_student = cur.fetchone()

    if existing_student:
        conn.close()
        raise HTTPException(
            status_code=400,
            detail="A student with this phone number already exists"
        )

    # Create new student
    cur.execute(
        """
        INSERT INTO students
        (name, phone, skills, preferred_places)
        VALUES (?, ?, ?, ?)
        """,
        (
            student.name,
            student.phone,
            student.skills,
            student.preferred_places
        )
    )

    student_id = cur.lastrowid

    conn.commit()

    cur.execute(
        "SELECT * FROM students WHERE id = ?",
        (student_id,)
    )

    new_student = cur.fetchone()

    conn.close()

    return dict(new_student)

@app.put("/api/students/{student_id}/preferences")
def update_student_preferences(
    student_id: int,
    skills: str,
    preferred_places: str
):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        UPDATE students
        SET skills = ?, preferred_places = ?
        WHERE id = ?
        """,
        (skills, preferred_places, student_id)
    )

    if cur.rowcount == 0:
        conn.close()
        raise HTTPException(
            status_code=404,
            detail="Student not found"
        )

    conn.commit()

    cur.execute(
        "SELECT * FROM students WHERE id = ?",
        (student_id,)
    )

    student = cur.fetchone()
    conn.close()

    return dict(student)

class Opportunity(BaseModel):
    id: int
    title: str
    company: Optional[str] = None
    required_skills: Optional[str] = None
    location: Optional[str] = None
    deadline: Optional[str] = None


@app.get("/api/opportunities")
def get_opportunities(student_id: int = Query(..., description="ID of the student")):
    """
    Look up the student's preferred_places, then return every opportunity
    whose location matches one of those cities OR is 'Remote'.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM students WHERE id = ?", (student_id,))
    student = cur.fetchone()

    if student is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Student with id {student_id} not found")

    preferred_places = student_preferred_locations(student)

    placeholders = ",".join("?" for _ in preferred_places)
    cur.execute(
        f"SELECT * FROM opportunities WHERE location IN ({placeholders})",
        preferred_places,
    )
    rows = cur.fetchall()
    conn.close()

    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# WhatsApp integration (Twilio webhook + Gemini query parsing)
# ---------------------------------------------------------------------------

class ParsedQuery(BaseModel):
    """Structured shape we ask Gemini to extract from a free-text WhatsApp message."""
    skills: List[str] = []
    # Only set if the user explicitly names a city/mode in their message
    # (e.g. "remote only", "anything in Bengaluru"); overrides the student's
    # saved preferences for that single query when present.
    location: Optional[str] = None


def parse_query_with_gemini(message: str) -> ParsedQuery:
    """
    Ask Gemini to turn a free-text WhatsApp message into structured filters.
    Falls back to an empty ParsedQuery (triggering a raw keyword search)
    if the model call fails for any reason.
    """
    try:
        # Constructed per-call (not at import time) so a missing/invalid
        # GEMINI_API_KEY only fails this one request instead of crashing
        # the whole app at startup.
        client = genai.Client()
        prompt = (
            "A student is texting a job/internship-matching bot on WhatsApp. "
            "Extract the skills they're looking for and, only if explicitly "
            "mentioned, a specific location or 'Remote'. "
            f'Message: "{message}"'
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ParsedQuery,
            ),
        )
        return ParsedQuery.model_validate(json.loads(response.text))
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash the webhook
        print(f"[Gemini parse error] {exc}")
        return ParsedQuery()


def search_opportunities(
    skills: List[str],
    locations: Optional[List[str]],
    raw_message: str,
) -> List[sqlite3.Row]:
    """
    Filter opportunities by parsed skills AND/OR allowed locations.
    If Gemini gave us nothing usable, fall back to a naive keyword search
    across the raw message text.
    """
    conn = get_connection()
    cur = conn.cursor()

    clauses, params = [], []

    if skills:
        clauses.append("(" + " OR ".join(["required_skills LIKE ?"] * len(skills)) + ")")
        params.extend(f"%{skill}%" for skill in skills)

    if locations:
        clauses.append("(" + " OR ".join(["location = ?"] * len(locations)) + ")")
        params.extend(locations)

    if clauses:
        query = "SELECT * FROM opportunities WHERE " + " AND ".join(clauses)
    else:
        like = f"%{raw_message.strip()}%"
        query = (
            "SELECT * FROM opportunities "
            "WHERE title LIKE ? OR required_skills LIKE ? OR location LIKE ?"
        )
        params = [like, like, like]

    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return rows


def format_whatsapp_reply(rows: List[sqlite3.Row], student: Optional[sqlite3.Row]) -> str:
    """Build a clean, WhatsApp-friendly bulleted reply (WhatsApp uses *bold*)."""
    greeting = f"Hi {student['name']}! " if student else "Hi! "

    if not rows:
        return (
            f"{greeting}I couldn't find any matching opportunities right now. "
            "Try mentioning a skill (e.g. Python) or a city (e.g. Mumbai, Remote)."
        )

    lines = [f"{greeting}here's what I found for you:"]
    for r in rows:
        lines.append(
            f"• *{r['title']}* — {r['company']}\n"
            f"   📍 {r['location']}  |  🛠 {r['required_skills']}\n"
            f"   ⏳ Deadline: {r['deadline']}"
        )
    return "\n\n".join(lines)


@app.post("/api/whatsapp/webhook")
async def whatsapp_webhook(Body: str = Form(...), From: str = Form(...)):
    """
    Twilio calls this with form-encoded `Body` (the message text) and
    `From` (e.g. "whatsapp:+919876543210") on every incoming WhatsApp message.

    Flow:
      1. Identify the student by phone number, if registered.
      2. Ask Gemini to parse the free-text message into skills/location.
      3. Query opportunities using the parsed skills, restricted to the
         student's preferred cities (or Remote) unless the message names
         an explicit location itself.
      4. Reply with a formatted bulleted list via Twilio's TwiML.
    """
    phone = From.replace("whatsapp:", "").strip()
    student = get_student_by_phone(phone)

    parsed = parse_query_with_gemini(Body)

    if parsed.location:
        locations = [parsed.location]
    elif student:
        locations = student_preferred_locations(student)
    else:
        locations = None

    rows = search_opportunities(parsed.skills, locations, Body)
    reply_text = format_whatsapp_reply(rows, student)

    twiml = MessagingResponse()
    twiml.message(reply_text)

    return Response(content=str(twiml), media_type="application/xml")


# ---------------------------------------------------------------------------
# Entry point: `python app.py` runs the server,
#              `python app.py seed` seeds the database.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "seed":
        seed_db()
    else:
        import uvicorn

        init_db()
        uvicorn.run(app, host="0.0.0.0", port=8000)

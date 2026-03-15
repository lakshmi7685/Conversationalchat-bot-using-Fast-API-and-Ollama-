from fastapi import FastAPI
from pydantic import BaseModel
import psycopg2
import requests
import json

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"  # ✅ Single source of truth


# -----------------------------
# DATABASE CONNECTION
# -----------------------------

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="Conversation_bot",
        user="postgres",
        password="xyz",
        port="5432"
    )


# -----------------------------
# REQUEST MODEL
# -----------------------------

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str


# -----------------------------
# GET HISTORY
# -----------------------------

def get_history(user_id: str, session_id: str):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT conversation_history
        FROM session_management
        WHERE user_id = %s AND session_id = %s
        ORDER BY id DESC
        LIMIT 1
        """,
        (user_id, session_id),
    )

    result = cur.fetchone()
    cur.close()
    conn.close()

    if result is None:
        return []

    history = result[0]
    if isinstance(history, str):
        history = json.loads(history)

    return history


# -----------------------------
# SAVE HISTORY  ✅ UPSERT instead of always inserting
# -----------------------------

def save_history(user_id: str, session_id: str, history: list):
    conn = get_connection()
    cur = conn.cursor()

    # UPSERT into session_management
    cur.execute(
        """
        INSERT INTO session_management (user_id, session_id, conversation_history)
        VALUES (%s, %s, %s)
        ON CONFLICT (user_id, session_id)
        DO UPDATE SET conversation_history = EXCLUDED.conversation_history
        """,
        (user_id, session_id, json.dumps(history)),
    )

    # UPSERT into user_information
    cur.execute(
        """
        INSERT INTO user_information (user_id, session_id, conversation_history)
        VALUES (%s, %s, %s)
        ON CONFLICT (user_id)
        DO UPDATE SET
            session_id = EXCLUDED.session_id,
            conversation_history = EXCLUDED.conversation_history
        """,
        (user_id, session_id, json.dumps(history)),
    )

    conn.commit()
    cur.close()
    conn.close()


# -----------------------------
# BUILD SYSTEM PROMPT WITH HISTORY SUMMARY  ✅ NEW
# -----------------------------

def build_system_prompt(history: list) -> str:
    base_prompt = (
        "You are a conversational assistant with access to the full conversation history. "
        "When the user asks what they previously asked or discussed, refer to the history below.\n\n"
    )

    if not history:
        return base_prompt + "No previous conversation history."

    # Summarize past user questions explicitly
    past_user_messages = [
        f"{i+1}. {msg['content']}"
        for i, msg in enumerate(history)
        if msg["role"] == "user"
    ]

    if past_user_messages:
        base_prompt += "Previous questions asked by the user:\n"
        base_prompt += "\n".join(past_user_messages)

    return base_prompt


# -----------------------------
# OLLAMA CALL  ✅ Uses MODEL constant + enriched system prompt
# -----------------------------

def call_ollama(history: list) -> str:
    system_prompt = build_system_prompt(history)

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]
    messages.extend(history)

    payload = {
        "model": MODEL,  # ✅ uses the constant
        "messages": messages,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()  # ✅ raises error on bad status
    return response.json()["message"]["content"]

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": str(exc.body)}

    )

# -----------------------------
# CHAT ENDPOINT

# -----------------------------
# INTENT DETECTION
# -----------------------------

HISTORY_RECALL_PHRASES = [
    "what did i ask",
    "what were my questions",
    "what was my question",
    "what are my questions",
    "my previous question",
    "my questions today",
    "questions for today",
    "what have i asked",
    "list my questions",
]

def is_history_recall(message: str) -> bool:
    msg = message.lower().strip()
    return any(phrase in msg for phrase in HISTORY_RECALL_PHRASES)


def get_user_questions(history: list) -> str:
    questions = [
        f"{i+1}. {msg['content']}"
        for i, msg in enumerate(history)
        if msg["role"] == "user"
    ]
    if not questions:
        return "You haven't asked any questions yet."
    return "Here are your questions so far:\n" + "\n".join(questions)


# -----------------------------
# CHAT ENDPOINT  ✅ updated
# -----------------------------

@app.post("/chat")
def chat(request: ChatRequest):
    history = get_history(request.user_id, request.session_id)

    # Add user message
    history.append({
        "role": "user",
        "content": request.message
    })

    # ✅ Intercept history-recall questions — answer from code, not model
    if is_history_recall(request.message):
        # Exclude the current message itself from the list
        past_history = history[:-1]
        reply = get_user_questions(past_history)
    else:
        reply = call_ollama(history)

    # Add assistant reply
    history.append({
        "role": "assistant",
        "content": reply
    })

    # Save updated history
    save_history(request.user_id, request.session_id, history)

    return {
        "user_id": request.user_id,
        "session_id": request.session_id,
        "response": reply
    }




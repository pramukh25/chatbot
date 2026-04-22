import asyncio
import difflib
import os
import re
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import List, Optional

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag import RAGEngine

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv(Path(__file__).parent / ".env", override=True)

OPENROUTER_API_KEY: str = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PORT = int(os.getenv("PORT"))
ENABLE_FILES_WATCH = os.getenv("ENABLE_FILES_WATCH", "true").strip().lower() not in {"0", "false", "no"}
FILES_WATCH_INTERVAL_SECONDS = float(os.getenv("FILES_WATCH_INTERVAL_SECONDS", "5"))

rag = RAGEngine(files_dir=str(Path(__file__).parent / "files"))

# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


async def _watch_files_loop() -> None:
    while True:
        await asyncio.sleep(FILES_WATCH_INTERVAL_SECONDS)
        try:
            changed = await asyncio.to_thread(rag.sync_index_if_needed)
            if changed:
                print(f"[watcher] Index rebuilt. {len(rag.chunks)} chunks loaded.")
        except Exception as exc:
            print(f"[watcher] Warning: index sync failed: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    count = rag.load_documents()
    print(f"[startup] Indexed {count} chunks from documents.")

    watch_task = None
    if ENABLE_FILES_WATCH and FILES_WATCH_INTERVAL_SECONDS > 0:
        watch_task = asyncio.create_task(_watch_files_loop())
        print(f"[startup] Files watcher enabled ({FILES_WATCH_INTERVAL_SECONDS}s interval).")

    try:
        yield
    finally:
        if watch_task:
            watch_task.cancel()
            with suppress(asyncio.CancelledError):
                await watch_task
        rag.close()


app = FastAPI(
    title="RAG Chatbot API",
    description=(
        "Ask questions about your PDF documents. "
        "Answers are grounded in the document content and include source references."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer.")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of context chunks to retrieve.")


class ChatResponse(BaseModel):
    answer: str


class DocumentInfo(BaseModel):
    name: str
    chunks: int


class DocumentsResponse(BaseModel):
    total_chunks: int
    documents: List[DocumentInfo]


# ---------------------------------------------------------------------------
# Title / keyword → question expansion map
# ---------------------------------------------------------------------------

_TITLE_TO_QUESTION: dict[str, str] = {
    # --- Profile ---
    "profile update": "How can I update my profile information?",
    "profile": "How can I update my profile information?",
    "update profile": "How can I update my profile information?",
    # --- Technical Issues ---
    "technical issues": "What should I do if I encounter technical issues?",
    "technical issue": "What should I do if I encounter technical issues?",
    "technical": "What should I do if I encounter technical issues?",
    "issue": "What should I do if I encounter technical issues?",
    # --- Company Policies ---
    "company policies": "How can I access company policies and resources?",
    "company policy": "How can I access company policies and resources?",
    "policy": "How can I access company policies and resources?",
    "policies": "How can I access company policies and resources?",
    # --- Benefits Enrollment ---
    "benefits enrollment": "How do I access my benefits enrollment information?",
    "benefits": "How do I access my benefits enrollment information?",
    "enrollment": "How do I access my benefits enrollment information?",
    # --- PMS ---
    "performance management (pms)": "How do I access my Performance Management (PMS)?",
    "performance management": "How do I access my Performance Management (PMS)?",
    "pms": "How do I access my Performance Management (PMS)?",
    "performance": "How do I access my Performance Management (PMS)?",
    # --- Payslips ---
    "payslips, form16 and form 12a": "How can I access my Payslips, Form16 and Form 12A?",
    "payslips, form 16 and form 12a": "How can I access my Payslips, Form16 and Form 12A?",
    "payslip": "How can I access my Payslips, Form16 and Form 12A?",
    "payslips": "How can I access my Payslips, Form16 and Form 12A?",
    "pay slip": "How can I access my Payslips, Form16 and Form 12A?",
    "pay slips": "How can I access my Payslips, Form16 and Form 12A?",
    "salary slip": "How can I access my Payslips, Form16 and Form 12A?",
    "form 16": "How can I access my Payslips, Form16 and Form 12A?",
    "form16": "How can I access my Payslips, Form16 and Form 12A?",
    "form 12a": "How can I access my Payslips, Form16 and Form 12A?",
    "form12a": "How can I access my Payslips, Form16 and Form 12A?",
    "mypay": "How can I access my Payslips, Form16 and Form 12A?",
    # --- Emergency Contacts ---
    "emergency contacts": "How do I access emergency contact information?",
    "emergency contact": "How do I access emergency contact information?",
    "emergency": "How do I access emergency contact information?",
    # --- Leave & Holidays ---
    "leave & holidays": "How can I find company holidays and leave policies?",
    "leave and holidays": "How can I find company holidays and leave policies?",
    "leave": "How can I find company holidays and leave policies?",
    "holidays": "How can I find company holidays and leave policies?",
    "holiday": "How can I find company holidays and leave policies?",
    "leave balance": "How can I find company holidays and leave policies?",
    "calendar": "How can I find company holidays and leave policies?",
    # --- Travel ---
    "travel requests": "How do I raise a Travel request?",
    "travel request": "How do I raise a Travel request?",
    "travel": "How do I raise a Travel request?",
    # --- Idea Submissions ---
    "idea submissions": "How do I submit an idea?",
    "idea submission": "How do I submit an idea?",
    "idea": "How do I submit an idea?",
    "ideas": "How do I submit an idea?",
    "ideavault": "How do I submit an idea?",
    # --- Report an Issue ---
    "report an issue": "How do I report an issue, security risk, or non-compliance anonymously?",
    "report issue": "How do I report an issue, security risk, or non-compliance anonymously?",
    "sos": "How do I report an issue, security risk, or non-compliance anonymously?",
    "non compliance": "How do I report an issue, security risk, or non-compliance anonymously?",
    "security risk": "How do I report an issue, security risk, or non-compliance anonymously?",
    # --- Login / Access ---
    "login, access": "How do I login to the app? How can I reset my login credentials?",
    "login": "How do I login to the app? How can I reset my login credentials?",
    "access": "How do I login to the app? How can I reset my login credentials?",
    "sign in": "How do I login to the app? How can I reset my login credentials?",
    "password": "How do I login to the app? How can I reset my login credentials?",
    "reset password": "How do I login to the app? How can I reset my login credentials?",
    # --- Features / Tools ---
    "features, tools": "What features are available in the app? How do I customize app tools?",
    "features": "What features are available in the app? How do I customize app tools?",
    "tools": "What features are available in the app? How do I customize app tools?",
    # --- News / Announcements ---
    "news, announcements": "How do I stay updated with company news and announcements?",
    "news": "How do I stay updated with company news and announcements?",
    "announcements": "How do I stay updated with company news and announcements?",
    "announcement": "How do I stay updated with company news and announcements?",
    # --- Feedback / Anonymous / SOS ---
    "feedback, anonymous,sos": "How can I provide anonymous feedback?",
    "feedback, anonymous, sos": "How can I provide anonymous feedback?",
    "feedback": "How can I provide anonymous feedback?",
    "anonymous feedback": "How can I provide anonymous feedback?",
    "anonymous": "How can I provide anonymous feedback?",
    # --- Issues / Concerns ---
    "issues, concerns": "What is the process for reporting workplace issues or concerns?",
    "issues": "What is the process for reporting workplace issues or concerns?",
    "concerns": "What is the process for reporting workplace issues or concerns?",
    "concern": "What is the process for reporting workplace issues or concerns?",
    "workplace issue": "What is the process for reporting workplace issues or concerns?",
    # --- Schedule / Work ---
    "schedule, work": "How can I manage my work schedule?",
    "schedule": "How can I manage my work schedule?",
    "work schedule": "How can I manage my work schedule?",
    # --- Projects / Assignments / Deadlines ---
    "projects, assignments, deadlines": "How do I access my project assignments and deadlines?",
    "projects": "How do I access my project assignments and deadlines?",
    "assignments": "How do I access my project assignments and deadlines?",
    "deadlines": "How do I access my project assignments and deadlines?",
}


def _expand_query(question: str) -> str:
    """
    If the user's input matches a known FAQ title or keyword group
    (exactly or with minor spelling mistakes), replace it with the
    corresponding full question for better retrieval.
    """
    normalized = question.strip().lower()
    # Exact match first
    if normalized in _TITLE_TO_QUESTION:
        return _TITLE_TO_QUESTION[normalized]
    # Fuzzy match — cutoff 0.75 tolerates ~1-2 character typos
    keys = list(_TITLE_TO_QUESTION.keys())
    matches = difflib.get_close_matches(normalized, keys, n=1, cutoff=0.75)
    if matches:
        return _TITLE_TO_QUESTION[matches[0]]
    return question


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_prompt(question: str, context_chunks: list) -> str:
    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        context_parts.append(
            f"[{i}] Document: {chunk['document']}  |  Page: {chunk['page']}\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    return f"""You are ArcheBot, Arche's friendly and knowledgeable HR virtual assistant. You help employees confidently with questions about company policies, leave, attendance, benefits, and workplace guidelines.

Rules:
- Answer only from the context below; do not use outside knowledge.
- Speak in a warm, professional, and confident HR tone — like a helpful HR team member, not a search engine.
- Do NOT mention document names, file names, page numbers, or phrase answers as "According to [document]...". The source references are shown separately to the user.
- If the answer cannot be found in the context, respond with a varied, friendly message. Do NOT use the same phrasing every time. Rotate naturally between styles like:
    * "That's a great question! I don't have the details on that just yet — your best bet would be to connect with the People Excellence team directly."
    * "Hmm, I couldn't find anything on that in what I have right now. I'd suggest reaching out to the People Excellence team — they'll be able to help!"
    * "Good question! It looks like I don't have specific details on that right now. Feel free to reach out to the People Excellence team for the most accurate guidance."
    * "I'm not able to find a clear answer on that from the information I have. The People Excellence team will definitely be able to point you in the right direction!"
  Keep the tone warm, human, and encouraging — never robotic or repetitive.
- Keep answers concise, clear, and actionable.
- Use bullet points for multi-part answers where appropriate.
- Do NOT use any markdown formatting. No bold (**text**), no italics (*text*), no headers (#), no backticks. Plain text only.

=== CONTEXT START ===
{context}
=== CONTEXT END ===

Employee Question: {question}

ArcheBot:"""


async def _call_openrouter(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(OPENROUTER_URL, headers=headers, json=payload)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter error {resp.status_code}: {resp.text}",
        )

    data = resp.json()
    try:
        answer = data["choices"][0]["message"]["content"]
        return _strip_markdown(answer)
    except (KeyError, IndexError) as exc:
        raise HTTPException(status_code=502, detail=f"Unexpected OpenRouter response: {data}") from exc


def _strip_markdown(text: str) -> str:
    # Remove bold and italic markers
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    # Replace bullet points (* or -) at start of line with a dash
    text = re.sub(r"^\s*[\*\-]\s+", "- ", text, flags=re.MULTILINE)
    # Remove headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove backticks
    text = re.sub(r"`+(.+?)`+", r"\1", text)
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/chat", response_model=ChatResponse, summary="Ask a question")
async def chat(request: ChatRequest):
    """
    Send a question and get an answer grounded in the indexed PDF documents,
    along with the source references used to generate the answer.
    """
    if not rag.chunks:
        raise HTTPException(
            status_code=503,
            detail="No documents are indexed. Add PDFs to the 'files/' directory and call POST /reload.",
        )

    expanded_question = _expand_query(request.question)
    results = rag.search(expanded_question, top_k=request.top_k)
    if not results:
        raise HTTPException(status_code=503, detail="Document index is empty.")

    prompt = _build_prompt(expanded_question, results)
    answer = await _call_openrouter(prompt)
    answer = answer.strip("\n").strip()

    return ChatResponse(answer=answer)


@app.get("/documents", response_model=DocumentsResponse, summary="List indexed documents")
async def list_documents():
    """Return all indexed PDF files and how many chunks each contributed."""
    docs = rag.list_documents()
    return DocumentsResponse(
        total_chunks=len(rag.chunks),
        documents=[DocumentInfo(**d) for d in docs],
    )


@app.post("/reload", summary="Reload documents from disk")
async def reload_documents():
    """
    Re-scan the 'files/' directory and rebuild the index.
    Call this after adding or removing PDF files without restarting the server.
    """
    count = rag.load_documents(force_rebuild=True)
    return {"message": f"Index rebuilt. {count} chunks loaded.", "total_chunks": count}


@app.get("/health", summary="Health check")
async def health():
    return {
        "status": "ok",
        "model": OPENROUTER_MODEL,
        "chunks_indexed": len(rag.chunks),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)

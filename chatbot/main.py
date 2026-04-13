import os
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag import RAGEngine

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY: str = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

rag = RAGEngine(files_dir="files")

# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    count = rag.load_documents()
    print(f"[startup] Indexed {count} chunks from documents.")
    yield


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
- If the answer cannot be found in the context, respond warmly: "That's a great question! Unfortunately, I don't have specific information on that right now. I'd recommend reaching out to the HR team directly for clarification."
- Keep answers concise, clear, and actionable.
- Use bullet points for multi-part answers where appropriate.

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
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise HTTPException(status_code=502, detail=f"Unexpected OpenRouter response: {data}") from exc


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

    results = rag.search(request.question, top_k=request.top_k)
    if not results:
        raise HTTPException(status_code=503, detail="Document index is empty.")

    prompt = _build_prompt(request.question, results)
    answer = await _call_openrouter(prompt)

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
    count = rag.load_documents()
    return {"message": f"Index rebuilt. {count} chunks loaded.", "total_chunks": count}


@app.get("/health", summary="Health check")
async def health():
    return {
        "status": "ok",
        "model": OPENROUTER_MODEL,
        "chunks_indexed": len(rag.chunks),
    }

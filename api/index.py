import json
import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

DATA = Path(__file__).parent / "data"
EMBEDDING_MODEL = json.loads((DATA / "meta.json").read_text())["embedding_model"]
CHUNKS = json.loads((DATA / "chunks.json").read_text())
EMB = np.load(DATA / "embeddings.npy").astype(np.float32)
EMB /= np.linalg.norm(EMB, axis=1, keepdims=True)

TOP_K = 8
MEMORY_WINDOW = 5  # last 5 question/answer pairs

SYSTEM_PROMPT = (
    "You are the IronStore enterprise assistant. Answer ONLY using the context below. "
    "If the answer is not in the context, say you don't know. "
    "Cite the source file name for each fact.\n\n"
    "Use the chat history only to understand what the user is referring to in follow-up "
    "questions (for example 'it', 'that', 'and for part-time staff?'). "
    "Facts in your answer must still come from the context, not from the history.\n\n"
    "Chat history:\n{history}\n\nContext:\n{context}"
)

app = FastAPI(title="IronStore Assistant API")

# Only needed for local development (frontend on :5173); on Vercel both share one origin
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])


router = APIRouter()


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # Serverless functions are stateless, so the browser sends the recent conversation
    history: list[Turn] = Field(default_factory=list, max_length=2 * MEMORY_WINDOW)


class ChatResponse(BaseModel):
    answer: str


def clean_key(raw):
    # Tolerate a pasted '.env' line such as OPENAI_API_KEY="sk-..."
    return raw.split("=")[-1].strip().strip("\"'")


def retrieve(client, question):
    q = client.embeddings.create(model=EMBEDDING_MODEL, input=question).data[0].embedding
    q = np.array(q, dtype=np.float32)
    q /= np.linalg.norm(q)
    top = np.argsort(EMB @ q)[::-1][:TOP_K]
    return [CHUNKS[i] for i in top]


def format_docs(chunks):
    parts = []
    for c in chunks:
        label = f"{c['source']}, p.{c['page']}" if c["page"] else c["source"]
        parts.append(f"[{label}]\n{c['text']}")
    return "\n\n".join(parts)


def format_history(history):
    return "\n".join(
        f"{'Human' if t.role == 'user' else 'AI'}: {t.content}" for t in history[-2 * MEMORY_WINDOW:]
    )


@router.get("/health")
def health():
    return {"status": "ok", "chunks": len(CHUNKS)}


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        client = OpenAI(api_key=clean_key(os.environ["OPENAI_API_KEY"]))
        context = format_docs(retrieve(client, req.question))
        system = SYSTEM_PROMPT.format(history=format_history(req.history), context=context)
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": req.question}],
        )
    except Exception as e:
        logging.exception("chat failed")
        raise HTTPException(status_code=502, detail=f"The assistant could not answer right now ({type(e).__name__}: {type(e.__cause__).__name__ if e.__cause__ else ''} {str(e.__cause__ or e)[:120]}).")
    return ChatResponse(answer=result.choices[0].message.content)


# Vercel may pass either the original path or the rewritten one, so accept both
app.include_router(router, prefix="/api")
app.include_router(router, prefix="/api/index")

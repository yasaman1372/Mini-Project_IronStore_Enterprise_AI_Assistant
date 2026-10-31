import os
import uuid
from collections import OrderedDict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

# Importing rag_setup loads (or builds) the vector store, retriever, llm, prompt and format_docs
from rag_setup import retriever, llm, prompt, format_docs

MAX_SESSIONS = 200
MEMORY_WINDOW = 5

app = FastAPI(title="IronStore Assistant API")

# Comma-separated list of allowed frontend origins, e.g. https://my-app.vercel.app
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=os.getenv("ALLOWED_ORIGIN_REGEX") or None,
    allow_methods=["*"],
    allow_headers=["*"],
)

# One memory per chat session, oldest sessions are dropped when the cap is reached
sessions: "OrderedDict[str, ConversationBufferWindowMemory]" = OrderedDict()


def get_memory(session_id):
    if session_id in sessions:
        sessions.move_to_end(session_id)
    else:
        sessions[session_id] = ConversationBufferWindowMemory(
            k=MEMORY_WINDOW, memory_key="history", return_messages=False
        )
        while len(sessions) > MAX_SESSIONS:
            sessions.popitem(last=False)
    return sessions[session_id]


def build_chain(memory):
    return (
        {"context": retriever | format_docs,
         "question": RunnablePassthrough(),
         "history": lambda _: memory.load_memory_variables({})["history"]}
        | prompt
        | llm
        | StrOutputParser()
    )


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=100)


class ChatResponse(BaseModel):
    answer: str
    session_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    memory = get_memory(session_id)
    try:
        answer = build_chain(memory).invoke(req.question)
    except Exception:
        raise HTTPException(status_code=502, detail="The assistant could not answer right now.")
    memory.save_context({"input": req.question}, {"output": answer})
    return ChatResponse(answer=answer, session_id=session_id)


@app.delete("/chat/{session_id}")
def reset(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "cleared"}

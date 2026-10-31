# Mini-Project_IronStore_Enterprise_AI_Assistant

RAG assistant over IronStore documents: a FastAPI backend (`app/`) and a React chat UI (`frontend/`).

## Run locally

```bash
# backend (Python 3.12), from the project root; needs OPENAI_API_KEY in .env
source .venv/langchain-v0.2/bin/activate
pip install -r requirements.txt
cd app && uvicorn api:app --port 8765

# frontend, in a second terminal
cd frontend
cp .env.example .env
npm install && npm run dev
```

The first start builds `chroma_db` from `documents/` (one-off OpenAI embedding cost).

## Deploy

1. **Backend on Render:** New > Blueprint, pick this repo (it reads `render.yaml`). Set `OPENAI_API_KEY`, and set `ALLOWED_ORIGINS` to your Vercel URL (e.g. `https://my-app.vercel.app`).
2. **Frontend on Vercel:** import the repo, set **Root Directory** to `frontend`, and add the env var `VITE_API_URL` = your Render URL (no trailing slash). Redeploy after changing it.
3. Test: open the Vercel URL and ask a question. For Vercel preview URLs, set `ALLOWED_ORIGIN_REGEX` on Render, e.g. `https://.*\.vercel\.app`.
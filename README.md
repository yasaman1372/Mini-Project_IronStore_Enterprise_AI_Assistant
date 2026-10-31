# Mini-Project_IronStore_Enterprise_AI_Assistant

RAG assistant over IronStore documents. Everything deploys to one Vercel project:
a React chat UI (`frontend/`) and a small Python API (`api/index.py`) that searches a precomputed index (`api/data/`).

`app/` and `main.ipynb` hold the original LangChain + Chroma code, used to experiment and to build the index.

## Run locally

```bash
source .venv/langchain-v0.2/bin/activate
pip install -r requirements-dev.txt      # full local tooling (LangChain, Chroma)

# backend, from the project root
set -a; source .env; set +a              # needs OPENAI_API_KEY
uvicorn api.index:app --port 8765

# frontend, in a second terminal
cd frontend
cp .env.example .env                     # VITE_API_URL=http://localhost:8765
npm install && npm run dev
```

## Rebuild the index (when `documents/` change)

```bash
rm -rf chroma_db                         # forces re-embedding
python scripts/export_index.py           # rebuilds chroma_db, writes api/data/
```

Commit the updated `api/data/` files.

## Deploy (Vercel, one project)

1. Push the repo to GitHub (keep `api/data/`, but not `.env`).
2. Vercel > Add New > Project > import the repo. Leave **Root Directory** as the repo root. `vercel.json` sets the build.
3. Add the environment variable `OPENAI_API_KEY`, then deploy.
4. Do not set `VITE_API_URL` on Vercel: the UI calls `/api/chat` on the same domain.

Chat memory is kept in the browser (the last 5 exchanges are sent with each question), because serverless functions are stateless.

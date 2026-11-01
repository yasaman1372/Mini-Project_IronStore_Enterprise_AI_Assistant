# IronStore Enterprise AI Assistant

A RAG chat assistant that answers questions about IronStore's internal policies and product catalogs, and cites the source file and page for each answer.

**Live demo:** https://mini-project-iron-store-enterprise-five.vercel.app

## How it works

1. The PDFs are loaded, split into chunks (1000 characters, 150 overlap) and embedded with OpenAI (`text-embedding-ada-002`).
2. For each question the 8 most similar chunks are retrieved and sent to `gpt-4o-mini` with a prompt that allows answers only from that context and requires citations.
3. The last 5 question/answer pairs are sent along as chat history, so follow-ups like "And for bank transfers?" work.

## Project structure

| Path | Purpose |
|------|---------|
| `frontend/` | React + Vite chat UI |
| `api/index.py` | FastAPI backend deployed as a Vercel function. Searches `api/data/` with numpy and calls OpenAI |
| `api/data/` | Precomputed index: `chunks.json`, `embeddings.npy`, `meta.json` (must stay in git) |
| `scripts/export_index.py` | Builds `api/data/` from the local Chroma index |
| `app/rag_setup.py` | LangChain + Chroma pipeline (loading, chunking, embedding, chain, memory) used for development |
| `vercel.json` | Vercel build and routing config |
| `requirements.txt` | Deployed backend only (`fastapi`, `openai`, `numpy`) |
| `requirements-dev.txt` | Full local tooling (LangChain, Chroma, notebook) |

Not in git (see `.gitignore`): `.env`, `.venv/`, `chroma_db/`, `documents/` (the source PDFs), `main.ipynb` and `app/main.py` (local experiments).

## Run locally

Requires Python 3.12 or 3.13 and Node 18+. Create `.env` in the project root:

```
OPENAI_API_KEY=sk-...
```

```bash
python3.12 -m venv .venv/langchain-v0.2
source .venv/langchain-v0.2/bin/activate
pip install -r requirements-dev.txt

# backend, from the project root
set -a; source .env; set +a
uvicorn api.index:app --port 8765

# frontend, in a second terminal
cd frontend
cp .env.example .env                     # VITE_API_URL=http://localhost:8765
npm install && npm run dev               # http://localhost:5173
```

## Rebuild the index

Do this when the PDFs in `documents/` change. Put the PDFs in `documents/`, then:

```bash
rm -rf chroma_db                         # forces re-embedding (one-off OpenAI cost)
python scripts/export_index.py           # writes api/data/
```

Commit the updated `api/data/` files and push.

## Deploy to Vercel (one project: UI and API)

1. Push the repo to GitHub. Include `api/data/`, never `.env`.
2. In Vercel: **Add New > Project**, import the repo and leave **Root Directory** as the repo root. `vercel.json` sets the framework to none, builds `frontend/`, serves `frontend/dist`, and routes `/api/*` to `api/index.py`.
3. Under **Settings > Environment Variables** add `OPENAI_API_KEY`. The value is only the key (`sk-...`), without the name, quotes or spaces. Redeploy after any change to variables.
4. Do not set `VITE_API_URL`: the UI calls `/api/chat` on the same domain.

### Troubleshooting

- **`{"detail":"Not Found"}` on every URL:** Vercel treated the project as a FastAPI app. Keep `"framework": null` in `vercel.json`.
- **502 "could not answer", `AuthenticationError`:** the `OPENAI_API_KEY` value on Vercel is wrong. Fix it and redeploy.
- **Other 502 errors:** the message includes the error type. Full details are in **Deployments > Logs**.

## Notes and limitations

- Chat memory lives in the browser, because serverless functions are stateless. **New chat** clears it.
- The index contains the full text of the documents (`api/data/chunks.json`), so a public repo makes that text public. Use a private repo if the documents are confidential.
- Retrieval searches with the raw question, so very vague follow-ups can retrieve poorly.

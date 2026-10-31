"""Export the local Chroma index to plain files that the Vercel function loads.

Run from the project root with the local venv:  python scripts/export_index.py
Re-run it whenever the documents change (delete chroma_db/ first to re-embed).
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from rag_setup import vs  # noqa: E402  (loads or builds chroma_db)

OUT = os.path.join(os.path.dirname(__file__), "..", "api", "data")
os.makedirs(OUT, exist_ok=True)

data = vs._collection.get(include=["embeddings", "documents", "metadatas"])

seen, chunks, vectors = set(), [], []
for text, meta, emb in zip(data["documents"], data["metadatas"], data["embeddings"]):
    key = hashlib.md5(text.encode()).hexdigest()
    if key in seen:
        continue
    seen.add(key)
    page = meta.get("page")
    chunks.append({
        "text": text,
        "source": os.path.basename(meta.get("source", "unknown")),
        "page": page + 1 if isinstance(page, int) else None,
    })
    vectors.append(emb)

np.save(os.path.join(OUT, "embeddings.npy"), np.array(vectors, dtype=np.float16))
with open(os.path.join(OUT, "chunks.json"), "w") as f:
    json.dump(chunks, f, ensure_ascii=False)
with open(os.path.join(OUT, "meta.json"), "w") as f:
    json.dump({"embedding_model": vs._embedding_function.model}, f)

print(f"Exported {len(chunks)} chunks (from {len(data['documents'])}) to {os.path.abspath(OUT)}")

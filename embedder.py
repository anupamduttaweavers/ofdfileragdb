"""
embedder.py
────────────
Embedding and LLM generation via Ollama.

Supports remote Ollama instances — configure via environment variables:
  OLLAMA_BASE_URL    = http://192.168.0.207:8080
  OLLAMA_EMBED_MODEL = nomic-embed-text:latest
  OLLAMA_LLM_MODEL   = llama3.1:8b

Model: nomic-embed-text → 768-dimensional vectors.
"""

import json
import logging
import os
import time
import urllib.request
from typing import List

import numpy as np

log = logging.getLogger("embedder")

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://192.168.0.207:8080").rstrip("/")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
EMBED_DIM = int(os.getenv("EMBEDDING_DIMS", "768"))
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")
_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))


# ─────────────────────────────────────────────────────────────────────
# Low-level HTTP helpers (pure stdlib — no requests dependency)
# ─────────────────────────────────────────────────────────────────────

def _post(endpoint: str, payload: dict) -> dict:
    """Send a POST request to Ollama and return the parsed JSON response."""
    url = f"{OLLAMA_BASE}{endpoint}"
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode())


def _check_ollama():
    """Raise a clear error if Ollama is not reachable."""
    try:
        urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=5)
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_BASE}. "
            f"Ensure the Ollama server is running. Original error: {e}"
        ) from e


def is_ollama_available() -> bool:
    """Non-throwing connectivity check."""
    try:
        _check_ollama()
        return True
    except RuntimeError:
        return False


# ─────────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────────

def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Embed a list of strings via nomic-embed-text through Ollama.

    Returns float32 numpy array of shape (N, EMBED_DIM), L2-normalised
    so that dot-product == cosine similarity.
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    all_vecs: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        for text in chunk:
            retries = 3
            for attempt in range(retries):
                try:
                    resp = _post("/api/embeddings", {
                        "model": EMBED_MODEL,
                        "prompt": text,
                    })
                    all_vecs.append(resp["embedding"])
                    break
                except Exception as exc:
                    if attempt == retries - 1:
                        log.error("Embedding failed after %d attempts: %s", retries, exc)
                        all_vecs.append([0.0] * EMBED_DIM)
                    else:
                        time.sleep(0.5 * (attempt + 1))

    arr = np.array(all_vecs, dtype=np.float32)

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns shape (1, EMBED_DIM)."""
    return embed_texts([query])


# ─────────────────────────────────────────────────────────────────────
# LLM call (used by RAG service and schema_intelligence)
# ─────────────────────────────────────────────────────────────────────

def llm_generate(prompt: str, system: str = "", temperature: float = 0.1) -> str:
    """
    Call the configured LLM model via Ollama /api/generate.
    Returns the full response string.
    """
    payload: dict = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 4096},
    }
    if system:
        payload["system"] = system

    resp = _post("/api/generate", payload)
    return resp.get("response", "").strip()

"""
app.core.embedder
──────────────────
Embedding and LLM generation via Ollama.

All config is read from centralised Settings (app.config).
No direct os.getenv() calls.
"""

import json
import logging
import time
import urllib.request
from typing import List

import numpy as np

from app.config import get_settings

log = logging.getLogger("app.core.embedder")

_settings = get_settings()

OLLAMA_BASE: str = _settings.ollama_base_url.rstrip("/")
EMBED_MODEL: str = _settings.ollama_embed_model
EMBED_DIM: int = _settings.embedding_dims
LLM_MODEL: str = _settings.ollama_llm_model
_TIMEOUT: int = _settings.ollama_timeout


def _post(endpoint: str, payload: dict) -> dict:
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
    try:
        urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=5)
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_BASE}. "
            f"Ensure the Ollama server is running. Original error: {e}"
        ) from e


def is_ollama_available() -> bool:
    try:
        _check_ollama()
        return True
    except RuntimeError:
        return False


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
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
    return embed_texts([query])


def llm_generate(prompt: str, system: str = "", temperature: float = 0.1) -> str:
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

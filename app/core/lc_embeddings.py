"""
app.core.lc_embeddings
───────────────────────
LangChain OllamaEmbeddings wrapper with fallback to custom embedder.

Single Responsibility: provides a unified embedding interface.
Open/Closed: switch providers by changing config, not code.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config import get_settings

log = logging.getLogger("app.core.lc_embeddings")

_embeddings_instance = None


def get_lc_embeddings():
    """Return a cached LangChain OllamaEmbeddings instance."""
    global _embeddings_instance
    if _embeddings_instance is not None:
        return _embeddings_instance

    settings = get_settings()

    try:
        from langchain_ollama import OllamaEmbeddings

        _embeddings_instance = OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )
        log.info(
            "LangChain OllamaEmbeddings initialized: model=%s, base_url=%s",
            settings.ollama_embed_model, settings.ollama_base_url,
        )
        return _embeddings_instance
    except Exception as exc:
        log.warning("Failed to initialize LangChain OllamaEmbeddings: %s. Using fallback.", exc)
        return _FallbackEmbeddings()


class _FallbackEmbeddings:
    """Thin adapter over app.core.embedder when LangChain integration fails."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from app.core.embedder import embed_texts
        return embed_texts(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        from app.core.embedder import embed_query
        return embed_query(text).tolist()[0]

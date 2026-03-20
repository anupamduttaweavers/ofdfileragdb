"""
app.core.lc_llm
─────────────────
LangChain ChatOllama wrapper with graceful fallback.

Provides a unified LLM interface used by the LangGraph RAG pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config import get_settings

log = logging.getLogger("app.core.lc_llm")

_llm_instance = None


def get_lc_llm():
    """Return a cached ChatOllama instance."""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    settings = get_settings()

    try:
        from langchain_ollama import ChatOllama

        _llm_instance = ChatOllama(
            model=settings.ollama_llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.rag_temperature,
            num_predict=4096,
        )
        log.info(
            "LangChain ChatOllama initialized: model=%s, base_url=%s",
            settings.ollama_llm_model, settings.ollama_base_url,
        )
        return _llm_instance
    except Exception as exc:
        log.warning("Failed to initialize ChatOllama: %s", exc)
        return None


def invoke_llm(prompt: str, system: str = "", temperature: float = 0.1) -> str:
    """
    Invoke LLM with fallback. First tries LangChain ChatOllama,
    then falls back to raw Ollama HTTP.
    """
    llm = get_lc_llm()
    if llm is not None:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = []
            if system:
                messages.append(SystemMessage(content=system))
            messages.append(HumanMessage(content=prompt))

            response = llm.invoke(messages)
            return response.content.strip()
        except Exception as exc:
            log.warning("LangChain LLM invoke failed, falling back: %s", exc)

    from app.core.embedder import llm_generate
    return llm_generate(prompt, system=system, temperature=temperature)

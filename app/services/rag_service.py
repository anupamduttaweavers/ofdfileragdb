"""
app/services/rag_service.py
────────────────────────────
RAG (Retrieval-Augmented Generation) pipeline with optional reranking.

Architecture follows Interface Segregation + Single Responsibility:
  - retrieve()   → vector search via existing DocumentSearchEngine
  - rerank()     → score-based reranking using LLM relevance judgement
  - generate()   → context-stuffed LLM call via Ollama
  - ask()        → full pipeline: retrieve → (rerank) → generate

Fallback strategy:
  - If Ollama is down, ask() returns retrieved docs with a clear
    "LLM unavailable" message instead of raising 500.
  - If reranking fails, falls back to the original retrieval order.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from app.exceptions import EmbeddingError, LLMGenerationError, OllamaUnavailableError

log = logging.getLogger("app.services.rag")

_RAG_SYSTEM_PROMPT = (
    "You are an expert document analyst for the Ordnance Factory Board (OFB) system. "
    "Answer the user's question based ONLY on the provided context documents. "
    "If the information is not in the context, state that clearly. "
    "When citing information, reference the source document number in square brackets like [1], [2]. "
    "Be concise, accurate, and professional."
)

_RERANK_PROMPT_TEMPLATE = (
    "Rate the relevance of the following document to the query on a scale of 0 to 10.\n"
    "Query: {query}\n\n"
    "Document:\n{document}\n\n"
    "Respond with ONLY a single integer from 0 to 10."
)


@dataclass
class RagSource:
    """One source document used in RAG context."""
    rank: int
    doc_id: str
    score: float
    label: str
    source_db: str
    source_table: str
    snippet: str
    metadata: dict
    rerank_score: Optional[float] = None


@dataclass
class RagResult:
    """Full RAG pipeline output."""
    query: str
    answer: str
    sources: List[RagSource]
    model_used: str
    reranked: bool
    elapsed_ms: float


class RagService:
    """
    Stateless RAG orchestrator.
    Injected with a search engine instance; calls embedder for LLM generation.
    """

    def __init__(self, search_engine, *, default_top_k: int = 5, rerank_enabled: bool = True):
        self._engine = search_engine
        self._default_top_k = default_top_k
        self._rerank_enabled = rerank_enabled

    def ask(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        db_filter: Optional[str] = None,
        temperature: float = 0.1,
        rerank: Optional[bool] = None,
    ) -> RagResult:
        """
        Full RAG pipeline: retrieve → (optional rerank) → generate answer.

        Falls back gracefully if Ollama is unavailable.
        """
        start = time.perf_counter()
        k = top_k or self._default_top_k
        should_rerank = rerank if rerank is not None else self._rerank_enabled

        # Step 1: Retrieve
        sources = self._retrieve(query, top_k=k, db_filter=db_filter)

        if not sources:
            elapsed = (time.perf_counter() - start) * 1000
            return RagResult(
                query=query,
                answer="No relevant documents found for your query.",
                sources=[],
                model_used="none",
                reranked=False,
                elapsed_ms=round(elapsed, 2),
            )

        # Step 2: Rerank (optional, with fallback)
        reranked = False
        if should_rerank and len(sources) > 1:
            try:
                sources = self._rerank(query, sources)
                reranked = True
            except Exception as exc:
                log.warning("Reranking failed, using original order: %s", exc)

        # Step 3: Generate answer
        try:
            answer = self._generate(query, sources, temperature=temperature)
            model_used = self._get_llm_model()
        except (OllamaUnavailableError, LLMGenerationError) as exc:
            log.warning("LLM generation failed, returning sources only: %s", exc)
            answer = (
                "The LLM service is currently unavailable. "
                "Below are the most relevant documents found for your query. "
                "Please review them directly."
            )
            model_used = "unavailable"

        elapsed = (time.perf_counter() - start) * 1000
        return RagResult(
            query=query,
            answer=answer,
            sources=sources,
            model_used=model_used,
            reranked=reranked,
            elapsed_ms=round(elapsed, 2),
        )

    def _retrieve(self, query: str, *, top_k: int, db_filter: Optional[str]) -> List[RagSource]:
        """Vector search via DocumentSearchEngine."""
        try:
            results = self._engine.search(query, top_k=top_k, db_filter=db_filter)
        except Exception as exc:
            log.error("Search failed during RAG retrieve: %s", exc)
            raise EmbeddingError(f"Search failed: {exc}") from exc

        return [
            RagSource(
                rank=r.rank,
                doc_id=r.doc_id,
                score=r.score,
                label=r.label,
                source_db=r.source_db,
                source_table=r.source_table,
                snippet=r.snippet,
                metadata=r.metadata,
            )
            for r in results
        ]

    def _rerank(self, query: str, sources: List[RagSource]) -> List[RagSource]:
        """
        LLM-based reranking: ask the model to score each document's relevance.
        Falls back to original order on any failure.
        """
        from app.core.embedder import llm_generate

        scored = []
        for src in sources:
            prompt = _RERANK_PROMPT_TEMPLATE.format(query=query, document=src.snippet)
            try:
                raw = llm_generate(prompt, temperature=0.0)
                score = self._parse_rerank_score(raw)
                src.rerank_score = score
                scored.append(src)
            except Exception as exc:
                log.debug("Rerank scoring failed for doc %s: %s", src.doc_id, exc)
                src.rerank_score = src.score * 10
                scored.append(src)

        scored.sort(key=lambda s: s.rerank_score or 0, reverse=True)

        for i, src in enumerate(scored):
            src.rank = i + 1

        return scored

    @staticmethod
    def _parse_rerank_score(raw: str) -> float:
        """
        Extract a numeric score from LLM response.
        Handles responses like "7", "Score: 7", "7/10", etc.
        No regex -- simple string parsing.
        """
        cleaned = raw.strip()

        for char in cleaned:
            if char.isdigit():
                start = cleaned.index(char)
                end = start
                while end < len(cleaned) and (cleaned[end].isdigit() or cleaned[end] == "."):
                    end += 1
                try:
                    return float(cleaned[start:end])
                except ValueError:
                    break

        return 5.0

    def _generate(self, query: str, sources: List[RagSource], *, temperature: float) -> str:
        """Build context prompt and call LLM."""
        from app.core.embedder import llm_generate, _check_ollama

        try:
            _check_ollama()
        except RuntimeError as exc:
            raise OllamaUnavailableError() from exc

        context_parts = []
        for src in sources:
            header = f"[{src.rank}] {src.label} ({src.source_db}.{src.source_table}) — similarity: {src.score:.4f}"
            context_parts.append(f"{header}\n{src.snippet}")

        context_block = "\n\n---\n\n".join(context_parts)

        prompt = (
            f"Context Documents:\n\n{context_block}\n\n"
            f"─────────────────────────────────────\n"
            f"User Question: {query}\n\n"
            f"Provide a clear, well-structured answer based on the context above."
        )

        try:
            answer = llm_generate(prompt, system=_RAG_SYSTEM_PROMPT, temperature=temperature)
        except Exception as exc:
            raise LLMGenerationError(f"LLM generation failed: {exc}") from exc

        if not answer:
            raise LLMGenerationError("LLM returned an empty response.")

        return answer

    @staticmethod
    def _get_llm_model() -> str:
        from app.core.embedder import LLM_MODEL
        return LLM_MODEL

"""
app.core.lc_reranker
─────────────────────
FlashRank cross-encoder reranker.

Ultra-lightweight local reranking (~4ms per query).
Replaces the N-serial-LLM-calls approach that was slow and unreliable.

Fallback: if FlashRank is unavailable, falls back to LLM-based scoring.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document

log = logging.getLogger("app.core.lc_reranker")

_ranker = None


def _get_ranker():
    """Lazy-init the FlashRank model (downloads on first use, ~100MB)."""
    global _ranker
    if _ranker is not None:
        return _ranker

    try:
        from flashrank import Ranker

        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./flashrank_cache")
        log.info("FlashRank reranker initialized (ms-marco-MiniLM-L-12-v2).")
        return _ranker
    except Exception as exc:
        log.warning("FlashRank initialization failed: %s", exc)
        return None


def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: Optional[int] = None,
) -> List[Document]:
    """
    Rerank documents using FlashRank cross-encoder.

    Returns documents sorted by relevance score (highest first).
    Falls back to original order if FlashRank is unavailable.
    """
    if not documents:
        return documents

    ranker = _get_ranker()
    if ranker is None:
        log.debug("FlashRank unavailable, returning documents in original order.")
        return documents

    try:
        from flashrank import RerankRequest

        passages = [
            {"id": str(i), "text": doc.page_content[:2000], "meta": doc.metadata}
            for i, doc in enumerate(documents)
        ]

        rerank_req = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerank_req)

        reranked = []
        for result in results:
            idx = int(result["id"])
            doc = documents[idx]
            doc.metadata["rerank_score"] = round(float(result["score"]), 4)
            reranked.append(doc)

        if top_n is not None:
            reranked = reranked[:top_n]

        return reranked

    except Exception as exc:
        log.warning("FlashRank reranking failed: %s. Using original order.", exc)
        return documents


def rerank_with_scores(
    query: str,
    documents: List[Document],
    top_n: Optional[int] = None,
) -> List[Tuple[Document, float]]:
    """Return (Document, score) tuples after reranking."""
    reranked = rerank_documents(query, documents, top_n=top_n)
    return [
        (doc, doc.metadata.get("rerank_score", 0.0))
        for doc in reranked
    ]

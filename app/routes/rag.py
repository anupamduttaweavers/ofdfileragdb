"""
app/routes/rag.py
──────────────────
RAG (Retrieval-Augmented Generation) endpoint.
Uses LangGraph-based pipeline with grading, reranking, and hallucination checking.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.dependencies import require_api_key, get_search_engine
from app.models.requests import RagAskRequest
from app.models.responses import RagResponse, SearchResultItem
from app.core.lc_vector_store import FaissRetrieverAdapter
from app.services.rag_graph import RAGPipeline

log = logging.getLogger("app.routes.rag")
router = APIRouter(prefix="/api/v1", tags=["RAG"], dependencies=[Depends(require_api_key)])


@router.post(
    "/rag/ask",
    response_model=RagResponse,
    summary="Ask a question — self-corrective RAG with grading, reranking, and hallucination checking",
)
async def rag_ask(body: RagAskRequest, settings: Settings = Depends(get_settings)):
    engine = get_search_engine()

    retriever = FaissRetrieverAdapter(engine.store, top_k=settings.rag_top_k)

    pipeline = RAGPipeline(
        retriever,
        default_top_k=settings.rag_top_k,
        rerank_enabled=settings.rag_rerank_enabled,
    )

    result = pipeline.ask(
        query=body.query,
        top_k=body.top_k,
        db_filter=body.db_filter,
        temperature=body.temperature,
        rerank=body.rerank,
    )

    sources = [
        SearchResultItem(
            rank=s.rank,
            doc_id=s.doc_id,
            score=s.score,
            label=s.label,
            source_db=s.source_db,
            source_table=s.source_table,
            snippet=s.snippet,
            metadata=s.metadata,
            file_download_url=s.file_download_url,
        )
        for s in result.sources
    ]

    return RagResponse(
        query=result.query,
        answer=result.answer,
        sources=sources,
        model_used=result.model_used,
        reranked=result.reranked,
        elapsed_ms=result.elapsed_ms,
    )

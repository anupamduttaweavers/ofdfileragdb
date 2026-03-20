"""
app/routes/search.py
─────────────────────
Semantic vector search endpoint.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends

from app.dependencies import require_api_key, get_search_engine
from app.exceptions import EmbeddingError
from app.models.requests import SearchRequest
from app.models.responses import SearchResponse, SearchResultItem

log = logging.getLogger("app.routes.search")
router = APIRouter(prefix="/api/v1", tags=["Search"], dependencies=[Depends(require_api_key)])


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic vector search across indexed documents",
)
async def search_documents(body: SearchRequest):
    engine = get_search_engine()
    start = time.perf_counter()

    try:
        results = engine.search(
            query=body.query,
            top_k=body.top_k,
            db_filter=body.db_filter,
            table_filter=body.table_filter,
        )
    except Exception as exc:
        log.error("Search failed: %s", exc)
        raise EmbeddingError(f"Search operation failed: {exc}") from exc

    elapsed = (time.perf_counter() - start) * 1000

    store = engine.store

    items = []
    for r in results:
        file_url = None
        if r.metadata.get("file_path"):
            file_url = f"/api/v1/files/download/{r.doc_id}"
        else:
            fp = store.find_file_path_for_doc(r.doc_id)
            if fp:
                file_url = f"/api/v1/files/download/{r.doc_id}"

        items.append(SearchResultItem(
            rank=r.rank,
            doc_id=r.doc_id,
            score=r.score,
            label=r.label,
            source_db=r.source_db,
            source_table=r.source_table,
            snippet=r.snippet,
            metadata=r.metadata,
            file_download_url=file_url,
        ))

    return SearchResponse(
        query=body.query,
        results=items,
        total=len(items),
        elapsed_ms=round(elapsed, 2),
    )

"""
app/routes/health.py
─────────────────────
Health check endpoint -- no authentication required.
Reports overall system readiness: Ollama, FAISS index, DB connectivity.

Dynamically checks all registered databases from ConnectionStore.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.core.connection_store import get_connection_store
from app.core.embedder import is_ollama_available
from app.models.responses import HealthResponse

log = logging.getLogger("app.routes.health")
router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
)
async def health_check(settings: Settings = Depends(get_settings)):
    from app.dependencies import get_search_engine

    ollama_ok = is_ollama_available()

    index_loaded = False
    doc_count = 0
    try:
        engine = get_search_engine()
        doc_count = engine.index_count()
        index_loaded = True
    except Exception:
        pass

    store = get_connection_store()
    all_conns = store.load_all()
    db_status = {}
    for name, cred in all_conns.items():
        ok, _ = store.test_connection(cred)
        db_status[name] = ok

    if ollama_ok and index_loaded and all(db_status.values()):
        status = "healthy"
    elif index_loaded:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        ollama_reachable=ollama_ok,
        index_loaded=index_loaded,
        doc_count=doc_count,
        database_connections=db_status,
    )

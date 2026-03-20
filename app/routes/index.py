"""
app/routes/index.py
────────────────────
Indexing and sync endpoints.
Triggers background vectorisation jobs.

Uses ConnectionStore for dynamic DB credential resolution
instead of fragile env-var guessing.
"""

from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.core.connection_store import get_connection_store
from app.dependencies import require_api_key, get_vectorizer_config, get_search_engine
from app.exceptions import (
    DatabaseConnectionError,
    IndexingInProgressError,
    ResourceNotFoundError,
)
from app.models.requests import IndexRequest, SyncRequest
from app.models.responses import IndexResponse, SyncResponse

log = logging.getLogger("app.routes.index")
router = APIRouter(prefix="/api/v1", tags=["Indexing"], dependencies=[Depends(require_api_key)])

_active_engine = None
_engine_lock = threading.Lock()


@router.post(
    "/index",
    response_model=IndexResponse,
    summary="Trigger full or filtered background re-indexing",
)
async def start_indexing(body: IndexRequest, settings: Settings = Depends(get_settings)):
    global _active_engine

    from app.core.vectorizer import VectorizationEngine, DBConnectionConfig
    from app.core.schema_config import get_all_configs

    vec_config = get_vectorizer_config()

    with _engine_lock:
        if _active_engine is not None and _active_engine.is_indexing():
            raise IndexingInProgressError()

        all_cfgs = get_all_configs()
        if body.db_filter:
            all_cfgs = [c for c in all_cfgs if c.db == body.db_filter]
            if not all_cfgs:
                raise ResourceNotFoundError("database config", body.db_filter)

        store = get_connection_store()
        for cfg in all_cfgs:
            if cfg.db not in vec_config.db_connections:
                cred = store.get(cfg.db)
                if cred is None:
                    log.warning("No connection found for '%s' in ConnectionStore. Skipping.", cfg.db)
                    continue
                vec_config.db_connections[cfg.db] = DBConnectionConfig(
                    host=cred.host,
                    port=cred.port,
                    user=cred.user,
                    password=cred.password,
                    database=cred.database,
                )

        try:
            engine = VectorizationEngine(vec_config)
        except RuntimeError as exc:
            raise DatabaseConnectionError("ollama", str(exc)) from exc

        def _on_indexing_complete():
            try:
                search_eng = get_search_engine()
                search_eng.reload_index()
            except Exception as exc:
                log.warning("Could not reload search engine after indexing: %s", exc)

        engine.start_background_indexing(all_cfgs, on_complete=_on_indexing_complete)
        _active_engine = engine

    db_label = body.db_filter or "all databases"
    return IndexResponse(
        status="started",
        tables_queued=len(all_cfgs),
        message=f"Background indexing started for {len(all_cfgs)} table(s) from {db_label}.",
    )


@router.post(
    "/sync",
    response_model=SyncResponse,
    summary="Incremental re-index a single table",
)
async def sync_table(body: SyncRequest):
    from app.core.vectorizer import VectorizationEngine

    vec_config = get_vectorizer_config()

    if body.db_name not in vec_config.db_connections:
        raise ResourceNotFoundError("database connection", body.db_name)

    try:
        engine = VectorizationEngine(vec_config)
    except RuntimeError as exc:
        raise DatabaseConnectionError(body.db_name, str(exc)) from exc

    try:
        count = engine.sync_table(body.db_name, body.table_name)
    except ValueError as exc:
        raise ResourceNotFoundError("table config", f"{body.db_name}.{body.table_name}") from exc
    except Exception as exc:
        raise DatabaseConnectionError(body.db_name, str(exc)) from exc

    try:
        search_eng = get_search_engine()
        search_eng.reload_index()
    except Exception:
        pass

    return SyncResponse(
        db_name=body.db_name,
        table_name=body.table_name,
        docs_synced=count,
        message=f"Synced {count} documents from {body.db_name}.{body.table_name}.",
    )

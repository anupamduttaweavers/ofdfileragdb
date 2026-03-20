"""
app/routes/discover.py
───────────────────────
Auto-discover a new database schema.

Default mode is heuristic (instant, no LLM). Optionally refines with LLM
in background when mode='auto' or mode='llm'.

After successful discovery, persists connection credentials to
ConnectionStore so the database is available on restart without
re-registration.
"""

from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, Depends

from app.core.connection_store import DBCredentials, get_connection_store
from app.dependencies import require_api_key
from app.exceptions import DatabaseConnectionError
from app.models.requests import DiscoverRequest
from app.models.responses import DiscoverResponse

log = logging.getLogger("app.routes.discover")
router = APIRouter(prefix="/api/v1", tags=["Discovery"], dependencies=[Depends(require_api_key)])


@router.post(
    "/discover",
    response_model=DiscoverResponse,
    summary="Auto-discover and configure a new database schema",
)
async def discover_database(body: DiscoverRequest):
    from app.core.schema_intelligence import (
        discover_and_configure, _persist_to_sqlite,
        _resolve_mode, run_llm_refinement_background,
    )

    effective_mode = _resolve_mode(body.mode)

    try:
        configs = discover_and_configure(
            host=body.host,
            port=body.port,
            user=body.user,
            password=body.password,
            database=body.database,
            force_rediscover=body.force,
            mode=effective_mode,
        )
    except Exception as exc:
        raise DatabaseConnectionError(body.database, f"Failed to discover schema: {exc}") from exc

    store = get_connection_store()
    if not store.exists(body.database):
        store.save(body.database, DBCredentials(
            host=body.host,
            port=body.port,
            user=body.user,
            password=body.password,
            database=body.database,
        ))
        log.info("Persisted connection credentials for '%s' after discovery.", body.database)

    if configs:
        _persist_to_sqlite(body.database, configs)

    if not configs:
        return DiscoverResponse(
            database=body.database,
            tables_discovered=0,
            tables=[],
            message="Discovery completed but no suitable tables were found.",
        )

    if effective_mode in ("llm", "auto"):
        thread = threading.Thread(
            target=run_llm_refinement_background,
            kwargs=dict(
                db_name=body.database,
                host=body.host, port=body.port,
                user=body.user, password=body.password,
                database=body.database,
            ),
            daemon=True,
            name=f"llm-refine-{body.database}",
        )
        thread.start()
        return DiscoverResponse(
            database=body.database,
            tables_discovered=len(configs),
            tables=[c.table for c in configs],
            message=(
                f"Discovered {len(configs)} tables (heuristic). "
                f"LLM refinement running in background."
            ),
        )

    return DiscoverResponse(
        database=body.database,
        tables_discovered=len(configs),
        tables=[c.table for c in configs],
        message=f"Discovered {len(configs)} tables (heuristic). Run POST /api/v1/index to vectorise them.",
    )

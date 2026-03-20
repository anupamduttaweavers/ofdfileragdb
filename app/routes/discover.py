"""
app/routes/discover.py
───────────────────────
Auto-discover a new database schema using Llama 3 8B.

After successful discovery, persists connection credentials to
ConnectionStore so the database is available on restart without
re-registration.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.core.connection_store import DBCredentials, get_connection_store
from app.dependencies import require_api_key
from app.exceptions import DatabaseConnectionError, OllamaUnavailableError, LLMGenerationError
from app.models.requests import DiscoverRequest
from app.models.responses import DiscoverResponse

log = logging.getLogger("app.routes.discover")
router = APIRouter(prefix="/api/v1", tags=["Discovery"], dependencies=[Depends(require_api_key)])


@router.post(
    "/discover",
    response_model=DiscoverResponse,
    summary="Auto-discover and configure a new database using LLM schema analysis",
)
async def discover_database(body: DiscoverRequest):
    from app.core.schema_intelligence import discover_and_configure

    try:
        configs = discover_and_configure(
            host=body.host,
            port=body.port,
            user=body.user,
            password=body.password,
            database=body.database,
            force_rediscover=body.force,
        )
    except RuntimeError as exc:
        error_msg = str(exc).lower()
        if "ollama" in error_msg or "cannot reach" in error_msg:
            raise OllamaUnavailableError(
                "Ollama is required for schema discovery but is not reachable."
            ) from exc
        raise LLMGenerationError(f"Discovery failed: {exc}") from exc
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

    if not configs:
        return DiscoverResponse(
            database=body.database,
            tables_discovered=0,
            tables=[],
            message="Discovery completed but no suitable tables were found.",
        )

    return DiscoverResponse(
        database=body.database,
        tables_discovered=len(configs),
        tables=[c.table for c in configs],
        message=f"Discovered {len(configs)} tables. Run POST /api/v1/index to vectorise them.",
    )

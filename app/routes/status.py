"""
app/routes/status.py
─────────────────────
System status: configured databases, index stats, active jobs.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.dependencies import require_api_key, get_search_engine
from app.models.responses import StatusResponse, DatabaseInfo

log = logging.getLogger("app.routes.status")
router = APIRouter(prefix="/api/v1", tags=["Status"], dependencies=[Depends(require_api_key)])


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Show configured databases and index statistics",
)
async def get_status():
    from app.core.schema_config import get_all_configs
    from app.core.schema_intelligence import list_known_databases

    all_cfgs = get_all_configs()
    db_map: dict[str, list[str]] = {}
    for c in all_cfgs:
        db_map.setdefault(c.db, []).append(c.table)

    auto_discovered = set(list_known_databases()) - {"ofbdb", "misofb"}

    databases = []
    for db_name, tables in db_map.items():
        source = "auto-discovered" if db_name in auto_discovered else "handcrafted"
        databases.append(DatabaseInfo(
            name=db_name,
            source=source,
            tables=tables,
            table_count=len(tables),
        ))

    index_count = 0
    indexing_active = False
    try:
        engine = get_search_engine()
        index_count = engine.index_count()
    except Exception:
        pass

    from app.routes.index import _active_engine
    if _active_engine is not None:
        indexing_active = _active_engine.is_indexing()

    return StatusResponse(
        databases=databases,
        index_count=index_count,
        indexing_active=indexing_active,
    )

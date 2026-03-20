"""
app/routes/databases.py
────────────────────────
CRUD endpoints for dynamic database connection management.

All registered databases survive server restarts (persisted to SQLite).
New databases trigger heuristic schema discovery instantly on registration.
LLM refinement runs in background when DISCOVERY_MODE is 'auto' or 'llm'.
Table configs are persisted to SQLite for selection management.

Fail-safe:
  - Connection test before persisting (reports failure but still registers)
  - Heuristic discovery is instant and never fails on timeout
  - LLM refinement failure does not affect heuristic baseline
  - Deleting a DB also removes its table configs from SQLite and schema config file
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends

from app.core.connection_store import ConnectionStore, DBCredentials, get_connection_store
from app.dependencies import require_admin_or_api_key
from app.exceptions import (
    DatabaseAlreadyExistsError,
    DatabaseConnectionError,
    ResourceNotFoundError,
)
from app.models.requests import DatabaseRegisterRequest, DatabaseUpdateRequest
from app.models.responses import (
    DatabaseDeleteResponse,
    DatabaseListResponse,
    DatabaseRegisterResponse,
    RegisteredDatabase,
)

log = logging.getLogger("app.routes.databases")
router = APIRouter(prefix="/api/v1", tags=["Database Management"], dependencies=[Depends(require_admin_or_api_key)])

CONFIGS_DIR = Path("./configs")


def _build_registered_db(name: str, cred: DBCredentials, store: ConnectionStore) -> RegisteredDatabase:
    ok, _ = store.test_connection(cred)

    from app.core.schema_intelligence import load_saved_config
    saved = load_saved_config(name)
    has_config = saved is not None
    table_count = len(saved) if saved else 0

    from app.core.config_db import get_table_configs_for_db, is_initialized
    if is_initialized():
        sqlite_tables = get_table_configs_for_db(name)
        if sqlite_tables:
            table_count = max(table_count, len(sqlite_tables))
            has_config = True

    from app.core.schema_config import OFBDB_CONFIGS, MISOFB_CONFIGS
    if name == "ofbdb":
        table_count = max(table_count, len(OFBDB_CONFIGS))
        has_config = True
    elif name == "misofb":
        table_count = max(table_count, len(MISOFB_CONFIGS))
        has_config = True

    return RegisteredDatabase(
        name=name,
        host=cred.host,
        port=cred.port,
        database=cred.database,
        connected=ok,
        tables_configured=table_count,
        has_schema_config=has_config,
    )


@router.get(
    "/databases",
    response_model=DatabaseListResponse,
    summary="List all registered database connections",
)
async def list_databases():
    store = get_connection_store()
    all_conns = store.load_all()

    databases: List[RegisteredDatabase] = []
    for name, cred in all_conns.items():
        databases.append(_build_registered_db(name, cred, store))

    return DatabaseListResponse(databases=databases, total=len(databases))


@router.post(
    "/databases",
    response_model=DatabaseRegisterResponse,
    summary="Register a new database connection (auto-discovers schema via heuristics)",
)
async def register_database(body: DatabaseRegisterRequest):
    store = get_connection_store()

    if store.exists(body.name):
        raise DatabaseAlreadyExistsError(body.name)

    cred = DBCredentials(
        host=body.host,
        port=body.port,
        user=body.user,
        password=body.password,
        database=body.database,
    )

    ok, err_msg = store.test_connection(cred)
    if not ok:
        store.save(body.name, cred)
        log.warning("Registered '%s' but connection test failed: %s", body.name, err_msg)
        return DatabaseRegisterResponse(
            name=body.name,
            status="connection_failed",
            message=f"Registered but connection test failed: {err_msg}. Fix credentials via PUT /api/v1/databases/{body.name}.",
        )

    store.save(body.name, cred)

    if not body.auto_discover:
        return DatabaseRegisterResponse(
            name=body.name,
            status="registered",
            message="Connection verified and saved. Run POST /api/v1/discover to generate schema config.",
        )

    from app.core.schema_intelligence import (
        discover_and_configure, _persist_to_sqlite, _resolve_mode,
        run_llm_refinement_background, save_config,
    )

    effective_mode = _resolve_mode(body.discovery_mode)

    try:
        configs = discover_and_configure(
            host=body.host, port=body.port,
            user=body.user, password=body.password,
            database=body.database,
            force_rediscover=True, mode=effective_mode,
        )

        if body.file_columns and configs:
            _apply_manual_file_columns(configs, body.file_columns)

        if configs:
            _persist_to_sqlite(body.name, configs)

    except Exception as exc:
        log.error("Heuristic discovery failed for '%s': %s", body.name, exc)
        return DatabaseRegisterResponse(
            name=body.name,
            status="registered",
            message=f"Connection saved but discovery failed: {exc}. Run POST /api/v1/discover manually.",
        )

    tables = [c.table for c in configs]

    if effective_mode in ("llm", "auto"):
        thread = threading.Thread(
            target=run_llm_refinement_background,
            kwargs=dict(
                db_name=body.name,
                host=body.host, port=body.port,
                user=body.user, password=body.password,
                database=body.database,
                file_columns=body.file_columns,
            ),
            daemon=True,
            name=f"llm-refine-{body.name}",
        )
        thread.start()
        return DatabaseRegisterResponse(
            name=body.name,
            status="registered_with_discovery",
            tables_discovered=len(configs),
            tables=tables,
            message=(
                f"Discovered {len(configs)} tables (heuristic). "
                f"LLM refinement running in background — "
                f"poll GET /api/v1/databases/{body.name}/discovery-status."
            ),
        )

    return DatabaseRegisterResponse(
        name=body.name,
        status="registered_with_discovery",
        tables_discovered=len(configs),
        tables=tables,
        message=f"Discovered {len(configs)} tables (heuristic). Run POST /api/v1/index?db_filter={body.name} to vectorize.",
    )


@router.put(
    "/databases/{name}",
    response_model=DatabaseRegisterResponse,
    summary="Update an existing database connection",
)
async def update_database(name: str, body: DatabaseUpdateRequest):
    store = get_connection_store()
    existing = store.get(name)
    if existing is None:
        raise ResourceNotFoundError("database", name)

    updated = DBCredentials(
        host=body.host or existing.host,
        port=body.port if body.port is not None else existing.port,
        user=body.user or existing.user,
        password=body.password if body.password is not None else existing.password,
        database=body.database or existing.database,
    )

    ok, err_msg = store.test_connection(updated)
    store.update(name, updated)

    if not ok:
        return DatabaseRegisterResponse(
            name=name,
            status="connection_failed",
            message=f"Credentials updated but connection test failed: {err_msg}.",
        )

    return DatabaseRegisterResponse(
        name=name,
        status="registered",
        message="Connection updated and verified.",
    )


@router.post(
    "/databases/{name}/discover",
    summary="Re-run schema discovery for an existing registered database",
)
async def discover_database_by_name(name: str):
    store = get_connection_store()
    cred = store.get(name)
    if cred is None:
        raise ResourceNotFoundError("database", name)

    from app.core.schema_intelligence import (
        get_discovery_status, discover_and_configure,
        _persist_to_sqlite, _resolve_mode, run_llm_refinement_background,
    )

    current = get_discovery_status(name)
    if current and current.get("status") == "running":
        return {
            "success": True,
            "database": name,
            "tables_discovered": 0,
            "tables": [],
            "message": f"Discovery already running. Poll GET /api/v1/databases/{name}/discovery-status.",
        }

    effective_mode = _resolve_mode()

    try:
        configs = discover_and_configure(
            host=cred.host, port=cred.port,
            user=cred.user, password=cred.password,
            database=cred.database,
            force_rediscover=True, mode=effective_mode,
        )
        if configs:
            _persist_to_sqlite(name, configs)
    except Exception as exc:
        log.error("Discovery failed for '%s': %s", name, exc)
        return {
            "success": False,
            "database": name,
            "tables_discovered": 0,
            "tables": [],
            "message": f"Discovery failed: {exc}",
        }

    tables = [c.table for c in configs]

    if effective_mode in ("llm", "auto"):
        thread = threading.Thread(
            target=run_llm_refinement_background,
            kwargs=dict(
                db_name=name,
                host=cred.host, port=cred.port,
                user=cred.user, password=cred.password,
                database=cred.database,
            ),
            daemon=True,
            name=f"llm-refine-{name}",
        )
        thread.start()

    msg_suffix = " LLM refinement running in background." if effective_mode in ("llm", "auto") else ""
    return {
        "success": True,
        "database": name,
        "tables_discovered": len(configs),
        "tables": tables,
        "message": f"Discovered {len(configs)} table(s) (heuristic).{msg_suffix}",
    }


@router.get(
    "/databases/{name}/discovery-status",
    summary="Poll schema discovery progress for a database",
)
async def discovery_status(name: str):
    store = get_connection_store()
    if not store.exists(name):
        raise ResourceNotFoundError("database", name)

    from app.core.schema_intelligence import get_discovery_status
    status = get_discovery_status(name)
    if status is None:
        return {"database": name, "discovery": "not_started"}
    return {"database": name, "discovery": status}


@router.delete(
    "/databases/{name}",
    response_model=DatabaseDeleteResponse,
    summary="Remove a registered database and its schema config",
)
async def delete_database(name: str):
    store = get_connection_store()

    if not store.exists(name):
        raise ResourceNotFoundError("database", name)

    if name in ("ofbdb", "misofb"):
        raise DatabaseConnectionError(
            name,
            f"Cannot delete built-in database '{name}'. It is a core system database.",
        )

    store.delete(name)

    from app.core.schema_intelligence import clear_discovery_status
    clear_discovery_status(name)

    from app.core.config_db import delete_table_configs_for_db, is_initialized
    if is_initialized():
        delete_table_configs_for_db(name)

    config_file = CONFIGS_DIR / f"{name}.json"
    if config_file.exists():
        try:
            config_file.unlink()
            log.info("Removed schema config file: %s", config_file)
        except OSError as exc:
            log.warning("Failed to remove config file %s: %s", config_file, exc)

    return DatabaseDeleteResponse(
        name=name,
        deleted=True,
        message=f"Database '{name}' and its configuration have been removed.",
    )


# ── Helpers ─────────────────────────────────────────────────────

def _apply_manual_file_columns(configs, file_columns_map):
    """Apply user-provided file_columns overrides to discovered configs."""
    from app.core.schema_intelligence import save_config

    for cfg in configs:
        if cfg.table in file_columns_map:
            raw = file_columns_map[cfg.table]
            cfg.file_columns = [
                (pair[0], pair[1] if len(pair) > 1 else "")
                for pair in raw
                if isinstance(pair, (list, tuple)) and len(pair) >= 1
            ]

    if configs:
        save_config(configs[0].db, configs)

"""
app.core.database
──────────────────
Centralized database connection management.

Single point for creating MySQL connections from Settings.
Avoids duplicating connection logic across routes and services.
"""

from __future__ import annotations

import logging
from typing import Optional

import mysql.connector

from app.config import Settings, get_settings
from app.exceptions import DatabaseConnectionError

log = logging.getLogger("app.core.database")


def get_db_connection(db_name: str, settings: Optional[Settings] = None):
    """
    Create a MySQL connection for the given database name.

    Supports: 'ofbdb' (aliased from 'mydb'), 'misofb'.
    Raises DatabaseConnectionError on failure.
    """
    if settings is None:
        settings = get_settings()

    conn_params = _resolve_connection_params(db_name, settings)

    try:
        conn = mysql.connector.connect(
            host=conn_params["host"],
            port=conn_params["port"],
            user=conn_params["user"],
            password=conn_params["password"],
            database=conn_params["database"],
            charset="utf8mb4",
            use_unicode=True,
        )
        return conn
    except Exception as exc:
        raise DatabaseConnectionError(db_name, f"Connection failed: {exc}") from exc


def _resolve_connection_params(db_name: str, settings: Settings) -> dict:
    """Map a logical DB name to connection parameters from Settings."""
    if db_name in ("mydb", "ofbdb"):
        return {
            "host": settings.ofbdb_host,
            "port": settings.ofbdb_port,
            "user": settings.ofbdb_user,
            "password": settings.ofbdb_password,
            "database": settings.ofbdb_database,
        }
    elif db_name == "misofb":
        return {
            "host": settings.misofb_host,
            "port": settings.misofb_port,
            "user": settings.misofb_user,
            "password": settings.misofb_password,
            "database": settings.misofb_database,
        }
    else:
        raise DatabaseConnectionError(db_name, f"Unknown database '{db_name}'")


def check_db_connection(db_name: str, settings: Optional[Settings] = None) -> bool:
    """Non-throwing connection check. Returns True if connectable."""
    try:
        conn = get_db_connection(db_name, settings)
        conn.close()
        return True
    except Exception:
        return False

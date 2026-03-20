"""
app.core.connection_store
──────────────────────────
Thread-safe persistent store for database connection credentials.

Backend: SQLite via config_db.py (replaces the former JSON file).
Public interface is unchanged so all existing consumers
(dependencies.py, sync_service.py, routes) work without modification.

Fallback: if SQLite is not yet initialised, returns empty data
gracefully so the startup sequence can proceed.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import mysql.connector

log = logging.getLogger("app.core.connection_store")


@dataclass
class DBCredentials:
    """Serializable database connection credentials."""
    host: str
    port: int
    user: str
    password: str
    database: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> DBCredentials:
        return cls(
            host=data["host"],
            port=int(data["port"]),
            user=data["user"],
            password=data["password"],
            database=data["database"],
        )


class ConnectionStore:
    """
    Persistent credential store backed by SQLite.

    All public methods are thread-safe. Each call reads directly from
    SQLite, so changes made by other processes are immediately visible.
    """

    def __init__(self):
        self._lock = threading.Lock()

    def _db_available(self) -> bool:
        from app.core.config_db import is_initialized
        return is_initialized()

    def seed_defaults(self, settings) -> None:
        """Seed ofbdb and misofb from Settings if not already present."""
        if not self._db_available():
            log.warning("SQLite not initialised yet; skipping seed_defaults.")
            return
        from app.core.config_db import seed_defaults
        seed_defaults(settings)

    def load_all(self) -> Dict[str, DBCredentials]:
        if not self._db_available():
            return {}
        from app.core.config_db import get_all_connections
        with self._lock:
            result = {}
            for name, data in get_all_connections().items():
                result[name] = DBCredentials(
                    host=data["host"],
                    port=data["port"],
                    user=data["user"],
                    password=data["password"],
                    database=data["database"],
                )
            return result

    def get(self, name: str) -> Optional[DBCredentials]:
        if not self._db_available():
            return None
        from app.core.config_db import get_connection
        with self._lock:
            data = get_connection(name)
            if data is None:
                return None
            return DBCredentials(
                host=data["host"],
                port=data["port"],
                user=data["user"],
                password=data["password"],
                database=data["database"],
            )

    def save(self, name: str, credentials: DBCredentials) -> None:
        if not self._db_available():
            log.warning("SQLite not initialised; cannot save connection '%s'.", name)
            return
        from app.core.config_db import save_connection
        with self._lock:
            save_connection(
                name=name,
                host=credentials.host,
                port=credentials.port,
                user=credentials.user,
                password=credentials.password,
                database=credentials.database,
            )
        log.info("Saved connection '%s' -> %s:%d/%s", name, credentials.host, credentials.port, credentials.database)

    def update(self, name: str, credentials: DBCredentials) -> bool:
        if not self._db_available():
            return False
        from app.core.config_db import connection_exists, save_connection
        with self._lock:
            if not connection_exists(name):
                return False
            save_connection(
                name=name,
                host=credentials.host,
                port=credentials.port,
                user=credentials.user,
                password=credentials.password,
                database=credentials.database,
            )
        log.info("Updated connection '%s'.", name)
        return True

    def delete(self, name: str) -> bool:
        if not self._db_available():
            return False
        from app.core.config_db import delete_connection
        with self._lock:
            result = delete_connection(name)
        if result:
            log.info("Deleted connection '%s'.", name)
        return result

    def exists(self, name: str) -> bool:
        if not self._db_available():
            return False
        from app.core.config_db import connection_exists
        with self._lock:
            return connection_exists(name)

    def names(self) -> list:
        return list(self.load_all().keys())

    @staticmethod
    def test_connection(credentials: DBCredentials, timeout: int = 5) -> tuple:
        """
        Test a database connection.
        Returns (success: bool, error_message: Optional[str]).
        """
        try:
            conn = mysql.connector.connect(
                host=credentials.host,
                port=credentials.port,
                user=credentials.user,
                password=credentials.password,
                database=credentials.database,
                charset="utf8mb4",
                connection_timeout=timeout,
            )
            conn.close()
            return True, None
        except mysql.connector.Error as exc:
            return False, str(exc)
        except Exception as exc:
            return False, f"Unexpected error: {exc}"


_store_instance: Optional[ConnectionStore] = None
_store_lock = threading.Lock()


def get_connection_store() -> ConnectionStore:
    """Module-level singleton accessor for ConnectionStore."""
    global _store_instance
    if _store_instance is None:
        with _store_lock:
            if _store_instance is None:
                _store_instance = ConnectionStore()
    return _store_instance

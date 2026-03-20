"""
app/services/sync_state.py
───────────────────────────
Sync state persistence layer.

Tracks per-table sync timestamps, record counts, and error states.
Persists to a JSON file on disk — no external infrastructure required.

Liskov Substitution: the SyncStateStore class can later be replaced with a
Redis-backed or DB-backed implementation sharing the same interface.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger("app.services.sync_state")

_DATETIME_FMT = "%Y-%m-%dT%H:%M:%S"


@dataclass
class TableSyncState:
    """Sync state for a single table."""
    table_key: str
    last_sync_at: Optional[str] = None
    records_synced: int = 0
    status: str = "pending"
    error_msg: Optional[str] = None
    duration_ms: float = 0.0

    @property
    def last_sync_datetime(self) -> Optional[datetime]:
        if self.last_sync_at is None:
            return None
        try:
            return datetime.strptime(self.last_sync_at, _DATETIME_FMT)
        except (ValueError, TypeError):
            return None


@dataclass
class SyncSnapshot:
    """Overall sync state snapshot."""
    last_incremental: Optional[str] = None
    last_full_rescan: Optional[str] = None
    tables: Dict[str, TableSyncState] = field(default_factory=dict)


class SyncStateStore:
    """
    File-backed sync state store.

    Thread-safe: all reads and writes are guarded by a lock.
    The JSON file is read once on construction, then kept in memory
    and flushed to disk after every update.
    """

    def __init__(self, state_file: str = "./sync_state.json"):
        self._path = Path(state_file)
        self._lock = threading.Lock()
        self._state = self._load()

    def _load(self) -> SyncSnapshot:
        if not self._path.exists():
            log.info("No sync state file found at %s — starting fresh.", self._path)
            return SyncSnapshot()

        try:
            with open(self._path, "r") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Corrupt sync state file, starting fresh: %s", exc)
            return SyncSnapshot()

        tables = {}
        for key, tbl_raw in raw.get("tables", {}).items():
            tables[key] = TableSyncState(
                table_key=key,
                last_sync_at=tbl_raw.get("last_sync_at"),
                records_synced=tbl_raw.get("records_synced", 0),
                status=tbl_raw.get("status", "pending"),
                error_msg=tbl_raw.get("error_msg"),
                duration_ms=tbl_raw.get("duration_ms", 0.0),
            )

        return SyncSnapshot(
            last_incremental=raw.get("last_incremental"),
            last_full_rescan=raw.get("last_full_rescan"),
            tables=tables,
        )

    def _flush(self) -> None:
        """Write current state to disk."""
        data = {
            "last_incremental": self._state.last_incremental,
            "last_full_rescan": self._state.last_full_rescan,
            "tables": {},
        }
        for key, tbl in self._state.tables.items():
            data["tables"][key] = {
                "last_sync_at": tbl.last_sync_at,
                "records_synced": tbl.records_synced,
                "status": tbl.status,
                "error_msg": tbl.error_msg,
                "duration_ms": tbl.duration_ms,
            }

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as exc:
            log.error("Failed to write sync state to %s: %s", self._path, exc)

    # ── Public API ──────────────────────────────────────────────

    def get_snapshot(self) -> SyncSnapshot:
        with self._lock:
            return self._state

    def get_table_state(self, table_key: str) -> Optional[TableSyncState]:
        with self._lock:
            return self._state.tables.get(table_key)

    def update_table(
        self,
        table_key: str,
        *,
        last_sync_at: str,
        records_synced: int,
        status: str,
        error_msg: Optional[str] = None,
        duration_ms: float = 0.0,
    ) -> None:
        with self._lock:
            self._state.tables[table_key] = TableSyncState(
                table_key=table_key,
                last_sync_at=last_sync_at,
                records_synced=records_synced,
                status=status,
                error_msg=error_msg,
                duration_ms=duration_ms,
            )
            self._flush()

    def mark_incremental_complete(self, timestamp: str) -> None:
        with self._lock:
            self._state.last_incremental = timestamp
            self._flush()

    def mark_full_rescan_complete(self, timestamp: str) -> None:
        with self._lock:
            self._state.last_full_rescan = timestamp
            self._flush()

    @staticmethod
    def now_iso() -> str:
        return datetime.now().strftime(_DATETIME_FMT)

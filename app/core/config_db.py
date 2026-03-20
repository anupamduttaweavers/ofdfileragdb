"""
app.core.config_db
────────────────────
Production-grade SQLite configuration database.

Responsibilities:
  - Schema creation and migration (init_db)
  - Seeding defaults from .env on first run
  - CRUD for admin_users, database_connections, table_configurations, system_settings
  - Fernet symmetric encryption for stored database passwords
  - Thread-safe via WAL mode + connection-per-call pattern

Fallback: if SQLite is unavailable, callers fall back to .env values.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cryptography.fernet import Fernet, InvalidToken

log = logging.getLogger("app.core.config_db")

_DB_PATH: Optional[str] = None
_FERNET: Optional[Fernet] = None

_DATETIME_FMT = "%Y-%m-%dT%H:%M:%S"


def _now() -> str:
    return datetime.now().strftime(_DATETIME_FMT)


# ── Encryption helpers ──────────────────────────────────────────

def _get_or_create_fernet(encryption_key: str) -> Fernet:
    """Get or create a Fernet instance. Generates a key if none provided."""
    global _FERNET
    if _FERNET is not None:
        return _FERNET

    if encryption_key:
        try:
            _FERNET = Fernet(encryption_key.encode())
            return _FERNET
        except Exception:
            log.warning("Invalid ENCRYPTION_KEY, generating a new one.")

    key = Fernet.generate_key()
    log.info("Generated new Fernet encryption key. Store it in .env as ENCRYPTION_KEY=%s", key.decode())
    _FERNET = Fernet(key)
    return _FERNET


def encrypt_password(plaintext: str) -> str:
    if _FERNET is None:
        return plaintext
    return _FERNET.encrypt(plaintext.encode()).decode()


def decrypt_password(ciphertext: str) -> str:
    if _FERNET is None:
        return ciphertext
    try:
        return _FERNET.decrypt(ciphertext.encode()).decode()
    except (InvalidToken, Exception):
        return ciphertext


# ── Connection management ───────────────────────────────────────

@contextmanager
def _get_conn():
    """Yield a SQLite connection with WAL mode and foreign keys enabled."""
    conn = sqlite3.connect(_DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema initialisation ──────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS admin_users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    UNIQUE NOT NULL,
    password_hash TEXT    NOT NULL,
    role          TEXT    NOT NULL DEFAULT 'admin',
    is_active     INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT,
    updated_at    TEXT
);

CREATE TABLE IF NOT EXISTS database_connections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    UNIQUE NOT NULL,
    host          TEXT    NOT NULL,
    port          INTEGER NOT NULL DEFAULT 3307,
    db_user       TEXT    NOT NULL,
    db_password   TEXT    NOT NULL,
    database_name TEXT    NOT NULL,
    is_active     INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT,
    updated_at    TEXT
);

CREATE TABLE IF NOT EXISTS table_configurations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    db_connection_name TEXT    NOT NULL,
    table_name         TEXT    NOT NULL,
    is_selected        INTEGER NOT NULL DEFAULT 1,
    text_columns       TEXT    DEFAULT '[]',
    metadata_columns   TEXT    DEFAULT '[]',
    pk_column          TEXT,
    label              TEXT,
    description        TEXT,
    date_column        TEXT,
    file_columns       TEXT    DEFAULT '[]',
    source             TEXT    NOT NULL DEFAULT 'auto',
    created_at         TEXT,
    updated_at         TEXT,
    UNIQUE(db_connection_name, table_name)
);

CREATE TABLE IF NOT EXISTS system_settings (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT
);
"""


def init_db(settings) -> None:
    """Create the SQLite database and tables. Call once at startup."""
    global _DB_PATH
    _DB_PATH = settings.sqlite_db_path

    db_dir = Path(_DB_PATH).parent
    db_dir.mkdir(parents=True, exist_ok=True)

    _get_or_create_fernet(settings.encryption_key)

    with _get_conn() as conn:
        conn.executescript(_SCHEMA_SQL)

    log.info("SQLite config database initialised at %s", _DB_PATH)


def seed_defaults(settings) -> None:
    """Seed super admin and default DB connections from .env if tables are empty."""
    _seed_super_admin(settings)
    _seed_default_connections(settings)


def _seed_super_admin(settings) -> None:
    import bcrypt
    with _get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM admin_users WHERE role='superadmin'").fetchone()
        if row["cnt"] > 0:
            return

        pw_hash = bcrypt.hashpw(settings.super_admin_password.encode(), bcrypt.gensalt()).decode()
        now = _now()
        conn.execute(
            "INSERT INTO admin_users (username, password_hash, role, is_active, created_at, updated_at) VALUES (?,?,?,1,?,?)",
            (settings.super_admin_username, pw_hash, "superadmin", now, now),
        )
        log.info("Seeded super admin user '%s' from .env.", settings.super_admin_username)


def _seed_default_connections(settings) -> None:
    defaults = {
        "ofbdb": {
            "host": settings.ofbdb_host,
            "port": settings.ofbdb_port,
            "user": settings.ofbdb_user,
            "password": settings.ofbdb_password,
            "database": settings.ofbdb_database,
        },
        "misofb": {
            "host": settings.misofb_host,
            "port": settings.misofb_port,
            "user": settings.misofb_user,
            "password": settings.misofb_password,
            "database": settings.misofb_database,
        },
    }

    with _get_conn() as conn:
        for name, cfg in defaults.items():
            existing = conn.execute("SELECT id FROM database_connections WHERE name=?", (name,)).fetchone()
            if existing:
                continue
            now = _now()
            conn.execute(
                "INSERT INTO database_connections (name, host, port, db_user, db_password, database_name, is_active, created_at, updated_at) VALUES (?,?,?,?,?,?,1,?,?)",
                (name, cfg["host"], cfg["port"], cfg["user"], encrypt_password(cfg["password"]), cfg["database"], now, now),
            )
            log.info("Seeded default connection '%s'.", name)


def migrate_from_json(json_path: str) -> int:
    """Migrate connections from legacy connections.json into SQLite. Returns count migrated."""
    p = Path(json_path)
    if not p.exists():
        return 0

    try:
        with open(p) as f:
            data = json.load(f)
    except Exception as exc:
        log.warning("Cannot read legacy JSON %s: %s", p, exc)
        return 0

    count = 0
    for name, cred in data.items():
        if get_connection(name) is not None:
            continue
        save_connection(
            name=name,
            host=cred.get("host", "127.0.0.1"),
            port=int(cred.get("port", 3307)),
            user=cred.get("user", "root"),
            password=cred.get("password", ""),
            database=cred.get("database", name),
        )
        count += 1

    if count > 0:
        log.info("Migrated %d connections from %s to SQLite.", count, p)
        backup = p.with_suffix(".json.migrated")
        try:
            p.rename(backup)
            log.info("Renamed legacy file to %s", backup)
        except OSError:
            pass

    return count


# ── Admin Users CRUD ───────────────────────────────────────────

def get_admin_user(username: str) -> Optional[Dict[str, Any]]:
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM admin_users WHERE username=? AND is_active=1", (username,)).fetchone()
        return dict(row) if row else None


def list_admin_users() -> List[Dict[str, Any]]:
    with _get_conn() as conn:
        rows = conn.execute("SELECT id, username, role, is_active, created_at, updated_at FROM admin_users ORDER BY id").fetchall()
        return [dict(r) for r in rows]


def create_admin_user(username: str, password: str, role: str = "admin") -> Dict[str, Any]:
    import bcrypt
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    now = _now()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO admin_users (username, password_hash, role, is_active, created_at, updated_at) VALUES (?,?,?,1,?,?)",
            (username, pw_hash, role, now, now),
        )
        row = conn.execute("SELECT * FROM admin_users WHERE username=?", (username,)).fetchone()
        return dict(row)


def deactivate_admin_user(user_id: int) -> bool:
    with _get_conn() as conn:
        cur = conn.execute("UPDATE admin_users SET is_active=0, updated_at=? WHERE id=?", (_now(), user_id))
        return cur.rowcount > 0


# ── Database Connections CRUD ──────────────────────────────────

def get_all_connections() -> Dict[str, Dict[str, Any]]:
    with _get_conn() as conn:
        rows = conn.execute("SELECT * FROM database_connections WHERE is_active=1 ORDER BY id").fetchall()
        result = {}
        for r in rows:
            result[r["name"]] = {
                "id": r["id"],
                "host": r["host"],
                "port": r["port"],
                "user": r["db_user"],
                "password": decrypt_password(r["db_password"]),
                "database": r["database_name"],
                "is_active": r["is_active"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
        return result


def get_connection(name: str) -> Optional[Dict[str, Any]]:
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM database_connections WHERE name=? AND is_active=1", (name,)).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "host": row["host"],
            "port": row["port"],
            "user": row["db_user"],
            "password": decrypt_password(row["db_password"]),
            "database": row["database_name"],
            "is_active": row["is_active"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


def save_connection(name: str, host: str, port: int, user: str, password: str, database: str) -> None:
    now = _now()
    enc_pw = encrypt_password(password)
    with _get_conn() as conn:
        existing = conn.execute("SELECT id FROM database_connections WHERE name=?", (name,)).fetchone()
        if existing:
            conn.execute(
                "UPDATE database_connections SET host=?, port=?, db_user=?, db_password=?, database_name=?, is_active=1, updated_at=? WHERE name=?",
                (host, port, user, enc_pw, database, now, name),
            )
        else:
            conn.execute(
                "INSERT INTO database_connections (name, host, port, db_user, db_password, database_name, is_active, created_at, updated_at) VALUES (?,?,?,?,?,?,1,?,?)",
                (name, host, port, user, enc_pw, database, now, now),
            )


def delete_connection(name: str) -> bool:
    with _get_conn() as conn:
        cur = conn.execute("UPDATE database_connections SET is_active=0, updated_at=? WHERE name=? AND is_active=1", (_now(), name))
        if cur.rowcount > 0:
            conn.execute("DELETE FROM table_configurations WHERE db_connection_name=?", (name,))
        return cur.rowcount > 0


def connection_exists(name: str) -> bool:
    with _get_conn() as conn:
        row = conn.execute("SELECT id FROM database_connections WHERE name=? AND is_active=1", (name,)).fetchone()
        return row is not None


# ── Table Configurations CRUD ──────────────────────────────────

def get_table_configs_for_db(db_name: str, selected_only: bool = False) -> List[Dict[str, Any]]:
    with _get_conn() as conn:
        sql = "SELECT * FROM table_configurations WHERE db_connection_name=?"
        params: list = [db_name]
        if selected_only:
            sql += " AND is_selected=1"
        sql += " ORDER BY table_name"
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_table_config(r) for r in rows]


def get_all_selected_table_configs() -> List[Dict[str, Any]]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT tc.* FROM table_configurations tc "
            "JOIN database_connections dc ON tc.db_connection_name = dc.name "
            "WHERE tc.is_selected=1 AND dc.is_active=1 "
            "ORDER BY tc.db_connection_name, tc.table_name"
        ).fetchall()
        return [_row_to_table_config(r) for r in rows]


def upsert_table_config(
    db_name: str,
    table_name: str,
    text_columns: List[str],
    metadata_columns: List[str],
    pk_column: Optional[str],
    label: str,
    description: str,
    date_column: Optional[str],
    file_columns: List[Tuple[str, str]],
    source: str = "auto",
    is_selected: int = 1,
) -> None:
    now = _now()
    tc_json = json.dumps(text_columns)
    mc_json = json.dumps(metadata_columns)
    fc_json = json.dumps([list(fc) for fc in file_columns])

    with _get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM table_configurations WHERE db_connection_name=? AND table_name=?",
            (db_name, table_name),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE table_configurations SET text_columns=?, metadata_columns=?, pk_column=?, "
                "label=?, description=?, date_column=?, file_columns=?, source=?, is_selected=?, updated_at=? "
                "WHERE db_connection_name=? AND table_name=?",
                (tc_json, mc_json, pk_column, label, description, date_column, fc_json, source, is_selected, now, db_name, table_name),
            )
        else:
            conn.execute(
                "INSERT INTO table_configurations "
                "(db_connection_name, table_name, is_selected, text_columns, metadata_columns, pk_column, "
                "label, description, date_column, file_columns, source, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (db_name, table_name, is_selected, tc_json, mc_json, pk_column, label, description, date_column, fc_json, source, now, now),
            )


def toggle_table_selection(db_name: str, table_name: str, selected: bool) -> bool:
    with _get_conn() as conn:
        cur = conn.execute(
            "UPDATE table_configurations SET is_selected=?, updated_at=? WHERE db_connection_name=? AND table_name=?",
            (1 if selected else 0, _now(), db_name, table_name),
        )
        return cur.rowcount > 0


def set_all_tables_selection(db_name: str, selected: bool) -> int:
    with _get_conn() as conn:
        cur = conn.execute(
            "UPDATE table_configurations SET is_selected=?, updated_at=? WHERE db_connection_name=?",
            (1 if selected else 0, _now(), db_name),
        )
        return cur.rowcount


def delete_table_configs_for_db(db_name: str) -> int:
    with _get_conn() as conn:
        cur = conn.execute("DELETE FROM table_configurations WHERE db_connection_name=?", (db_name,))
        return cur.rowcount


def _row_to_table_config(row) -> Dict[str, Any]:
    def _parse_json(val, default):
        if not val:
            return default
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return default

    return {
        "id": row["id"],
        "db_connection_name": row["db_connection_name"],
        "table_name": row["table_name"],
        "is_selected": bool(row["is_selected"]),
        "text_columns": _parse_json(row["text_columns"], []),
        "metadata_columns": _parse_json(row["metadata_columns"], []),
        "pk_column": row["pk_column"],
        "label": row["label"],
        "description": row["description"],
        "date_column": row["date_column"],
        "file_columns": _parse_json(row["file_columns"], []),
        "source": row["source"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


# ── System Settings ────────────────────────────────────────────

def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    with _get_conn() as conn:
        row = conn.execute("SELECT value FROM system_settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO system_settings (key, value, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=?, updated_at=?",
            (key, value, _now(), value, _now()),
        )


def is_initialized() -> bool:
    """Check if the SQLite database is ready."""
    return _DB_PATH is not None and Path(_DB_PATH).exists()

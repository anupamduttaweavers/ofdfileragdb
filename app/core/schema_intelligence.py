"""
app.core.schema_intelligence
─────────────────────────────
Three-tier schema discovery for new databases:

  Tier A — Heuristic rules (instant, deterministic, no LLM).
  Tier B — LLM refinement via Ollama (optional, merges over heuristic).
  Tier C — Background thread execution with in-memory status polling.

Controlled by DISCOVERY_MODE (.env):
  "heuristic" — Tier A only (default, instant).
  "llm"       — Tier A first, then Tier B in background.
  "auto"      — Same as "llm" (heuristic + LLM refinement).
"""

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mysql.connector

from app.core.schema_config import TableConfig

log = logging.getLogger("app.core.schema_intelligence")

CONFIGS_DIR = Path("./configs")
CONFIGS_DIR.mkdir(exist_ok=True)

# ── In-memory discovery status tracker ──────────────────────────

_discovery_lock = threading.Lock()
_discovery_status: Dict[str, Dict[str, Any]] = {}


def get_discovery_status(database: str) -> Optional[Dict[str, Any]]:
    with _discovery_lock:
        entry = _discovery_status.get(database)
        return dict(entry) if entry else None


def _set_discovery_status(database: str, **fields: Any) -> None:
    with _discovery_lock:
        entry = _discovery_status.setdefault(database, {})
        entry.update(fields)


def clear_discovery_status(database: str) -> None:
    with _discovery_lock:
        _discovery_status.pop(database, None)


# ── Tables / columns to always skip ────────────────────────────

_SKIP_TABLES = {
    "activity_log", "user_login_try", "user_password_history",
    "user_permission", "update_log", "clarification_reminder_log",
    "test", "website_setting",
}

_SKIP_TABLE_SUFFIXES = ("_log", "_session", "_sessions", "_migration", "_migrations", "_cache")

_SKIP_COLUMNS = {
    "user_pass", "password", "login_token", "email_verification_code",
    "password_reset_code", "aadhar_no", "aadhar",
    "session_token", "api_key", "secret",
}

# ── MySQL column-type sets for heuristic classification ─────────

_TEXT_TYPES = {"varchar", "text", "mediumtext", "longtext", "char", "tinytext", "json"}
_NUMERIC_TYPES = {"int", "bigint", "smallint", "tinyint", "mediumint", "decimal", "float", "double", "enum"}
_DATE_TYPES = {"date", "datetime", "timestamp"}

_DATE_NAME_PATTERNS = (
    "_date", "_at", "created_", "updated_", "modified_", "dt_",
    "date_of_", "dob", "expiry", "valid_",
)

_FILE_NAME_PATTERNS = (
    "_file", "_path", "file_", "attachment", "document",
    "upload", "certificate_file", "spec_file", "drawing_file",
)


# ── Schema introspection (shared by heuristic + LLM) ───────────

def _get_schema_info(host: str, port: int, user: str, password: str, database: str) -> Dict:
    conn = mysql.connector.connect(
        host=host, port=port, user=user, password=password,
        database="information_schema", charset="utf8mb4",
    )
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT TABLE_NAME, TABLE_COMMENT, TABLE_ROWS
        FROM TABLES
        WHERE TABLE_SCHEMA = %s AND TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_NAME
    """, (database,))
    tables = {row["TABLE_NAME"]: row for row in cur.fetchall()}

    cur.execute("""
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE,
               IS_NULLABLE, COLUMN_KEY, COLUMN_COMMENT
        FROM COLUMNS
        WHERE TABLE_SCHEMA = %s
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """, (database,))
    cols = cur.fetchall()
    cur.close()
    conn.close()

    schema: Dict[str, Any] = {}
    for col in cols:
        tbl = col["TABLE_NAME"]
        if _should_skip_table(tbl):
            continue
        col_name = col["COLUMN_NAME"]
        if col_name.lower() in _SKIP_COLUMNS:
            continue
        if tbl not in schema:
            schema[tbl] = {
                "comment": tables.get(tbl, {}).get("TABLE_COMMENT", ""),
                "est_rows": tables.get(tbl, {}).get("TABLE_ROWS", 0),
                "columns": [],
            }
        schema[tbl]["columns"].append({
            "name": col_name,
            "type": col["DATA_TYPE"],
            "nullable": col["IS_NULLABLE"],
            "key": col["COLUMN_KEY"],
        })

    return schema


def _should_skip_table(table_name: str) -> bool:
    if table_name in _SKIP_TABLES:
        return True
    lower = table_name.lower()
    return any(lower.endswith(suf) for suf in _SKIP_TABLE_SUFFIXES)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TIER A — Heuristic discovery (instant, deterministic)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def heuristic_discover(
    host: str, port: int, user: str, password: str, database: str,
    schema: Optional[Dict] = None,
) -> List[TableConfig]:
    """Classify columns using type + naming patterns. No LLM needed."""
    if schema is None:
        schema = _get_schema_info(host, port, user, password, database)

    if not schema:
        log.warning("Heuristic: no usable tables found in [%s].", database)
        return []

    configs: List[TableConfig] = []
    for tbl, info in schema.items():
        columns = info["columns"]
        col_names = {c["name"] for c in columns}

        pk_column = _find_pk(columns)
        text_columns = _find_text_columns(columns, pk_column)
        metadata_columns = _find_metadata_columns(columns, pk_column)
        date_column = _find_date_column(columns)
        file_columns = _find_file_columns(columns, col_names)

        if not text_columns and not file_columns:
            continue

        comment = info.get("comment", "")
        label = _table_to_label(tbl)
        description = comment if comment else f"Table {tbl} ({len(columns)} columns)"

        configs.append(TableConfig(
            db=database,
            table=tbl,
            label=label,
            description=description,
            pk_column=pk_column,
            text_columns=text_columns,
            metadata_columns=metadata_columns,
            date_column=date_column,
            file_columns=file_columns,
        ))

    log.info("Heuristic discovered %d tables for [%s].", len(configs), database)
    return configs


def _find_pk(columns: List[Dict]) -> Optional[str]:
    for c in columns:
        if c["key"] == "PRI":
            return c["name"]
    return None


def _find_text_columns(columns: List[Dict], pk_column: Optional[str]) -> List[str]:
    result = []
    for c in columns:
        if c["type"] not in _TEXT_TYPES:
            continue
        name = c["name"]
        if name == pk_column:
            continue
        if c["key"] in ("MUL", "UNI") and c["type"] not in ("text", "mediumtext", "longtext"):
            continue
        lower = name.lower()
        if any(lower.endswith(pat) or lower.startswith(pat.lstrip("_")) for pat in _FILE_NAME_PATTERNS):
            continue
        result.append(name)
    return result


def _find_metadata_columns(columns: List[Dict], pk_column: Optional[str]) -> List[str]:
    result = []
    if pk_column:
        result.append(pk_column)
    for c in columns:
        name = c["name"]
        if name == pk_column:
            continue
        if c["type"] in _NUMERIC_TYPES or c["key"] in ("MUL", "UNI"):
            result.append(name)
        elif c["type"] in _DATE_TYPES:
            result.append(name)
    return result


def _find_date_column(columns: List[Dict]) -> Optional[str]:
    for c in columns:
        if c["type"] not in _DATE_TYPES:
            continue
        lower = c["name"].lower()
        if any(pat in lower for pat in _DATE_NAME_PATTERNS):
            return c["name"]
    for c in columns:
        if c["type"] in _DATE_TYPES:
            return c["name"]
    return None


def _find_file_columns(columns: List[Dict], all_col_names: set) -> List[Tuple[str, str]]:
    result: List[Tuple[str, str]] = []
    for c in columns:
        lower = c["name"].lower()
        if not any(pat in lower for pat in _FILE_NAME_PATTERNS):
            continue
        if c["type"] not in _TEXT_TYPES:
            continue
        type_col = _guess_type_column(c["name"], all_col_names)
        result.append((c["name"], type_col))
    return result


def _guess_type_column(path_col: str, all_col_names: set) -> str:
    """Try to find an adjacent *_type column for a file path column."""
    base = path_col.lower()
    for suffix in ("_type", "_mime", "_ext", "_extension"):
        for candidate in all_col_names:
            if candidate.lower() == base.replace("_file", suffix).replace("_path", suffix):
                return candidate
    if "file_type" in {c.lower() for c in all_col_names}:
        for c in all_col_names:
            if c.lower() == "file_type":
                return c
    return ""


def _table_to_label(table_name: str) -> str:
    return table_name.replace("_", " ").title()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TIER B — LLM discovery + merge over heuristic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _schema_to_prompt(database: str, schema: Dict) -> str:
    lines = [
        f"Database: {database}",
        "Tables (columns listed as name:type):",
        "",
    ]
    for tbl, info in schema.items():
        col_str = ", ".join(
            f"{c['name']}:{c['type']}"
            + (" PK" if c["key"] == "PRI" else "")
            for c in info["columns"]
        )
        comment = f"  # {info['comment']}" if info["comment"] else ""
        lines.append(f"  {tbl}{comment}: {col_str}")
    return "\n".join(lines)


_SYSTEM_PROMPT = """
You are a database analyst assistant. Given a MySQL schema, you decide which
tables and columns should be vectorised for a semantic document search system.

Rules:
- Include tables that contain meaningful human-readable content.
- Exclude pure join/log/auth/session tables.
- text_columns: columns whose VALUES should be embedded.
- metadata_columns: columns to store as filterable metadata. Include the PK.
- pk_column: the primary key column name, or null.
- label: a short human-readable name for a row.
- description: one sentence explaining the table.
- date_column: the most relevant date column or null.
- file_columns: list of [path_column, type_column] pairs for columns that store
  file paths (PDF, DOCX, CSV, Excel, etc.). Identify columns named *_file,
  *_path, file_*, attachment*, document*, upload*, certificate_file, etc.
  type_column should be the column that stores the file MIME type or extension,
  or empty string "" if none exists. Set to empty array [] if no file columns.

Respond ONLY with a valid JSON array. No markdown, no explanation.
Each element: table, label, description, pk_column, text_columns, metadata_columns, date_column, file_columns
""".strip()


def llm_discover(
    host: str, port: int, user: str, password: str, database: str,
    schema: Optional[Dict] = None,
) -> List[TableConfig]:
    """Full LLM-based discovery. Requires Ollama to be reachable."""
    from app.core.embedder import llm_generate

    if schema is None:
        schema = _get_schema_info(host, port, user, password, database)

    if not schema:
        return []

    schema_str = _schema_to_prompt(database, schema)
    log.info("LLM discovery: schema built (%d tables). Calling LLM...", len(schema))

    prompt = (
        f"{schema_str}\n\n"
        f"Generate the JSON vectorisation config array for database '{database}'."
    )

    raw_response = llm_generate(prompt, system=_SYSTEM_PROMPT)
    return _parse_llm_response(raw_response, database)


def _parse_llm_response(raw: str, database: str) -> List[TableConfig]:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        log.error("LLM did not return a JSON array. Raw response:\n%s", raw[:500])
        return []

    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        log.error("JSON parse error: %s\nRaw: %s", e, match.group(0)[:500])
        return []

    configs: List[TableConfig] = []
    for item in items:
        try:
            raw_file_cols = item.get("file_columns", [])
            file_columns: List[Tuple[str, str]] = []
            for fc in raw_file_cols:
                if isinstance(fc, (list, tuple)) and len(fc) >= 1:
                    path_col = str(fc[0])
                    type_col = str(fc[1]) if len(fc) > 1 else ""
                    file_columns.append((path_col, type_col))

            cfg = TableConfig(
                db=database,
                table=item["table"],
                label=item.get("label", item["table"]),
                description=item.get("description", ""),
                pk_column=item.get("pk_column"),
                text_columns=item.get("text_columns", []),
                metadata_columns=item.get("metadata_columns", []),
                date_column=item.get("date_column"),
                file_columns=file_columns,
            )
            configs.append(cfg)
        except (KeyError, TypeError) as e:
            log.warning("Skipping malformed config item %s: %s", item, e)

    log.info("LLM generated %d table configs for [%s].", len(configs), database)
    return configs


def merge_llm_over_heuristic(
    heuristic: List[TableConfig], llm: List[TableConfig],
) -> List[TableConfig]:
    """Merge LLM results on top of heuristic baseline by table name.

    LLM configs override label, description, text_columns, metadata_columns,
    date_column, and file_columns — but only when the LLM actually produced
    non-empty values. Tables only in heuristic are kept as-is.
    """
    llm_map: Dict[str, TableConfig] = {c.table: c for c in llm}
    merged: List[TableConfig] = []

    seen = set()
    for h in heuristic:
        seen.add(h.table)
        l = llm_map.get(h.table)
        if l is None:
            merged.append(h)
            continue
        merged.append(TableConfig(
            db=h.db,
            table=h.table,
            label=l.label or h.label,
            description=l.description or h.description,
            pk_column=l.pk_column or h.pk_column,
            text_columns=l.text_columns if l.text_columns else h.text_columns,
            metadata_columns=l.metadata_columns if l.metadata_columns else h.metadata_columns,
            date_column=l.date_column if l.date_column else h.date_column,
            file_columns=l.file_columns if l.file_columns else h.file_columns,
        ))

    for l in llm:
        if l.table not in seen:
            merged.append(l)

    return merged


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TIER C — Background execution + persistence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _configs_to_json(configs: List[TableConfig]) -> List[dict]:
    return [
        {
            "db": c.db, "table": c.table, "label": c.label,
            "description": c.description, "pk_column": c.pk_column,
            "text_columns": c.text_columns,
            "metadata_columns": c.metadata_columns,
            "date_column": c.date_column,
            "file_columns": [list(fc) for fc in c.file_columns],
        }
        for c in configs
    ]


def _configs_from_json(data: List[dict]) -> List[TableConfig]:
    result = []
    for d in data:
        raw_fc = d.get("file_columns", [])
        file_columns: List[Tuple[str, str]] = []
        for fc in raw_fc:
            if isinstance(fc, (list, tuple)) and len(fc) >= 1:
                file_columns.append((str(fc[0]), str(fc[1]) if len(fc) > 1 else ""))
        result.append(TableConfig(
            db=d["db"], table=d["table"],
            label=d.get("label", d["table"]),
            description=d.get("description", ""),
            pk_column=d.get("pk_column"),
            text_columns=d.get("text_columns", []),
            metadata_columns=d.get("metadata_columns", []),
            date_column=d.get("date_column"),
            file_columns=file_columns,
        ))
    return result


def load_saved_config(database: str) -> Optional[List[TableConfig]]:
    path = CONFIGS_DIR / f"{database}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        configs = _configs_from_json(data)
        log.info("Loaded cached config for [%s]: %d tables.", database, len(configs))
        return configs
    except Exception as e:
        log.error("Failed to load config for [%s]: %s", database, e)
        return None


def save_config(database: str, configs: List[TableConfig]):
    path = CONFIGS_DIR / f"{database}.json"
    with open(path, "w") as f:
        json.dump(_configs_to_json(configs), f, indent=2)
    log.info("Saved config for [%s] -> %s", database, path)


def _persist_to_sqlite(db_name: str, configs: List[TableConfig], source: str = "auto") -> None:
    from app.core.config_db import upsert_table_config, is_initialized
    if not is_initialized():
        return
    for cfg in configs:
        upsert_table_config(
            db_name=db_name,
            table_name=cfg.table,
            text_columns=cfg.text_columns,
            metadata_columns=cfg.metadata_columns,
            pk_column=cfg.pk_column,
            label=cfg.label,
            description=cfg.description,
            date_column=cfg.date_column,
            file_columns=list(cfg.file_columns),
            source=source,
        )
    log.info("Persisted %d table configs to SQLite for '%s' (source=%s).", len(configs), db_name, source)


# ── Public discovery API ────────────────────────────────────────

def _resolve_mode(mode: Optional[str] = None) -> str:
    """Return effective discovery mode, falling back to settings default."""
    if mode and mode in ("heuristic", "llm", "auto"):
        return mode
    from app.config import get_settings
    return get_settings().discovery_mode


def discover_and_configure(
    host: str, port: int, user: str, password: str,
    database: str, force_rediscover: bool = False,
    mode: Optional[str] = None,
) -> List[TableConfig]:
    """Synchronous discovery — always runs heuristic first.

    Returns the heuristic configs immediately.  If mode is 'llm' or 'auto',
    the caller is responsible for launching LLM refinement in background.
    """
    if not force_rediscover:
        cached = load_saved_config(database)
        if cached is not None:
            return cached

    effective_mode = _resolve_mode(mode)

    _set_discovery_status(
        database, status="running", phase="schema_introspection",
        started_at=time.time(), error=None, tables_discovered=0,
        source="heuristic", mode=effective_mode,
    )

    log.info("Discovering schema for [%s] (mode=%s)", database, effective_mode)
    schema = _get_schema_info(host, port, user, password, database)

    if not schema:
        log.warning("No usable tables found in [%s].", database)
        _set_discovery_status(
            database, status="completed", phase="done",
            tables_discovered=0, finished_at=time.time(),
        )
        return []

    _set_discovery_status(database, phase="heuristic_analysis", table_count=len(schema))

    configs = heuristic_discover(host, port, user, password, database, schema=schema)

    if configs:
        save_config(database, configs)

    _set_discovery_status(
        database, status="completed", phase="done",
        tables_discovered=len(configs), finished_at=time.time(),
        source="heuristic",
    )

    log.info("Heuristic discovery complete for [%s]: %d tables.", database, len(configs))
    return configs


def run_llm_refinement_background(
    db_name: str, host: str, port: int, user: str, password: str,
    database: str, file_columns: Optional[Dict] = None,
) -> None:
    """Background thread: run LLM, merge over existing heuristic, persist."""
    try:
        _set_discovery_status(
            db_name, status="running", phase="llm_analysis",
            llm_started_at=time.time(), source="heuristic+llm",
        )

        schema = _get_schema_info(host, port, user, password, database)
        llm_configs = llm_discover(host, port, user, password, database, schema=schema)

        existing = load_saved_config(database) or []
        if llm_configs:
            merged = merge_llm_over_heuristic(existing, llm_configs)
        else:
            log.warning("LLM returned no configs for [%s]; keeping heuristic.", database)
            merged = existing

        if file_columns:
            for cfg in merged:
                if cfg.table in file_columns:
                    raw = file_columns[cfg.table]
                    cfg.file_columns = [
                        (pair[0], pair[1] if len(pair) > 1 else "")
                        for pair in raw
                        if isinstance(pair, (list, tuple)) and len(pair) >= 1
                    ]

        if merged:
            save_config(database, merged)
            _persist_to_sqlite(db_name, merged, source="auto")

        _set_discovery_status(
            db_name, status="completed", phase="done",
            tables_discovered=len(merged), finished_at=time.time(),
            source="heuristic+llm" if llm_configs else "heuristic",
        )
        log.info("LLM refinement complete for '%s': %d tables.", db_name, len(merged))

    except Exception as exc:
        log.error("LLM refinement failed for '%s': %s", db_name, exc)
        _set_discovery_status(
            db_name, status="completed", phase="done",
            error=f"LLM refinement failed: {exc}", finished_at=time.time(),
        )


def run_discovery_background(
    db_name: str, host: str, port: int, user: str, password: str,
    database: str, file_columns: Optional[Dict] = None,
    mode: Optional[str] = None,
) -> None:
    """Full background discovery: heuristic + optional LLM refinement."""
    effective_mode = _resolve_mode(mode)

    try:
        configs = discover_and_configure(
            host=host, port=port, user=user, password=password,
            database=database, force_rediscover=True, mode=effective_mode,
        )

        if file_columns and configs:
            for cfg in configs:
                if cfg.table in file_columns:
                    raw = file_columns[cfg.table]
                    cfg.file_columns = [
                        (pair[0], pair[1] if len(pair) > 1 else "")
                        for pair in raw
                        if isinstance(pair, (list, tuple)) and len(pair) >= 1
                    ]
            save_config(configs[0].db, configs)

        if configs:
            _persist_to_sqlite(db_name, configs)

        if effective_mode in ("llm", "auto"):
            run_llm_refinement_background(
                db_name=db_name, host=host, port=port,
                user=user, password=password, database=database,
                file_columns=file_columns,
            )

    except Exception as exc:
        log.error("Background discovery failed for '%s': %s", db_name, exc)
        _set_discovery_status(
            db_name, status="failed", phase="error",
            error=str(exc), finished_at=time.time(),
        )


# ── Utilities ───────────────────────────────────────────────────

_NON_SCHEMA_FILES = {"connections"}

def list_known_databases() -> List[str]:
    return [
        p.stem for p in CONFIGS_DIR.glob("*.json")
        if p.stem not in _NON_SCHEMA_FILES
    ]


def print_config_summary(configs: List[TableConfig]):
    for c in configs:
        print(f"  [{c.db}].{c.table}  ->  '{c.label}'")
        print(f"    text cols  : {c.text_columns}")
        print(f"    meta cols  : {c.metadata_columns}")
        print(f"    pk         : {c.pk_column}  |  date: {c.date_column}")
        if c.file_columns:
            print(f"    file cols  : {c.file_columns}")
        print()

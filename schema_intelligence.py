"""
schema_intelligence.py
───────────────────────
Uses Llama 3 8B (via Ollama, fully offline) to intelligently analyse any
MySQL database schema and auto-generate a vectorisation config.

Workflow for a NEW database:
  1. Connect to DB → introspect INFORMATION_SCHEMA
  2. Build a compact schema description (table + column names + types)
  3. Send to llama3:8b with a structured prompt
  4. Parse JSON response → List[TableConfig]
  5. Save to  configs/<db_name>.json  (cached, never re-asks LLM)

On next startup the saved JSON is loaded automatically.

Manual override: edit  configs/<db_name>.json  to adjust any config.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import mysql.connector

from embedder import llm_generate
from schema_config import TableConfig

log = logging.getLogger("schema_intelligence")

CONFIGS_DIR = Path("./configs")
CONFIGS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# DB introspection
# ─────────────────────────────────────────────────────────────────────

_SKIP_TABLES = {
    # Tables that are never useful to vectorise
    "activity_log", "user_login_try", "user_password_history",
    "user_permission", "update_log", "clarification_reminder_log",
    "test", "website_setting",
}

_SKIP_COLUMNS = {
    # Columns that must NEVER enter the vector store
    "user_pass", "password", "login_token", "email_verification_code",
    "password_reset_code", "aadhar_no", "aadhar",
    "session_token", "api_key", "secret",
}


def _get_schema_info(host: str, port: int, user: str, password: str, database: str) -> Dict:
    """
    Returns { table_name: [ {column, type, nullable, key} ] } for every
    user-created table in the database.
    """
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
        if tbl in _SKIP_TABLES:
            continue
        col_name = col["COLUMN_NAME"]
        if col_name.lower() in _SKIP_COLUMNS:
            continue
        if tbl not in schema:
            schema[tbl] = {
                "comment":  tables.get(tbl, {}).get("TABLE_COMMENT", ""),
                "est_rows": tables.get(tbl, {}).get("TABLE_ROWS", 0),
                "columns":  [],
            }
        schema[tbl]["columns"].append({
            "name":     col_name,
            "type":     col["DATA_TYPE"],
            "nullable": col["IS_NULLABLE"],
            "key":      col["COLUMN_KEY"],
        })

    return schema


def _schema_to_prompt(database: str, schema: Dict) -> str:
    """
    Build a compact schema description that fits in a prompt.
    Each table gets one line per column to keep token count manageable.
    """
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


# ─────────────────────────────────────────────────────────────────────
# LLM prompt + parser
# ─────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are a database analyst assistant. Given a MySQL schema, you decide which
tables and columns should be vectorised for a semantic document search system.

Rules:
- Include tables that contain meaningful human-readable content (names, descriptions,
  specifications, status, reports, remarks, certificates, etc.)
- Exclude pure join/log/auth/session tables.
- text_columns: columns whose VALUES should be embedded (free text, names, descriptions).
  These should be VARCHAR, TEXT, ENUM, or any column with human-readable content.
- metadata_columns: columns to store as filterable metadata (IDs, dates, status codes,
  numeric values, flags). Include the PK here.
- pk_column: the primary key column name, or null if no single PK.
- label: a short human-readable name for what a row in this table represents.
- description: one sentence explaining what this table contains.
- date_column: the most relevant date column (created_at, registration_date, etc.) or null.

Respond ONLY with a valid JSON array. No markdown, no explanation, no preamble.
Each element must have exactly these keys:
  table, label, description, pk_column, text_columns, metadata_columns, date_column

Example element:
{
  "table": "employee",
  "label": "Employee Record",
  "description": "Personnel records with name, grade and pay details.",
  "pk_column": "emp_id",
  "text_columns": ["name", "designation", "department", "email"],
  "metadata_columns": ["emp_id", "grade", "status", "join_date"],
  "date_column": "join_date"
}
""".strip()


def _parse_llm_response(raw: str, database: str) -> List[TableConfig]:
    """
    Extract JSON from LLM response (handles ```json fences or raw JSON).
    Returns list of TableConfig.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    # Find the first [ ... ] block
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        log.error("LLM did not return a JSON array. Raw response:\n" + raw[:500])
        return []

    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        log.error(f"JSON parse error: {e}\nRaw: {match.group(0)[:500]}")
        return []

    configs: List[TableConfig] = []
    for item in items:
        try:
            cfg = TableConfig(
                db=database,
                table=item["table"],
                label=item.get("label", item["table"]),
                description=item.get("description", ""),
                pk_column=item.get("pk_column"),
                text_columns=item.get("text_columns", []),
                metadata_columns=item.get("metadata_columns", []),
                date_column=item.get("date_column"),
            )
            configs.append(cfg)
        except (KeyError, TypeError) as e:
            log.warning(f"Skipping malformed config item {item}: {e}")

    log.info(f"LLM generated {len(configs)} table configs for [{database}].")
    return configs


# ─────────────────────────────────────────────────────────────────────
# Serialise / deserialise TableConfig as JSON
# ─────────────────────────────────────────────────────────────────────

def _configs_to_json(configs: List[TableConfig]) -> List[dict]:
    return [
        {
            "db":               c.db,
            "table":            c.table,
            "label":            c.label,
            "description":      c.description,
            "pk_column":        c.pk_column,
            "text_columns":     c.text_columns,
            "metadata_columns": c.metadata_columns,
            "date_column":      c.date_column,
        }
        for c in configs
    ]


def _configs_from_json(data: List[dict]) -> List[TableConfig]:
    return [
        TableConfig(
            db=d["db"],
            table=d["table"],
            label=d.get("label", d["table"]),
            description=d.get("description", ""),
            pk_column=d.get("pk_column"),
            text_columns=d.get("text_columns", []),
            metadata_columns=d.get("metadata_columns", []),
            date_column=d.get("date_column"),
        )
        for d in data
    ]


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def load_saved_config(database: str) -> Optional[List[TableConfig]]:
    """
    Load a previously saved config for a database.
    Returns None if not found (first-time discovery needed).
    """
    path = CONFIGS_DIR / f"{database}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        configs = _configs_from_json(data)
        log.info(f"Loaded cached config for [{database}]: {len(configs)} tables.")
        return configs
    except Exception as e:
        log.error(f"Failed to load config for [{database}]: {e}")
        return None


def save_config(database: str, configs: List[TableConfig]):
    """Persist configs to disk so LLM is not called again next time."""
    path = CONFIGS_DIR / f"{database}.json"
    with open(path, "w") as f:
        json.dump(_configs_to_json(configs), f, indent=2)
    log.info(f"Saved config for [{database}] → {path}")


def discover_and_configure(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    force_rediscover: bool = False,
) -> List[TableConfig]:
    """
    Main entry point for new-DB intelligence.

    1. If a saved config exists (and force_rediscover=False) → return it.
    2. Otherwise: introspect schema → call Llama 3 8B → parse → save → return.

    The LLM is called ONCE per database (or when force_rediscover=True).
    """
    if not force_rediscover:
        cached = load_saved_config(database)
        if cached is not None:
            return cached

    log.info(f"Discovering schema for new database: [{database}]")
    schema = _get_schema_info(host, port, user, password, database)

    if not schema:
        log.warning(f"No usable tables found in [{database}].")
        return []

    schema_str = _schema_to_prompt(database, schema)
    log.info(f"Schema description built ({len(schema)} tables). Calling Llama 3 8B…")
    log.debug(f"Schema prompt:\n{schema_str}")

    prompt = (
        f"{schema_str}\n\n"
        f"Generate the JSON vectorisation config array for database '{database}'."
    )

    raw_response = llm_generate(prompt, system=_SYSTEM_PROMPT)
    log.debug(f"LLM raw response:\n{raw_response[:800]}")

    configs = _parse_llm_response(raw_response, database)

    if configs:
        save_config(database, configs)
    else:
        log.error(f"LLM returned no valid configs for [{database}]. Check Ollama logs.")

    return configs


def list_known_databases() -> List[str]:
    """Return list of databases that have saved configs."""
    return [p.stem for p in CONFIGS_DIR.glob("*.json")]


def print_config_summary(configs: List[TableConfig]):
    """Pretty-print a config list for CLI inspection."""
    for c in configs:
        print(f"  [{c.db}].{c.table}  →  '{c.label}'")
        print(f"    text cols  : {c.text_columns}")
        print(f"    meta cols  : {c.metadata_columns}")
        print(f"    pk         : {c.pk_column}  |  date: {c.date_column}")
        print()

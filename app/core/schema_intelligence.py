"""
app.core.schema_intelligence
─────────────────────────────
Uses LLM (via Ollama) to auto-generate vectorisation config for new databases.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import mysql.connector

from app.core.embedder import llm_generate
from app.core.schema_config import TableConfig

log = logging.getLogger("app.core.schema_intelligence")

CONFIGS_DIR = Path("./configs")
CONFIGS_DIR.mkdir(exist_ok=True)

_SKIP_TABLES = {
    "activity_log", "user_login_try", "user_password_history",
    "user_permission", "update_log", "clarification_reminder_log",
    "test", "website_setting",
}

_SKIP_COLUMNS = {
    "user_pass", "password", "login_token", "email_verification_code",
    "password_reset_code", "aadhar_no", "aadhar",
    "session_token", "api_key", "secret",
}


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
        if tbl in _SKIP_TABLES:
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
            file_columns = []
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
        file_columns = []
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


def discover_and_configure(
    host: str, port: int, user: str, password: str,
    database: str, force_rediscover: bool = False,
) -> List[TableConfig]:
    if not force_rediscover:
        cached = load_saved_config(database)
        if cached is not None:
            return cached

    log.info("Discovering schema for new database: [%s]", database)
    schema = _get_schema_info(host, port, user, password, database)

    if not schema:
        log.warning("No usable tables found in [%s].", database)
        return []

    schema_str = _schema_to_prompt(database, schema)
    log.info("Schema description built (%d tables). Calling LLM...", len(schema))

    prompt = (
        f"{schema_str}\n\n"
        f"Generate the JSON vectorisation config array for database '{database}'."
    )

    raw_response = llm_generate(prompt, system=_SYSTEM_PROMPT)
    configs = _parse_llm_response(raw_response, database)

    if configs:
        save_config(database, configs)
    else:
        log.error("LLM returned no valid configs for [%s].", database)

    return configs


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
        print()

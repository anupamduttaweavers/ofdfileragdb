"""
app/services/sync_service.py
─────────────────────────────
Background vectorization sync orchestrator.

Responsibilities (Single Responsibility):
  - Incremental sync: detect changed rows via date_column, vectorize only the delta
  - Full rescan: re-index all tables as a daily safety net
  - Per-table error isolation: one table failing does not block the rest
  - File processing: extract text from vendor-uploaded files using multithreading

Fallback strategy:
  - Ollama down → skip cycle, log warning, retry next tick
  - DB connection fails → skip that DB, continue with others
  - Table query fails → skip table, log error, continue
  - File extraction fails → log warning, vectorize DB record text only

Threading model:
  - ThreadPoolExecutor for parallel table processing during full rescan
  - FileProcessor (internal ThreadPoolExecutor) for parallel file I/O
  - Thread-safe FAISS upsert via the store's internal lock
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional

import mysql.connector

from app.services.file_processor import FileProcessor, has_file_columns, get_file_columns_for_table
from app.services.sync_state import SyncStateStore

log = logging.getLogger("app.services.sync")

_DATETIME_FMT = "%Y-%m-%dT%H:%M:%S"
_TABLE_WORKERS = int(os.getenv("SYNC_TABLE_WORKERS", "3"))
_FILE_WORKERS = int(os.getenv("SYNC_FILE_WORKERS", "4"))


class SyncService:
    """
    Stateful sync orchestrator.

    Initialised once at startup with a VectorizerConfig and SyncStateStore.
    The scheduler calls run_incremental() and run_full_rescan() on their
    respective intervals.
    """

    def __init__(self, vectorizer_config, state_store: SyncStateStore, faiss_store):
        self._vec_config = vectorizer_config
        self._state = state_store
        self._store = faiss_store
        self._running = False
        self._file_processor = FileProcessor(max_workers=_FILE_WORKERS)

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Incremental sync ────────────────────────────────────────

    def run_incremental(self) -> Dict[str, int]:
        """
        For each table with a date_column, fetch rows where
        date_column >= last_sync_at. Embed and upsert the delta.

        Returns {table_key: records_synced}.
        """
        if self._running:
            log.warning("Sync already in progress, skipping this cycle.")
            return {}

        self._running = True
        results: Dict[str, int] = {}

        try:
            self._reload_connections()

            if not self._check_ollama():
                log.warning("Ollama unreachable — skipping incremental sync cycle.")
                return results

            from app.core.schema_config import get_all_configs
            all_cfgs = get_all_configs()

            for cfg in all_cfgs:
                if not cfg.date_column:
                    continue

                table_key = f"{cfg.db}.{cfg.table}"
                tbl_state = self._state.get_table_state(table_key)
                since = tbl_state.last_sync_at if tbl_state else None

                count = self._sync_table_incremental(cfg, since)
                results[table_key] = count

            self._state.mark_incremental_complete(SyncStateStore.now_iso())
            log.info("Incremental sync complete: %s", results)

        except Exception as exc:
            log.error("Incremental sync failed unexpectedly: %s", exc, exc_info=True)
        finally:
            self._running = False

        return results

    # ── Full rescan ─────────────────────────────────────────────

    def run_full_rescan(self) -> Dict[str, int]:
        """
        Re-index every row from every configured table.
        Uses ThreadPoolExecutor for parallel table processing.
        Used as a daily safety net to catch anything the incremental sync missed.
        """
        if self._running:
            log.warning("Sync already in progress, skipping full rescan.")
            return {}

        self._running = True
        results: Dict[str, int] = {}

        try:
            self._reload_connections()

            if not self._check_ollama():
                log.warning("Ollama unreachable — skipping full rescan.")
                return results

            from app.core.schema_config import get_all_configs
            all_cfgs = get_all_configs()

            worker_count = min(_TABLE_WORKERS, len(all_cfgs))

            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="rescan") as executor:
                future_map = {
                    executor.submit(self._sync_table_full, cfg): f"{cfg.db}.{cfg.table}"
                    for cfg in all_cfgs
                }
                for future in as_completed(future_map):
                    table_key = future_map[future]
                    try:
                        results[table_key] = future.result()
                    except Exception as exc:
                        log.error("Rescan thread failed for %s: %s", table_key, exc)
                        results[table_key] = 0

            self._store.save()
            self._store.rebuild_index()
            self._store.save()

            self._state.mark_full_rescan_complete(SyncStateStore.now_iso())
            log.info("Full rescan complete: %d tables, %d total records",
                     len(results), sum(results.values()))

        except Exception as exc:
            log.error("Full rescan failed unexpectedly: %s", exc, exc_info=True)
        finally:
            self._running = False

        return results

    # ── Internal: per-table sync ────────────────────────────────

    def _sync_table_incremental(self, cfg, since: Optional[str]) -> int:
        """Fetch rows newer than `since` from one table, embed, and upsert."""
        table_key = f"{cfg.db}.{cfg.table}"
        start = time.perf_counter()

        conn_cfg = self._vec_config.db_connections.get(cfg.db)
        if conn_cfg is None:
            self._record_error(table_key, start, f"No DB connection for '{cfg.db}'")
            return 0

        try:
            conn = mysql.connector.connect(
                host=conn_cfg.host, port=conn_cfg.port,
                user=conn_cfg.user, password=conn_cfg.password,
                database=conn_cfg.database,
                charset="utf8mb4", use_unicode=True,
            )
        except Exception as exc:
            self._record_error(table_key, start, f"DB connect failed: {exc}")
            return 0

        try:
            rows = self._fetch_changed_rows(conn, cfg.table, cfg.date_column, since)
            if not rows:
                self._record_success(table_key, start, 0)
                return 0

            count = self._embed_and_upsert(cfg, rows)
            self._store.save()
            self._record_success(table_key, start, count)
            return count

        except Exception as exc:
            self._record_error(table_key, start, f"Sync error: {exc}")
            return 0
        finally:
            conn.close()

    def _sync_table_full(self, cfg) -> int:
        """Full re-read of a single table."""
        table_key = f"{cfg.db}.{cfg.table}"
        start = time.perf_counter()

        conn_cfg = self._vec_config.db_connections.get(cfg.db)
        if conn_cfg is None:
            self._record_error(table_key, start, f"No DB connection for '{cfg.db}'")
            return 0

        try:
            conn = mysql.connector.connect(
                host=conn_cfg.host, port=conn_cfg.port,
                user=conn_cfg.user, password=conn_cfg.password,
                database=conn_cfg.database,
                charset="utf8mb4", use_unicode=True,
            )
        except Exception as exc:
            self._record_error(table_key, start, f"DB connect failed: {exc}")
            return 0

        try:
            rows = self._fetch_all_rows(conn, cfg.table)
            if not rows:
                self._record_success(table_key, start, 0)
                return 0

            count = self._embed_and_upsert(cfg, rows)
            self._record_success(table_key, start, count)
            return count

        except Exception as exc:
            self._record_error(table_key, start, f"Full rescan error: {exc}")
            return 0
        finally:
            conn.close()

    # ── DB query helpers ────────────────────────────────────────

    def _fetch_changed_rows(
        self, conn, table: str, date_column: str, since: Optional[str],
    ) -> List[dict]:
        cursor = conn.cursor(dictionary=True)
        try:
            if since:
                cursor.execute(
                    f"SELECT * FROM `{table}` WHERE `{date_column}` >= %s ORDER BY `{date_column}`",
                    (since,),
                )
            else:
                cursor.execute(f"SELECT * FROM `{table}` ORDER BY `{date_column}`")
            return cursor.fetchall()
        finally:
            cursor.close()

    def _fetch_all_rows(self, conn, table: str) -> List[dict]:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(f"SELECT * FROM `{table}`")
            return cursor.fetchall()
        finally:
            cursor.close()

    # ── Embed + upsert ──────────────────────────────────────────

    def _embed_and_upsert(self, cfg, rows: List[dict]) -> int:
        from app.core.document_builder import build_document
        from app.core.embedder import embed_texts

        ids, texts, metas = [], [], []
        for row in rows:
            doc_id, text, meta = build_document(cfg, row)
            ids.append(doc_id)
            texts.append(text)
            metas.append(meta)

        if has_file_columns(cfg.table, db_name=cfg.db):
            self._vectorize_file_chunks(cfg, rows)

        batch_size = self._vec_config.embed_batch_size
        for i in range(0, len(texts), batch_size):
            chunk_ids = ids[i : i + batch_size]
            chunk_texts = texts[i : i + batch_size]
            chunk_metas = metas[i : i + batch_size]
            vecs = embed_texts(chunk_texts, batch_size=batch_size)
            self._store.upsert(chunk_ids, vecs.tolist(), chunk_texts, chunk_metas)

        return len(ids)

    def _vectorize_file_chunks(self, cfg, rows: List[dict]) -> int:
        """Use LangChain loaders to extract, chunk, and vectorize file content."""
        from app.core.embedder import embed_texts
        from app.core.lc_loaders import load_file, chunk_documents
        from app.services.file_processor import get_file_columns_for_table

        file_cols = get_file_columns_for_table(cfg.table, db_name=cfg.db)
        if not file_cols:
            return 0

        total_chunks = 0
        for row in rows:
            pk_val = str(row.get(cfg.pk_column, "")) if cfg.pk_column else ""
            for path_col, type_col in file_cols:
                raw_path = row.get(path_col)
                if not raw_path or not str(raw_path).strip():
                    continue

                file_type = str(row.get(type_col, "")) if type_col else ""
                source_meta = {
                    "source_db": cfg.db,
                    "source_table": cfg.table,
                    "doc_label": cfg.label,
                    "pk_value": pk_val,
                    "file_path": str(raw_path),
                    "file_name": str(row.get("file_name", "")),
                    "file_type": file_type,
                }

                try:
                    docs = load_file(str(raw_path), file_type=file_type, source_metadata=source_meta)
                    if not docs:
                        continue

                    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)

                    chunk_ids, chunk_texts, chunk_metas = [], [], []
                    for ci, chunk in enumerate(chunks):
                        chunk_id = f"{cfg.db}.{cfg.table}.{pk_val}.chunk_{ci}"
                        chunk_ids.append(chunk_id)
                        chunk_texts.append(chunk.page_content)
                        chunk_metas.append({
                            **chunk.metadata,
                            "doc_id": chunk_id,
                            "chunk_index": ci,
                            "total_chunks": len(chunks),
                        })

                    if chunk_texts:
                        batch_size = self._vec_config.embed_batch_size
                        for i in range(0, len(chunk_texts), batch_size):
                            b_ids = chunk_ids[i : i + batch_size]
                            b_texts = chunk_texts[i : i + batch_size]
                            b_metas = chunk_metas[i : i + batch_size]
                            vecs = embed_texts(b_texts, batch_size=batch_size)
                            self._store.upsert(b_ids, vecs.tolist(), b_texts, b_metas)
                        total_chunks += len(chunks)

                except Exception as exc:
                    log.warning("File chunk vectorization failed for %s: %s", raw_path, exc)

        if total_chunks > 0:
            log.info("Vectorized %d file chunks for %s.%s", total_chunks, cfg.db, cfg.table)
        return total_chunks

    # ── State recording helpers ─────────────────────────────────

    def _record_success(self, table_key: str, start: float, count: int) -> None:
        elapsed = (time.perf_counter() - start) * 1000
        self._state.update_table(
            table_key,
            last_sync_at=SyncStateStore.now_iso(),
            records_synced=count,
            status="success",
            duration_ms=round(elapsed, 2),
        )
        if count > 0:
            log.info("Synced %d records from %s (%.0fms)", count, table_key, elapsed)

    def _record_error(self, table_key: str, start: float, error_msg: str) -> None:
        elapsed = (time.perf_counter() - start) * 1000
        self._state.update_table(
            table_key,
            last_sync_at=SyncStateStore.now_iso(),
            records_synced=0,
            status="error",
            error_msg=error_msg,
            duration_ms=round(elapsed, 2),
        )
        log.error("Sync failed for %s: %s", table_key, error_msg)

    # ── Connection reload ────────────────────────────────────────

    def _reload_connections(self) -> None:
        """
        Reload DB connections from ConnectionStore at the start of each cycle.
        This picks up any databases registered at runtime via the API.
        """
        try:
            from app.core.connection_store import get_connection_store
            from app.core.vectorizer import DBConnectionConfig

            store = get_connection_store()
            for name, cred in store.load_all().items():
                if name not in self._vec_config.db_connections:
                    self._vec_config.db_connections[name] = DBConnectionConfig(
                        host=cred.host,
                        port=cred.port,
                        user=cred.user,
                        password=cred.password,
                        database=cred.database,
                    )
                    log.info("Sync service picked up new database: '%s'", name)
        except Exception as exc:
            log.warning("Failed to reload connections from store: %s", exc)

    # ── Ollama health check ─────────────────────────────────────

    @staticmethod
    def _check_ollama() -> bool:
        from app.core.embedder import is_ollama_available
        return is_ollama_available()

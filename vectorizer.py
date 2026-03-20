"""
vectorizer.py
──────────────
Background vectorisation engine.

Stack:
  MySQL  →  document_builder  →  nomic-embed-text (Ollama)  →  FAISS (in-memory + disk)

Threading model:
  ┌──────────────────────────────────────┐
  │ Indexer thread (1)                   │
  │  fetches rows → builds docs          │
  │  → puts (id, text, meta) on queue    │
  └──────────────┬───────────────────────┘
                 │  work queue
  ┌──────────────▼───────────────────────┐
  │ Embed + store workers (N)            │
  │  pop batch → embed via Ollama        │
  │  → FAISS upsert                      │
  └──────────────────────────────────────┘

After completion the index is saved to disk automatically.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mysql.connector

from document_builder import build_document
from embedder import embed_texts, _check_ollama
from faiss_store import FaissStore
from schema_config import TableConfig, get_all_configs

log = logging.getLogger("vectorizer")

SENTINEL = None


# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

@dataclass
class DBConnectionConfig:
    host:     str
    port:     int
    user:     str
    password: str
    database: str


@dataclass
class VectorizerConfig:
    # One entry per DB name: {"ofbdb": DBConnectionConfig(...), ...}
    db_connections:   Dict[str, DBConnectionConfig]
    faiss_persist_dir: str  = "./faiss_index"
    collection_name:   str  = "ofb_documents"
    embed_batch_size:  int  = 16    # texts per Ollama call batch
    fetch_chunk_size:  int  = 500   # rows per SQL fetch
    num_embed_workers: int  = 2
    max_queue_size:    int  = 1000
    save_every_n:      int  = 2000  # save index to disk every N upserts


# ─────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────

def _get_connection(cfg: DBConnectionConfig):
    return mysql.connector.connect(
        host=cfg.host, port=cfg.port,
        user=cfg.user, password=cfg.password,
        database=cfg.database,
        charset="utf8mb4", use_unicode=True,
    )


def _stream_table(conn, table: str, chunk: int = 500):
    cursor = conn.cursor(dictionary=True)
    offset = 0
    while True:
        cursor.execute(f"SELECT * FROM `{table}` LIMIT %s OFFSET %s", (chunk, offset))
        rows = cursor.fetchall()
        if not rows:
            break
        yield from rows
        offset += len(rows)
        if len(rows) < chunk:
            break
    cursor.close()


# ─────────────────────────────────────────────────────────────────────
# Embed + store worker
# ─────────────────────────────────────────────────────────────────────

class EmbedWorker(threading.Thread):
    def __init__(
        self,
        work_queue: queue.Queue,
        store: FaissStore,
        batch_size: int,
        save_every_n: int,
        worker_id: int,
    ):
        super().__init__(name=f"embed-worker-{worker_id}", daemon=True)
        self.q            = work_queue
        self.store        = store
        self.batch_size   = batch_size
        self.save_every_n = save_every_n
        self.total        = 0

    def run(self):
        batch: List[Tuple[str, str, Dict]] = []

        def flush():
            if not batch:
                return
            ids, texts, metas = zip(*batch)
            try:
                vecs = embed_texts(list(texts), batch_size=self.batch_size)
                self.store.upsert(
                    ids        = list(ids),
                    embeddings = vecs.tolist(),
                    documents  = list(texts),
                    metadatas  = list(metas),
                )
                self.total += len(batch)
                if self.total % self.save_every_n < len(batch):
                    self.store.save()
                    log.info(f"[{self.name}] checkpoint save at {self.total} docs")
            except Exception as exc:
                log.error(f"[{self.name}] flush error: {exc}", exc_info=True)
            batch.clear()

        while True:
            try:
                item = self.q.get(timeout=10)
            except queue.Empty:
                flush()
                continue

            if item is SENTINEL:
                flush()
                log.info(f"[{self.name}] done. total={self.total}")
                self.q.task_done()
                return

            batch.append(item)
            self.q.task_done()

            if len(batch) >= self.batch_size:
                flush()


# ─────────────────────────────────────────────────────────────────────
# Main engine
# ─────────────────────────────────────────────────────────────────────

class VectorizationEngine:
    """
    Full pipeline.  Works with any list of TableConfig objects —
    including auto-discovered ones from schema_intelligence.py.
    """

    def __init__(self, config: VectorizerConfig):
        self.config = config
        _check_ollama()
        self._store = FaissStore(config.faiss_persist_dir, config.collection_name)
        self._work_queue: queue.Queue = queue.Queue(maxsize=config.max_queue_size)
        self._workers: List[EmbedWorker] = []
        self._index_thread: Optional[threading.Thread] = None
        self._progress: Dict[str, int] = {}
        self._lock = threading.Lock()

    # ── Workers ────────────────────────────────────────────────────

    def _start_workers(self):
        self._workers = [
            EmbedWorker(
                work_queue   = self._work_queue,
                store        = self._store,
                batch_size   = self.config.embed_batch_size,
                save_every_n = self.config.save_every_n,
                worker_id    = i,
            )
            for i in range(self.config.num_embed_workers)
        ]
        for w in self._workers:
            w.start()

    def _stop_workers(self):
        for _ in self._workers:
            self._work_queue.put(SENTINEL)
        for w in self._workers:
            w.join()
        self._store.save()
        self._store.rebuild_index()
        self._store.save()

    # ── Indexing ───────────────────────────────────────────────────

    def _index_all(self, configs: List[TableConfig]):
        start = time.time()
        total = 0

        for cfg in configs:
            conn_cfg = self.config.db_connections.get(cfg.db)
            if conn_cfg is None:
                log.warning(f"No DB connection configured for [{cfg.db}]. Skipping {cfg.table}.")
                continue

            try:
                conn = _get_connection(conn_cfg)
            except Exception as e:
                log.error(f"Cannot connect to [{cfg.db}]: {e}")
                continue

            count = 0
            log.info(f"Indexing [{cfg.db}].{cfg.table}  ({cfg.label}) …")
            try:
                for row in _stream_table(conn, cfg.table, self.config.fetch_chunk_size):
                    doc_id, text, meta = build_document(cfg, row)
                    self._work_queue.put((doc_id, text, meta))
                    count += 1
            except Exception as exc:
                log.error(f"Error reading {cfg.db}.{cfg.table}: {exc}")
            finally:
                conn.close()

            log.info(f"  → queued {count} rows")
            with self._lock:
                self._progress[f"{cfg.db}.{cfg.table}"] = count
            total += count

        self._stop_workers()
        elapsed = time.time() - start
        log.info(f"Indexing complete. {total} docs in {elapsed:.1f}s. "
                 f"Index size: {self._store.count()} live docs.")

    # ── Public API ─────────────────────────────────────────────────

    def start_background_indexing(self, configs: Optional[List[TableConfig]] = None):
        """
        Non-blocking.  Pass explicit configs or leave None to use get_all_configs().
        """
        if configs is None:
            configs = get_all_configs()
        self._start_workers()
        self._index_thread = threading.Thread(
            target=self._index_all, args=(configs,), name="indexer", daemon=True,
        )
        self._index_thread.start()
        log.info(f"Background indexing started for {len(configs)} table(s).")

    def wait_until_complete(self):
        if self._index_thread:
            self._index_thread.join()

    def is_indexing(self) -> bool:
        return self._index_thread is not None and self._index_thread.is_alive()

    def progress(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._progress)

    def collection_count(self) -> int:
        return self._store.count()

    def get_store(self) -> FaissStore:
        return self._store

    # ── Incremental sync for a single table ────────────────────────

    def sync_table(self, db_name: str, table_name: str):
        """
        Re-index one table.  Call after DB updates.
        Embeds synchronously (no queue needed for small incremental batches).
        """
        configs = get_all_configs()
        cfg = next((c for c in configs if c.db == db_name and c.table == table_name), None)
        if cfg is None:
            raise ValueError(f"No config found for {db_name}.{table_name}")

        conn_cfg = self.config.db_connections.get(db_name)
        if conn_cfg is None:
            raise ValueError(f"No DB connection for {db_name}")

        conn = _get_connection(conn_cfg)
        ids, texts, metas = [], [], []

        for row in _stream_table(conn, table_name, self.config.fetch_chunk_size):
            doc_id, text, meta = build_document(cfg, row)
            ids.append(doc_id); texts.append(text); metas.append(meta)

        conn.close()

        if not ids:
            log.info(f"sync_table: no rows in {db_name}.{table_name}")
            return 0

        # Batch embed
        for i in range(0, len(texts), self.config.embed_batch_size):
            chunk_ids   = ids   [i : i + self.config.embed_batch_size]
            chunk_texts = texts [i : i + self.config.embed_batch_size]
            chunk_metas = metas [i : i + self.config.embed_batch_size]
            vecs = embed_texts(chunk_texts, batch_size=self.config.embed_batch_size)
            self._store.upsert(chunk_ids, vecs.tolist(), chunk_texts, chunk_metas)

        self._store.save()
        log.info(f"sync_table: {len(ids)} docs synced for {db_name}.{table_name}")
        return len(ids)

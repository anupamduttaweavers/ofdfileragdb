"""
app.core.vector_store
──────────────────────
FAISS-based in-memory vector store. No external service -- pure local.

Index type: IndexFlatIP (inner-product on L2-normalised vectors = cosine)
Persistence: two files per collection saved to disk.
Thread-safety: RLock guards all mutations.
"""

import logging
import os
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from app.core.embedder import EMBED_DIM

log = logging.getLogger("app.core.vector_store")


class FaissStore:
    """
    FAISS vector store with persistence and tombstone-based upsert.

    Public API:
        store.upsert(ids, embeddings, documents, metadatas)
        store.query(query_embedding, n_results, where) -> dict
        store.count()
        store.save() / load()
    """

    def __init__(self, persist_dir: str, name: str):
        self._dir = Path(persist_dir)
        self._name = name
        self._dir.mkdir(parents=True, exist_ok=True)

        self._index_path = self._dir / f"{name}.faiss"
        self._meta_path = self._dir / f"{name}.meta"
        self._lock = threading.RLock()

        self._doc_ids: List[str] = []
        self._documents: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._id_to_row: Dict[str, int] = {}

        self._index = faiss.IndexFlatIP(EMBED_DIM)

        if self._index_path.exists() and self._meta_path.exists():
            self._load()
        else:
            log.info("[%s] New empty index (dim=%d).", name, EMBED_DIM)

    # ── Persistence ────────────────────────────────────────────────

    def save(self):
        with self._lock:
            faiss.write_index(self._index, str(self._index_path))
            with open(self._meta_path, "wb") as f:
                pickle.dump({
                    "doc_ids": self._doc_ids,
                    "documents": self._documents,
                    "metadatas": self._metadatas,
                    "id_to_row": self._id_to_row,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.debug("[%s] Saved %d vectors to disk.", self._name, self._index.ntotal)

    def _load(self):
        with self._lock:
            self._index = faiss.read_index(str(self._index_path))
            with open(self._meta_path, "rb") as f:
                data = pickle.load(f)
            self._doc_ids = data["doc_ids"]
            self._documents = data["documents"]
            self._metadatas = data["metadatas"]
            self._id_to_row = data["id_to_row"]
        log.info("[%s] Loaded %d vectors from disk.", self._name, self._index.ntotal)

    def reload(self):
        """Re-read persisted index from disk (call after external writer saves)."""
        if self._index_path.exists() and self._meta_path.exists():
            self._load()
        else:
            log.warning("[%s] No persisted index found to reload.", self._name)

    # ── Mutations ──────────────────────────────────────────────────

    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        vecs = np.array(embeddings, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)

        with self._lock:
            for i, doc_id in enumerate(ids):
                vec = vecs[i : i + 1]

                if doc_id in self._id_to_row:
                    old_row = self._id_to_row[doc_id]
                    zero = np.zeros((1, EMBED_DIM), dtype=np.float32)
                    self._index.reconstruct(old_row, zero.reshape(-1))
                    self._documents[old_row] = ""
                    self._metadatas[old_row] = {"_tombstone": True}

                new_row = len(self._doc_ids)
                self._index.add(vec)
                self._doc_ids.append(doc_id)
                self._documents.append(documents[i])
                self._metadatas.append(metadatas[i])
                self._id_to_row[doc_id] = new_row

    def rebuild_index(self):
        with self._lock:
            live = [
                (doc_id, doc, meta, self._id_to_row[doc_id])
                for doc_id, doc, meta in zip(
                    self._doc_ids, self._documents, self._metadatas
                )
                if not meta.get("_tombstone", False)
            ]
            if not live:
                return

            new_index = faiss.IndexFlatIP(EMBED_DIM)
            new_ids, new_docs, new_metas, new_id_to_row = [], [], [], {}

            for new_row, (doc_id, doc, meta, old_row) in enumerate(live):
                vec = np.zeros((1, EMBED_DIM), dtype=np.float32)
                self._index.reconstruct(old_row, vec.reshape(-1))
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
                new_index.add(vec)
                new_ids.append(doc_id)
                new_docs.append(doc)
                new_metas.append(meta)
                new_id_to_row[doc_id] = new_row

            self._index = new_index
            self._doc_ids = new_ids
            self._documents = new_docs
            self._metadatas = new_metas
            self._id_to_row = new_id_to_row
            log.info("[%s] Rebuilt index: %d live docs.", self._name, len(new_ids))

    # ── Queries ────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            total = self._index.ntotal
            if total == 0:
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

            k = min(total, max(n_results * 10, 50))
            vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

            scores, rows = self._index.search(vec, k)
            scores = scores[0]
            rows = rows[0]

            results_ids, results_docs, results_metas, results_scores = [], [], [], []

            for score, row in zip(scores, rows):
                if row < 0 or row >= len(self._doc_ids):
                    continue
                meta = self._metadatas[row]
                if meta.get("_tombstone"):
                    continue
                if where and not _matches_where(meta, where):
                    continue

                results_ids.append(self._doc_ids[row])
                results_docs.append(self._documents[row])
                results_metas.append(meta)
                results_scores.append(float(score))

                if len(results_ids) >= n_results:
                    break

        return {
            "ids": [results_ids],
            "documents": [results_docs],
            "metadatas": [results_metas],
            "distances": [results_scores],
        }

    def count(self) -> int:
        with self._lock:
            return sum(1 for m in self._metadatas if not m.get("_tombstone"))

    def get_metadata_by_doc_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._id_to_row.get(doc_id)
            if row is None:
                return None
            meta = self._metadatas[row]
            if meta.get("_tombstone"):
                return None
            return meta

    def get_document_by_doc_id(self, doc_id: str) -> Optional[str]:
        with self._lock:
            row = self._id_to_row.get(doc_id)
            if row is None:
                return None
            if self._metadatas[row].get("_tombstone"):
                return None
            return self._documents[row]

    def find_file_path_for_doc(self, doc_id: str) -> Optional[str]:
        """Find a file_path for a doc by checking the doc itself and related chunks."""
        with self._lock:
            row = self._id_to_row.get(doc_id)
            if row is not None:
                meta = self._metadatas[row]
                if not meta.get("_tombstone") and meta.get("file_path"):
                    return meta["file_path"]

            chunk_0 = doc_id + ".chunk_0"
            row = self._id_to_row.get(chunk_0)
            if row is not None:
                meta = self._metadatas[row]
                if not meta.get("_tombstone") and meta.get("file_path"):
                    return meta["file_path"]

            return None


def _matches_where(meta: Dict, where: Dict) -> bool:
    if "$and" in where:
        return all(_matches_where(meta, clause) for clause in where["$and"])
    if "$or" in where:
        return any(_matches_where(meta, clause) for clause in where["$or"])

    for key, condition in where.items():
        val = meta.get(key)
        if isinstance(condition, dict):
            op, operand = next(iter(condition.items()))
            if op == "$eq":
                if str(val) != str(operand): return False
            elif op == "$ne":
                if str(val) == str(operand): return False
            elif op == "$in":
                if str(val) not in [str(x) for x in operand]: return False
            elif op == "$nin":
                if str(val) in [str(x) for x in operand]: return False
            elif op == "$gt":
                if not (val is not None and float(val) > float(operand)): return False
            elif op == "$gte":
                if not (val is not None and float(val) >= float(operand)): return False
            elif op == "$lt":
                if not (val is not None and float(val) < float(operand)): return False
            elif op == "$lte":
                if not (val is not None and float(val) <= float(operand)): return False
        else:
            if str(val) != str(condition):
                return False
    return True

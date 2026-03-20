"""
faiss_store.py
───────────────
FAISS-based in-memory vector store.  No external service – pure local.

Index type: IndexFlatIP  (inner-product on L2-normalised vectors = cosine)
Persistence: two files per collection saved to disk
    <dir>/<name>.faiss   – the FAISS binary index
    <dir>/<name>.meta    – pickle of doc_ids, documents, metadatas lists

Thread-safety: a single RLock guards all mutations (reads are safe without lock
in FAISS but we lock anyway for metadata consistency).
"""

import logging
import os
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from embedder import EMBED_DIM

log = logging.getLogger("faiss_store")


class FaissStore:
    """
    Drop-in replacement for ChromaDB collection.

    Public API mirrors what vectorizer.py / search.py need:
        store.upsert(ids, embeddings, documents, metadatas)
        store.query(query_embedding, n_results, where)  → dict
        store.count()
        store.save()
        store.load()
    """

    def __init__(self, persist_dir: str, name: str):
        self._dir  = Path(persist_dir)
        self._name = name
        self._dir.mkdir(parents=True, exist_ok=True)

        self._index_path = self._dir / f"{name}.faiss"
        self._meta_path  = self._dir / f"{name}.meta"
        self._lock = threading.RLock()

        # Parallel lists (same order as FAISS internal row index)
        self._doc_ids:   List[str]            = []
        self._documents: List[str]            = []
        self._metadatas: List[Dict[str, Any]] = []

        # Map doc_id → faiss row index for fast upsert lookup
        self._id_to_row: Dict[str, int]       = {}

        # FAISS index – IndexFlatIP gives exact cosine if vecs are L2-normalised
        self._index = faiss.IndexFlatIP(EMBED_DIM)

        if self._index_path.exists() and self._meta_path.exists():
            self._load()
        else:
            log.info(f"[{name}] New empty index (dim={EMBED_DIM}).")

    # ── Persistence ────────────────────────────────────────────────

    def save(self):
        with self._lock:
            faiss.write_index(self._index, str(self._index_path))
            with open(self._meta_path, "wb") as f:
                pickle.dump({
                    "doc_ids":   self._doc_ids,
                    "documents": self._documents,
                    "metadatas": self._metadatas,
                    "id_to_row": self._id_to_row,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.debug(f"[{self._name}] Saved {self._index.ntotal} vectors to disk.")

    def _load(self):
        with self._lock:
            self._index = faiss.read_index(str(self._index_path))
            with open(self._meta_path, "rb") as f:
                data = pickle.load(f)
            self._doc_ids   = data["doc_ids"]
            self._documents = data["documents"]
            self._metadatas = data["metadatas"]
            self._id_to_row = data["id_to_row"]
        log.info(f"[{self._name}] Loaded {self._index.ntotal} vectors from disk.")

    # ── Mutations ──────────────────────────────────────────────────

    def upsert(
        self,
        ids:        List[str],
        embeddings: List[List[float]],
        documents:  List[str],
        metadatas:  List[Dict[str, Any]],
    ):
        """
        Insert-or-update documents.

        FAISS IndexFlatIP does not support in-place update, so we use a
        "soft delete + re-add" approach:
          • New docs  → add to index, append to parallel lists
          • Existing  → mark old row as tombstoned (zero vector), append new row

        Tombstones accumulate until you call rebuild_index().  In practice
        for a few hundred-thousand documents this is negligible.
        """
        vecs = np.array(embeddings, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)

        with self._lock:
            for i, doc_id in enumerate(ids):
                vec = vecs[i : i + 1]

                if doc_id in self._id_to_row:
                    # Tombstone old row by zeroing its vector in the index
                    old_row = self._id_to_row[doc_id]
                    zero = np.zeros((1, EMBED_DIM), dtype=np.float32)
                    self._index.reconstruct(old_row, zero.reshape(-1))
                    # Update metadata in-place (same row won't be returned
                    # because cosine of zero vector = 0 < any real vector)
                    self._documents[old_row] = ""
                    self._metadatas[old_row] = {"_tombstone": True}

                new_row = len(self._doc_ids)
                self._index.add(vec)
                self._doc_ids.append(doc_id)
                self._documents.append(documents[i])
                self._metadatas.append(metadatas[i])
                self._id_to_row[doc_id] = new_row

    def rebuild_index(self):
        """
        Compact the index: removes tombstoned rows.
        Call periodically (e.g. after a full re-index).
        """
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
                # Re-normalise in case of floating point drift
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
                new_index.add(vec)
                new_ids.append(doc_id)
                new_docs.append(doc)
                new_metas.append(meta)
                new_id_to_row[doc_id] = new_row

            self._index    = new_index
            self._doc_ids  = new_ids
            self._documents = new_docs
            self._metadatas = new_metas
            self._id_to_row = new_id_to_row
            log.info(f"[{self._name}] Rebuilt index: {len(new_ids)} live docs.")

    # ── Queries ────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Returns dict with keys: ids, documents, metadatas, distances
        (each value is a list-of-lists to match ChromaDB's response shape).
        """
        with self._lock:
            total = self._index.ntotal
            if total == 0:
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

            # Search more than needed to allow for post-filter + tombstones
            k = min(total, max(n_results * 10, 50))
            vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

            scores, rows = self._index.search(vec, k)
            scores = scores[0]
            rows   = rows[0]

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
            "ids":       [results_ids],
            "documents": [results_docs],
            "metadatas": [results_metas],
            "distances": [results_scores],   # scores here are similarities (IP)
        }

    def count(self) -> int:
        with self._lock:
            # Subtract tombstoned rows
            live = sum(1 for m in self._metadatas if not m.get("_tombstone"))
            return live


# ─────────────────────────────────────────────────────────────────────
# Metadata filter interpreter
# Supports: {"field": {"$eq": v}}, {"field": {"$in": [v1,v2]}},
#           {"$and": [...]}, {"$or": [...]}
# ─────────────────────────────────────────────────────────────────────

def _matches_where(meta: Dict, where: Dict) -> bool:
    if "$and" in where:
        return all(_matches_where(meta, clause) for clause in where["$and"])
    if "$or" in where:
        return any(_matches_where(meta, clause) for clause in where["$or"])

    for key, condition in where.items():
        val = meta.get(key)
        if isinstance(condition, dict):
            op, operand = next(iter(condition.items()))
            if   op == "$eq":  
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

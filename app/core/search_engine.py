"""
app.core.search_engine
───────────────────────
Semantic search over the FAISS index.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.embedder import embed_query
from app.core.vector_store import FaissStore

log = logging.getLogger("app.core.search_engine")


@dataclass
class SearchResult:
    rank: int
    doc_id: str
    score: float
    label: str
    source_db: str
    source_table: str
    snippet: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        bar = "█" * int(self.score * 20)
        return (
            f"#{self.rank}  [{self.label}]  similarity={self.score:.4f}  {bar}\n"
            f"    {self.source_db}.{self.source_table}\n"
            f"    {self.snippet[:280].replace(chr(10),' | ')}\n"
        )


class DocumentSearchEngine:
    """Wraps FaissStore for semantic NL search."""

    def __init__(
        self,
        faiss_persist_dir: str = "./faiss_index",
        collection_name: str = "ofb_documents",
    ):
        self._store = FaissStore(faiss_persist_dir, collection_name)
        log.info("Search engine ready. Index: %d live docs.", self._store.count())

    def search(
        self,
        query: str,
        top_k: int = 10,
        db_filter: Optional[str] = None,
        table_filter: Optional[str] = None,
        where: Optional[Dict] = None,
    ) -> List[SearchResult]:
        meta_where = where
        if db_filter or table_filter:
            conditions = []
            if db_filter:
                conditions.append({"source_db": {"$eq": db_filter}})
            if table_filter:
                conditions.append({"source_table": {"$eq": table_filter}})
            meta_where = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        qvec = embed_query(query).tolist()

        raw = self._store.query(
            query_embedding=qvec[0],
            n_results=top_k,
            where=meta_where,
        )

        results: List[SearchResult] = []
        for i, (doc_id, doc_text, meta, score) in enumerate(zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        )):
            results.append(SearchResult(
                rank=i + 1,
                doc_id=doc_id,
                score=round(float(score), 4),
                label=meta.get("doc_label", ""),
                source_db=meta.get("source_db", ""),
                source_table=meta.get("source_table", ""),
                snippet=doc_text[:400],
                metadata=meta,
            ))

        return results

    def index_count(self) -> int:
        return self._store.count()

    def reload_index(self):
        """Reload the FAISS index from disk after external writes."""
        self._store.reload()
        log.info("Search engine reloaded. Index: %d live docs.", self._store.count())

    @property
    def store(self) -> FaissStore:
        return self._store


EXAMPLE_QUERIES = [
    {"query": "Grade I vendors with valid registration expiring in 2026",
     "note": "vendorcom_filtered -> Grade, Dtexpiry"},
    {"query": "vendor registration certificates for propellant and ammunition items",
     "note": "certificate_report.items, certificate_report_product"},
    {"query": "debarred vendors and their reasons",
     "note": "vendor_debar.reason + debar_date"},
    {"query": "tender advertisements for 9mm pistol cartridge items",
     "note": "advertisement_items.item_name + item_description"},
]

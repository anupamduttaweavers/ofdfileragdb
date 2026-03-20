"""
search.py
──────────
Semantic search over the FAISS index.
Embeddings via nomic-embed-text through Ollama (fully offline).
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from embedder import embed_query
from faiss_store import FaissStore

log = logging.getLogger("search")


@dataclass
class SearchResult:
    rank:         int
    doc_id:       str
    score:        float         # cosine similarity 0–1
    label:        str
    source_db:    str
    source_table: str
    snippet:      str           # first 400 chars of document
    metadata:     Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        bar = "█" * int(self.score * 20)
        return (
            f"#{self.rank}  [{self.label}]  similarity={self.score:.4f}  {bar}\n"
            f"    {self.source_db}.{self.source_table}\n"
            f"    {self.snippet[:280].replace(chr(10),' | ')}\n"
        )


class DocumentSearchEngine:
    """
    Wraps FaissStore for semantic NL search.
    Instantiate once; reuse for all queries.
    """

    def __init__(
        self,
        faiss_persist_dir: str  = "./faiss_index",
        collection_name:   str  = "ofb_documents",
    ):
        self._store = FaissStore(faiss_persist_dir, collection_name)
        log.info(f"Search engine ready.  Index: {self._store.count()} live docs.")

    def search(
        self,
        query:        str,
        top_k:        int            = 10,
        db_filter:    Optional[str]  = None,   # restrict to "ofbdb" or "misofb" etc.
        table_filter: Optional[str]  = None,
        where:        Optional[Dict] = None,   # raw metadata filter
    ) -> List[SearchResult]:
        """
        Embed the query via nomic-embed-text → cosine search in FAISS.
        """
        # Build metadata where-clause
        meta_where = where
        if db_filter or table_filter:
            conditions = []
            if db_filter:
                conditions.append({"source_db":    {"$eq": db_filter}})
            if table_filter:
                conditions.append({"source_table": {"$eq": table_filter}})
            meta_where = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        qvec = embed_query(query).tolist()     # shape (1, 768)

        raw = self._store.query(
            query_embedding = qvec[0],
            n_results        = top_k,
            where            = meta_where,
        )

        results: List[SearchResult] = []
        for i, (doc_id, doc_text, meta, score) in enumerate(zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],    # these are already similarity scores (IP)
        )):
            results.append(SearchResult(
                rank         = i + 1,
                doc_id       = doc_id,
                score        = round(float(score), 4),
                label        = meta.get("doc_label", ""),
                source_db    = meta.get("source_db", ""),
                source_table = meta.get("source_table", ""),
                snippet      = doc_text[:400],
                metadata     = meta,
            ))

        return results

    def search_pretty(self, query: str, top_k: int = 5, **kwargs) -> str:
        results = self.search(query, top_k=top_k, **kwargs)
        lines = [
            "",
            "═" * 60,
            f"Query  : {query!r}",
            f"Results: {len(results)}",
            "═" * 60,
        ]
        lines += [str(r) for r in results]
        return "\n".join(lines)

    def index_count(self) -> int:
        return self._store.count()


# ─────────────────────────────────────────────────────────────────────
# Example queries (all realistic for ofbdb + misofb)
# ─────────────────────────────────────────────────────────────────────

EXAMPLE_QUERIES = [
    # ofbdb
    {"query": "Grade I vendors with valid registration expiring in 2026",
     "note":  "vendorcom_filtered → Grade, Dtexpiry"},
    {"query": "vendor registration certificates for propellant and ammunition items",
     "note":  "certificate_report.items, certificate_report_product"},
    {"query": "debarred vendors and their reasons",
     "note":  "vendor_debar.reason + debar_date"},
    {"query": "tender advertisements for 9mm pistol cartridge items",
     "note":  "advertisement_items.item_name + item_description"},
    {"query": "DDP category deemed registered vendors in Maharashtra",
     "note":  "vendorcom_filtered → Categorisation=DDP, deemed=1, State"},
    {"query": "item master drawing number for artillery shell 155mm",
     "note":  "item_master.drawing_no, item_description"},
    {"query": "factories located in Tamil Nadu",
     "note":  "factory_master.state"},
    # misofb
    {"query": "PAN card list of Group A officers in unit 101",
     "note":  "t_ofbpismas.pan_no, grade, unit"},
    {"query": "employees who have not completed mandatory iGOT courses",
     "note":  "consumptionreport.Live_CBP_Plan_Mandate=1, Status != Completed"},
    {"query": "top learners by karma points and learning hours",
     "note":  "userreport.Karma_Points, Total_Learning_Hours"},
    {"query": "pass percentage for cybersecurity assessment in Ministry of Defence",
     "note":  "userassessmentreport.Course_Name, Ministry, Latest_Percentage_Achieved"},
    {"query": "iGOT certificates generated for completed training in 2025",
     "note":  "consumptionreport.Certificate_Generated, Completed_On"},
    {"query": "female SC category employees joined after 2022",
     "note":  "t_ofbpismas.gender=F, category=SC, dt_joining"},
]


if __name__ == "__main__":
    import sys
    engine = DocumentSearchEngine()
    if len(sys.argv) > 1:
        print(engine.search_pretty(" ".join(sys.argv[1:]), top_k=5))
    else:
        for ex in EXAMPLE_QUERIES[:4]:
            print(engine.search_pretty(ex["query"], top_k=3))
            print(f"  ↳ {ex['note']}\n")

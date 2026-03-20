"""
app.core.lc_vector_store
─────────────────────────
LangChain FAISS vector store adapter.

Wraps the existing FaissStore behind LangChain's retriever interface
so that LangGraph chains can use it natively.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from app.core.lc_embeddings import get_lc_embeddings
from app.core.vector_store import FaissStore

log = logging.getLogger("app.core.lc_vector_store")


class FaissRetrieverAdapter:
    """
    Adapts FaissStore to a LangChain-compatible retriever interface.

    Provides similarity_search() that returns List[Document],
    which is what LangGraph RAG nodes expect.
    """

    def __init__(self, store: FaissStore, top_k: int = 10):
        self._store = store
        self._top_k = top_k
        self._embeddings = get_lc_embeddings()

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        db_filter: Optional[str] = None,
        table_filter: Optional[str] = None,
    ) -> List[Document]:
        """Search and return LangChain Document objects."""
        effective_k = k or self._top_k
        query_vec = self._embeddings.embed_query(query)

        where = None
        if db_filter or table_filter:
            conditions = []
            if db_filter:
                conditions.append({"source_db": {"$eq": db_filter}})
            if table_filter:
                conditions.append({"source_table": {"$eq": table_filter}})
            where = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        raw = self._store.query(
            query_embedding=query_vec,
            n_results=effective_k,
            where=where,
        )

        docs = []
        for doc_id, text, meta, score in zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            doc = Document(
                page_content=text,
                metadata={
                    **meta,
                    "doc_id": doc_id,
                    "similarity_score": round(float(score), 4),
                },
            )
            docs.append(doc)

        return docs

    def similarity_search_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        db_filter: Optional[str] = None,
    ) -> List[tuple]:
        """Return (Document, score) tuples."""
        docs = self.similarity_search(query, k=k, db_filter=db_filter)
        return [
            (doc, doc.metadata.get("similarity_score", 0.0))
            for doc in docs
        ]

    @property
    def store(self) -> FaissStore:
        return self._store

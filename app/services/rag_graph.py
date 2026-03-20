"""
app.services.rag_graph
───────────────────────
LangGraph-based self-corrective RAG pipeline.

Flow:
  Query -> Retrieve -> Grade Documents -> (if irrelevant) Rewrite Query -> Retrieve
                                       -> (if relevant) Rerank -> Generate -> Hallucination Check
                                                                           -> (if grounded) Return
                                                                           -> (if not) Regenerate

Max 2 retry loops to prevent infinite cycles.
Graceful degradation when Ollama is unavailable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document

from app.config import get_settings
from app.exceptions import EmbeddingError, OllamaUnavailableError

log = logging.getLogger("app.services.rag_graph")

_MAX_RETRIES = 2

_RAG_SYSTEM_PROMPT = (
    "You are an expert document analyst for the Ordnance Factory Board (OFB) system. "
    "Answer the user's question based ONLY on the provided context documents. "
    "IMPORTANT: Consider ALL provided documents carefully, even if they come from "
    "different database tables or have different formats. Synthesize information "
    "from all relevant sources into a comprehensive answer. "
    "Do NOT dismiss any document without a clear reason. "
    "If the information is not in the context, state that clearly. "
    "When citing information, reference the source document number in square brackets like [1], [2]. "
    "Be accurate and thorough."
)

_GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question.\n\n"
    "Question: {question}\n\n"
    "Document:\n{document}\n\n"
    "Does this document contain information relevant to answering the question? "
    "Answer with ONLY 'yes' or 'no'."
)

_REWRITE_PROMPT = (
    "You are a query rewriter. The original query did not retrieve good results.\n\n"
    "Original query: {question}\n\n"
    "Rewrite this query to be more specific and likely to retrieve relevant documents. "
    "Return ONLY the rewritten query, nothing else."
)

_HALLUCINATION_PROMPT = (
    "You are a grader assessing whether an answer is grounded in the provided documents.\n\n"
    "Documents:\n{documents}\n\n"
    "Answer:\n{generation}\n\n"
    "Is the answer supported by the documents? Answer with ONLY 'yes' or 'no'."
)


@dataclass
class RagSource:
    rank: int
    doc_id: str
    score: float
    label: str
    source_db: str
    source_table: str
    snippet: str
    metadata: dict
    rerank_score: Optional[float] = None
    file_download_url: Optional[str] = None


@dataclass
class RagResult:
    query: str
    answer: str
    sources: List[RagSource]
    model_used: str
    reranked: bool
    graded: bool
    hallucination_checked: bool
    query_rewritten: bool
    final_query: str
    elapsed_ms: float


class RAGState(TypedDict, total=False):
    question: str
    original_question: str
    documents: List[Document]
    generation: str
    retry_count: int
    relevant_docs: List[Document]
    reranked: bool
    graded: bool
    hallucination_passed: bool
    query_rewritten: bool


class RAGPipeline:
    """
    LangGraph-style RAG pipeline with self-correction.

    Nodes:
    1. retrieve - vector search
    2. grade_documents - LLM grades each doc for relevance
    3. rewrite_query - LLM rewrites query if docs are irrelevant
    4. rerank - FlashRank cross-encoder reranking
    5. generate - LLM generates answer
    6. check_hallucination - LLM verifies answer is grounded
    """

    def __init__(self, retriever, *, default_top_k: int = 10, rerank_enabled: bool = True):
        self._retriever = retriever
        self._default_top_k = default_top_k
        self._rerank_enabled = rerank_enabled

    def ask(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        db_filter: Optional[str] = None,
        temperature: float = 0.1,
        rerank: Optional[bool] = None,
    ) -> RagResult:
        start = time.perf_counter()
        k = top_k or self._default_top_k
        should_rerank = rerank if rerank is not None else self._rerank_enabled

        state: RAGState = {
            "question": query,
            "original_question": query,
            "documents": [],
            "generation": "",
            "retry_count": 0,
            "relevant_docs": [],
            "reranked": False,
            "graded": False,
            "hallucination_passed": False,
            "query_rewritten": False,
        }

        state = self._node_retrieve(state, k=k, db_filter=db_filter)

        if not state["documents"]:
            elapsed = (time.perf_counter() - start) * 1000
            return RagResult(
                query=query, answer="No relevant documents found for your query.",
                sources=[], model_used="none", reranked=False, graded=False,
                hallucination_checked=False, query_rewritten=False,
                final_query=query, elapsed_ms=round(elapsed, 2),
            )

        if self._is_ollama_available():
            state = self._node_grade_documents(state)

            if not state["relevant_docs"] and state["retry_count"] < _MAX_RETRIES:
                state = self._node_rewrite_query(state)
                state["retry_count"] += 1
                state = self._node_retrieve(state, k=k, db_filter=db_filter)
                state = self._node_grade_documents(state)

            working_docs = state["relevant_docs"] if state["relevant_docs"] else state["documents"]
        else:
            working_docs = state["documents"]

        if should_rerank and len(working_docs) > 1:
            state, working_docs = self._node_rerank(state, working_docs)

        if self._is_ollama_available():
            state = self._node_generate(state, working_docs, temperature=temperature)

            state = self._node_check_hallucination(state, working_docs)

            if not state["hallucination_passed"] and state["retry_count"] < _MAX_RETRIES:
                state["retry_count"] += 1
                state = self._node_generate(state, working_docs, temperature=temperature)
        else:
            state["generation"] = (
                "The LLM service is currently unavailable. "
                "Below are the most relevant documents found for your query."
            )

        sources = self._build_sources(working_docs)

        elapsed = (time.perf_counter() - start) * 1000
        return RagResult(
            query=query,
            answer=state["generation"],
            sources=sources,
            model_used=self._get_model_name(),
            reranked=state["reranked"],
            graded=state["graded"],
            hallucination_checked=state["hallucination_passed"],
            query_rewritten=state["query_rewritten"],
            final_query=state["question"],
            elapsed_ms=round(elapsed, 2),
        )

    def _node_retrieve(self, state: RAGState, *, k: int, db_filter: Optional[str]) -> RAGState:
        try:
            docs = self._retriever.similarity_search(
                state["question"], k=k, db_filter=db_filter,
            )
            docs = self._enrich_with_file_chunks(docs)
            state["documents"] = docs
        except Exception as exc:
            log.error("Retrieval failed: %s", exc)
            raise EmbeddingError(f"Search failed: {exc}") from exc
        return state

    @staticmethod
    def _enrich_with_file_chunks(docs: List[Document]) -> List[Document]:
        """For each retrieved DB record that has associated file chunks, inject
        the first chunk's content so the LLM sees the actual file text."""
        from app.dependencies import get_search_engine
        try:
            store = get_search_engine().store
        except Exception:
            return docs

        seen_ids = {d.metadata.get("doc_id", "") for d in docs}
        enriched: List[Document] = []

        for doc in docs:
            enriched.append(doc)
            doc_id = doc.metadata.get("doc_id", "")
            if not doc_id or ".chunk_" in doc_id:
                continue

            chunk_id = doc_id + ".chunk_0"
            if chunk_id in seen_ids:
                continue

            chunk_text = store.get_document_by_doc_id(chunk_id)
            chunk_meta = store.get_metadata_by_doc_id(chunk_id)
            if chunk_text and chunk_meta:
                chunk_meta = dict(chunk_meta)
                chunk_meta["doc_id"] = chunk_id
                parent_score = doc.metadata.get("similarity_score", 0.0)
                if not chunk_meta.get("similarity_score"):
                    chunk_meta["similarity_score"] = parent_score
                enriched.append(Document(page_content=chunk_text, metadata=chunk_meta))
                seen_ids.add(chunk_id)
                log.debug("Enriched context with file chunk: %s", chunk_id)

        return enriched

    def _node_grade_documents(self, state: RAGState) -> RAGState:
        relevant = []
        for doc in state["documents"]:
            try:
                score = self._grade_single(state["question"], doc.page_content[:1500])
                if score:
                    relevant.append(doc)
            except Exception as exc:
                log.debug("Grading failed for doc, including by default: %s", exc)
                relevant.append(doc)

        state["relevant_docs"] = relevant
        state["graded"] = True
        log.info("Graded %d docs: %d relevant", len(state["documents"]), len(relevant))
        return state

    def _node_rewrite_query(self, state: RAGState) -> RAGState:
        try:
            from app.core.lc_llm import invoke_llm

            prompt = _REWRITE_PROMPT.format(question=state["question"])
            rewritten = invoke_llm(prompt, temperature=0.0)
            if rewritten and len(rewritten) > 5:
                state["question"] = rewritten.strip()
                state["query_rewritten"] = True
                log.info("Query rewritten: '%s' -> '%s'", state["original_question"], state["question"])
        except Exception as exc:
            log.warning("Query rewrite failed: %s", exc)
        return state

    def _node_rerank(self, state: RAGState, docs: List[Document]) -> tuple:
        try:
            from app.core.lc_reranker import rerank_documents

            reranked = rerank_documents(state["question"], docs)
            state["reranked"] = True
            return state, reranked
        except Exception as exc:
            log.warning("Reranking failed: %s. Using original order.", exc)
            return state, docs

    def _node_generate(self, state: RAGState, docs: List[Document], *, temperature: float) -> RAGState:
        from app.core.lc_llm import invoke_llm

        context_parts = []
        for i, doc in enumerate(docs):
            meta = doc.metadata
            header = f"[{i+1}] {meta.get('doc_label', '')} ({meta.get('source_db', '')}.{meta.get('source_table', '')})"
            context_parts.append(f"{header}\n{doc.page_content[:1500]}")

        context_block = "\n\n---\n\n".join(context_parts)

        prompt = (
            f"Context Documents:\n\n{context_block}\n\n"
            f"User Question: {state['question']}\n\n"
            f"Instructions: Review ALL context documents above. Extract and combine "
            f"relevant information from every source, regardless of which database or "
            f"table it comes from. Provide a clear, well-structured, and comprehensive answer."
        )

        try:
            answer = invoke_llm(prompt, system=_RAG_SYSTEM_PROMPT, temperature=temperature)
            state["generation"] = answer if answer else "LLM returned an empty response."
        except Exception as exc:
            log.warning("Generation failed: %s", exc)
            state["generation"] = "Failed to generate an answer. Please review the source documents directly."

        return state

    def _node_check_hallucination(self, state: RAGState, docs: List[Document]) -> RAGState:
        try:
            from app.core.lc_llm import invoke_llm

            doc_texts = "\n\n".join(d.page_content[:1000] for d in docs[:5])
            prompt = _HALLUCINATION_PROMPT.format(
                documents=doc_texts,
                generation=state["generation"],
            )
            result = invoke_llm(prompt, temperature=0.0)
            state["hallucination_passed"] = "yes" in result.lower()

            if not state["hallucination_passed"]:
                log.info("Hallucination check failed, may regenerate.")
        except Exception as exc:
            log.debug("Hallucination check failed: %s. Assuming grounded.", exc)
            state["hallucination_passed"] = True

        return state

    def _grade_single(self, question: str, document: str) -> bool:
        from app.core.lc_llm import invoke_llm

        prompt = _GRADE_PROMPT.format(question=question, document=document)
        result = invoke_llm(prompt, temperature=0.0)
        return "yes" in result.lower()

    def _build_sources(self, docs: List[Document]) -> List[RagSource]:
        from app.dependencies import get_search_engine
        try:
            store = get_search_engine().store
        except Exception:
            store = None

        sources = []
        for i, doc in enumerate(docs):
            meta = doc.metadata
            source_table = meta.get("source_table", "")
            doc_id = meta.get("doc_id", "")

            file_url = None
            if meta.get("file_path"):
                file_url = f"/api/v1/files/download/{doc_id}"
            elif store and doc_id:
                fp = store.find_file_path_for_doc(doc_id)
                if fp:
                    file_url = f"/api/v1/files/download/{doc_id}"

            sources.append(RagSource(
                rank=i + 1,
                doc_id=doc_id,
                score=meta.get("similarity_score", 0.0),
                label=meta.get("doc_label", ""),
                source_db=meta.get("source_db", ""),
                source_table=source_table,
                snippet=doc.page_content[:400],
                metadata=meta,
                rerank_score=meta.get("rerank_score"),
                file_download_url=file_url,
            ))
        return sources

    @staticmethod
    def _is_ollama_available() -> bool:
        from app.core.embedder import is_ollama_available
        return is_ollama_available()

    @staticmethod
    def _get_model_name() -> str:
        from app.core.embedder import LLM_MODEL
        return LLM_MODEL

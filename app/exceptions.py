"""
app/exceptions.py
──────────────────
Centralised exception hierarchy following Open/Closed principle.

Base exception class provides structured error info; concrete subclasses
add domain-specific context. The global FastAPI handler in app/main.py
catches these and returns consistent JSON error responses.

Fallback strategy:
  - OllamaUnavailableError  → service degrades; search still works, RAG returns
    a clear "LLM offline" message instead of crashing.
  - DatabaseConnectionError → per-DB fallback; other DBs keep working.
  - IndexNotReadyError      → returns 503 with retry-after hint.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class AppError(Exception):
    """
    Base for all application-domain errors.
    Every subclass carries a machine-readable error_code and HTTP status.
    """

    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        detail: str = "An unexpected error occurred.",
        *,
        error_code: Optional[str] = None,
        status_code: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(detail)
        self.detail = detail
        if error_code is not None:
            self.error_code = error_code
        if status_code is not None:
            self.status_code = status_code
        self.context = context or {}


# ── Authentication / Authorisation ──────────────────────────────

class AuthenticationError(AppError):
    status_code = 401
    error_code = "AUTH_FAILED"

    def __init__(self, detail: str = "Invalid or missing API key."):
        super().__init__(detail)


class ForbiddenError(AppError):
    status_code = 403
    error_code = "FORBIDDEN"

    def __init__(self, detail: str = "You do not have permission for this action."):
        super().__init__(detail)


# ── External Service Errors ─────────────────────────────────────

class OllamaUnavailableError(AppError):
    """Ollama server is unreachable — the system degrades gracefully."""
    status_code = 503
    error_code = "OLLAMA_UNAVAILABLE"

    def __init__(self, detail: str = "Ollama LLM server is not reachable. RAG and embedding services are temporarily unavailable."):
        super().__init__(detail)


class EmbeddingError(AppError):
    """Failed to generate embeddings for the given text."""
    status_code = 502
    error_code = "EMBEDDING_FAILED"

    def __init__(self, detail: str = "Failed to generate embeddings."):
        super().__init__(detail)


class LLMGenerationError(AppError):
    """LLM failed to produce a response."""
    status_code = 502
    error_code = "LLM_GENERATION_FAILED"

    def __init__(self, detail: str = "LLM failed to generate a response."):
        super().__init__(detail)


# ── Database Errors ─────────────────────────────────────────────

class DatabaseConnectionError(AppError):
    """Cannot connect to a specific MySQL database."""
    status_code = 503
    error_code = "DB_CONNECTION_FAILED"

    def __init__(self, db_name: str, detail: Optional[str] = None):
        msg = detail or f"Cannot connect to database '{db_name}'."
        super().__init__(msg, context={"database": db_name})


class DatabaseQueryError(AppError):
    """A SQL query failed at runtime."""
    status_code = 500
    error_code = "DB_QUERY_FAILED"

    def __init__(self, detail: str = "A database query failed."):
        super().__init__(detail)


# ── Index / Vector Store Errors ─────────────────────────────────

class IndexNotReadyError(AppError):
    """FAISS index is not loaded or is being rebuilt."""
    status_code = 503
    error_code = "INDEX_NOT_READY"

    def __init__(self, detail: str = "The search index is not ready. It may be loading or rebuilding."):
        super().__init__(detail)


class IndexingInProgressError(AppError):
    """An indexing job is already running."""
    status_code = 409
    error_code = "INDEXING_IN_PROGRESS"

    def __init__(self, detail: str = "An indexing operation is already in progress."):
        super().__init__(detail)


# ── Validation / Business Logic ─────────────────────────────────

class ValidationError(AppError):
    status_code = 422
    error_code = "VALIDATION_ERROR"

    def __init__(self, detail: str = "Request validation failed."):
        super().__init__(detail)


class ResourceNotFoundError(AppError):
    status_code = 404
    error_code = "NOT_FOUND"

    def __init__(self, resource: str, identifier: str = ""):
        msg = f"{resource} not found."
        if identifier:
            msg = f"{resource} '{identifier}' not found."
        super().__init__(msg, context={"resource": resource, "identifier": identifier})


# ── Database Management Errors ───────────────────────────────────

class DatabaseAlreadyExistsError(AppError):
    """A database with this name is already registered."""
    status_code = 409
    error_code = "DB_ALREADY_EXISTS"

    def __init__(self, name: str):
        super().__init__(
            f"Database '{name}' is already registered.",
            context={"database": name},
        )


# ── File Processing Errors ───────────────────────────────────────

class FileExtractionError(AppError):
    """Failed to extract text from a file."""
    status_code = 422
    error_code = "FILE_EXTRACTION_FAILED"

    def __init__(self, file_path: str, detail: str = ""):
        msg = detail or f"Failed to extract text from file '{file_path}'."
        super().__init__(msg, context={"file_path": file_path})


# ── RAG Pipeline Errors ─────────────────────────────────────────

class RerankerError(AppError):
    """Reranking stage failed."""
    status_code = 502
    error_code = "RERANKER_FAILED"

    def __init__(self, detail: str = "Document reranking failed."):
        super().__init__(detail)


class QueryRewriteError(AppError):
    """Query rewrite stage failed."""
    status_code = 502
    error_code = "QUERY_REWRITE_FAILED"

    def __init__(self, detail: str = "Query rewriting failed."):
        super().__init__(detail)

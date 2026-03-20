"""
app/models/responses.py
────────────────────────
Pydantic v2 response schemas.

Every API endpoint returns one of these typed models so clients get
a predictable, documented JSON contract.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Shared building blocks ──────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error envelope returned by the global exception handler."""
    error_code: str = Field(..., description="Machine-readable error code")
    detail: str = Field(..., description="Human-readable error message")
    context: Dict[str, Any] = Field(default_factory=dict, description="Extra context about the error")


class SearchResultItem(BaseModel):
    """One document hit from vector search."""
    rank: int
    doc_id: str
    score: float = Field(..., description="Cosine similarity (0-1)")
    label: str
    source_db: str
    source_table: str
    snippet: str = Field(..., description="First 400 chars of document text")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    file_download_url: Optional[str] = Field(None, description="Download URL if result is from a file")


# ── Endpoint-specific responses ─────────────────────────────────

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]
    total: int
    elapsed_ms: float


class RagResponse(BaseModel):
    query: str
    answer: str
    sources: List[SearchResultItem]
    model_used: str
    reranked: bool = False
    elapsed_ms: float


class IndexResponse(BaseModel):
    status: str = Field(..., description="'started' or 'already_running'")
    tables_queued: int = 0
    message: str = ""


class SyncResponse(BaseModel):
    db_name: str
    table_name: str
    docs_synced: int
    message: str = ""


class DiscoverResponse(BaseModel):
    database: str
    tables_discovered: int
    tables: List[str] = Field(default_factory=list)
    message: str = ""


class DatabaseInfo(BaseModel):
    name: str
    source: str = Field(..., description="'handcrafted' or 'auto-discovered'")
    tables: List[str]
    table_count: int


class StatusResponse(BaseModel):
    databases: List[DatabaseInfo]
    index_count: int
    indexing_active: bool


class HealthResponse(BaseModel):
    status: str = Field(..., description="'healthy', 'degraded', or 'unhealthy'")
    ollama_reachable: bool
    index_loaded: bool
    doc_count: int
    database_connections: Dict[str, bool] = Field(default_factory=dict)


# ── Database Management responses ────────────────────────────────

class RegisteredDatabase(BaseModel):
    """One registered database connection."""
    name: str
    host: str
    port: int
    database: str
    connected: bool = Field(False, description="True if test connection succeeded")
    tables_configured: int = Field(0, description="Number of table configs available")
    has_schema_config: bool = Field(False, description="True if schema discovery has been run")


class DatabaseListResponse(BaseModel):
    databases: List[RegisteredDatabase]
    total: int


class DatabaseRegisterResponse(BaseModel):
    name: str
    status: str = Field(..., description="'registered', 'registered_with_discovery', 'connection_failed'")
    tables_discovered: int = 0
    tables: List[str] = Field(default_factory=list)
    message: str = ""


class DatabaseDeleteResponse(BaseModel):
    name: str
    deleted: bool
    message: str = ""


# ── Admin responses ──────────────────────────────────────────────

class AdminLoginResponse(BaseModel):
    token: str
    username: str
    role: str
    expires_in_hours: int


class AdminUserResponse(BaseModel):
    id: int
    username: str
    role: str
    is_active: bool
    created_at: Optional[str] = None


class AdminUserListResponse(BaseModel):
    users: List[AdminUserResponse]
    total: int


class TableConfigItem(BaseModel):
    table_name: str
    label: Optional[str] = None
    description: Optional[str] = None
    is_selected: bool = True
    source: str = "auto"
    pk_column: Optional[str] = None
    text_columns: List[str] = Field(default_factory=list)
    metadata_columns: List[str] = Field(default_factory=list)
    file_columns: List[List[str]] = Field(default_factory=list)
    date_column: Optional[str] = None


class TableConfigListResponse(BaseModel):
    database: str
    tables: List[TableConfigItem]
    total: int
    selected_count: int


class TableSelectionResponse(BaseModel):
    database: str
    updated: int
    message: str = ""


# ── Background Sync responses ───────────────────────────────────

class TableSyncInfo(BaseModel):
    table_key: str
    last_sync_at: Optional[str] = None
    records_synced: int = 0
    status: str = "pending"
    error_msg: Optional[str] = None
    duration_ms: float = 0.0


class LastRunInfo(BaseModel):
    completed_at: Optional[str] = None
    status: str = "never_run"
    total_records_synced: int = 0


class SyncStatusResponse(BaseModel):
    sync_enabled: bool
    scheduler_running: bool
    next_incremental_at: Optional[str] = None
    next_full_rescan_at: Optional[str] = None
    last_incremental: LastRunInfo
    last_full_rescan: LastRunInfo
    tables: List[TableSyncInfo] = Field(default_factory=list)


class SyncTriggerResponse(BaseModel):
    status: str = Field(..., description="'started' or 'already_running'")
    message: str = ""


# ── File processing responses ────────────────────────────────────

class FileProcessResultItem(BaseModel):
    table: str
    row_id: str
    file_path: str
    extracted: bool
    text_length: int = 0
    error: Optional[str] = None


class FileProcessResponse(BaseModel):
    status: str = Field(..., description="'completed' or 'error'")
    total_files: int
    extracted: int
    failed: int
    skipped: int
    elapsed_ms: float
    supported_extensions: List[str] = Field(default_factory=list)
    details: List[FileProcessResultItem] = Field(default_factory=list)

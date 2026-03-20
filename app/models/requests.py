"""
app/models/requests.py
───────────────────────
Pydantic v2 request schemas.

Each model validates and documents one API endpoint's input.
Defaults mirror sensible production values from Settings.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """POST /api/v1/search"""
    query: str = Field(..., min_length=1, max_length=1000, description="Natural-language search query")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    db_filter: Optional[str] = Field(None, description="Restrict results to a specific database name")
    table_filter: Optional[str] = Field(None, description="Restrict results to a specific table name")

    model_config = {"json_schema_extra": {"examples": [{"query": "vendor registration certificates for ammunition items", "top_k": 5}]}}


class RagAskRequest(BaseModel):
    """POST /api/v1/rag/ask"""
    query: str = Field(..., min_length=1, max_length=2000, description="Question for RAG-augmented answer")
    top_k: int = Field(5, ge=1, le=20, description="Number of source documents to retrieve")
    db_filter: Optional[str] = Field(None, description="Restrict source docs to a specific database")
    temperature: float = Field(0.1, ge=0.0, le=2.0, description="LLM sampling temperature")
    rerank: Optional[bool] = Field(None, description="Enable reranking (overrides server default)")

    model_config = {"json_schema_extra": {"examples": [{"query": "Which vendors are debarred and what are their reasons?", "top_k": 5}]}}


class IndexRequest(BaseModel):
    """POST /api/v1/index"""
    db_filter: Optional[str] = Field(None, description="Only index tables from this database")

    model_config = {"json_schema_extra": {"examples": [{"db_filter": "mydb"}]}}


class SyncRequest(BaseModel):
    """POST /api/v1/sync"""
    db_name: str = Field(..., min_length=1, description="Database name")
    table_name: str = Field(..., min_length=1, description="Table to re-index")

    model_config = {"json_schema_extra": {"examples": [{"db_name": "mydb", "table_name": "certificate_report"}]}}


class DiscoverRequest(BaseModel):
    """POST /api/v1/discover"""
    host: str = Field("127.0.0.1", description="MySQL host")
    port: int = Field(3307, ge=1, le=65535, description="MySQL port")
    user: str = Field(..., min_length=1, description="MySQL user")
    password: str = Field("", description="MySQL password")
    database: str = Field(..., min_length=1, description="Database name to discover")
    force: bool = Field(False, description="Re-discover even if config already exists")
    mode: Optional[str] = Field(
        None, pattern=r"^(heuristic|llm|auto)$",
        description="Discovery mode: 'heuristic' (instant), 'llm' (background), 'auto' (heuristic + LLM). Defaults to DISCOVERY_MODE from .env.",
    )

    model_config = {"json_schema_extra": {"examples": [{"host": "127.0.0.1", "port": 3307, "user": "root", "password": "rootpassword", "database": "mydb", "mode": "heuristic"}]}}


class DatabaseRegisterRequest(BaseModel):
    """POST /api/v1/databases"""
    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_]+$",
                      description="Logical name for this connection (alphanumeric + underscores)")
    host: str = Field("127.0.0.1", description="MySQL host")
    port: int = Field(3307, ge=1, le=65535, description="MySQL port")
    user: str = Field(..., min_length=1, description="MySQL user")
    password: str = Field("", description="MySQL password")
    database: str = Field(..., min_length=1, description="Actual MySQL database name")
    auto_discover: bool = Field(True, description="Run schema discovery after registration (heuristic by default)")
    discovery_mode: Optional[str] = Field(
        None, pattern=r"^(heuristic|llm|auto)$",
        description="Override discovery mode for this request. Defaults to DISCOVERY_MODE from .env.",
    )
    file_columns: Optional[Dict[str, List[List[str]]]] = Field(
        None,
        description="Manual override: {table_name: [[path_col, type_col], ...]}",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "name": "student_db",
                "host": "127.0.0.1",
                "port": 3307,
                "user": "root",
                "password": "rootpassword",
                "database": "student_onboarding",
                "auto_discover": True,
            }]
        }
    }


class DatabaseUpdateRequest(BaseModel):
    """PUT /api/v1/databases/{name}"""
    host: Optional[str] = Field(None, description="MySQL host")
    port: Optional[int] = Field(None, ge=1, le=65535, description="MySQL port")
    user: Optional[str] = Field(None, min_length=1, description="MySQL user")
    password: Optional[str] = Field(None, description="MySQL password")
    database: Optional[str] = Field(None, min_length=1, description="Actual MySQL database name")

    model_config = {
        "json_schema_extra": {
            "examples": [{"host": "192.168.1.10", "port": 3306}]
        }
    }


class AdminLoginRequest(BaseModel):
    """POST /api/v1/admin/login"""
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)

    model_config = {"json_schema_extra": {"examples": [{"username": "admin", "password": "OfbAdmin@2026"}]}}


class AdminCreateUserRequest(BaseModel):
    """POST /api/v1/admin/users"""
    username: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_]+$")
    password: str = Field(..., min_length=8, max_length=128)
    role: str = Field("admin", pattern=r"^(admin|superadmin)$")

    model_config = {"json_schema_extra": {"examples": [{"username": "john", "password": "SecurePass123", "role": "admin"}]}}


class TableSelectionRequest(BaseModel):
    """PUT /api/v1/databases/{name}/tables"""
    selections: Dict[str, bool] = Field(..., description="Map of table_name -> selected (true/false)")

    model_config = {"json_schema_extra": {"examples": [{"selections": {"certificate_report": True, "vendor_master": False}}]}}


class FileProcessRequest(BaseModel):
    """POST /api/v1/files/process"""
    table: str = Field("file_master", description="Table containing file references")
    db_name: str = Field("ofbdb", description="Database name")
    limit: int = Field(100, ge=1, le=10000, description="Max rows to process in this batch")
    max_workers: int = Field(4, ge=1, le=16, description="Thread pool size for file extraction")

    model_config = {"json_schema_extra": {"examples": [{"table": "file_master", "db_name": "ofbdb", "limit": 100, "max_workers": 4}]}}

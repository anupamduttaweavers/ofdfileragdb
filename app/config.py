"""
app/config.py
─────────────
Centralised application configuration via Pydantic Settings.
Loads from .env file at project root with typed defaults.

Single Responsibility: owns all config; nothing else reads os.environ directly.
Open/Closed: new LLM providers can be added by extending LLM_PROVIDER choices
without modifying existing provider logic.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Settings(BaseSettings):
    """Immutable, validated application settings."""

    # ── API Security ────────────────────────────────────────────
    api_keys: str = "ofb-dev-key-2026"

    @property
    def api_key_list(self) -> List[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    # ── LLM Provider ────────────────────────────────────────────
    llm_provider: str = "ollama"
    debug: bool = False

    @property
    def active_provider(self) -> LLMProvider:
        return LLMProvider(self.llm_provider.lower())

    # ── Ollama ──────────────────────────────────────────────────
    ollama_base_url: str = "http://192.168.0.207:8080"
    ollama_llm_model: str = "llama3.1:8b"
    ollama_embed_model: str = "nomic-embed-text:latest"
    ollama_timeout: int = 120

    # ── Embedding ───────────────────────────────────────────────
    embedding_dims: int = 768

    # ── OpenAI (alternative) ────────────────────────────────────
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── Anthropic (alternative) ─────────────────────────────────
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-sonnet-20240229"

    # ── MySQL: ofbdb ────────────────────────────────────────────
    ofbdb_host: str = "127.0.0.1"
    ofbdb_port: int = 3307
    ofbdb_user: str = "root"
    ofbdb_password: str = "rootpassword"
    ofbdb_database: str = "mydb"

    # ── MySQL: misofb ───────────────────────────────────────────
    misofb_host: str = "127.0.0.1"
    misofb_port: int = 3307
    misofb_user: str = "root"
    misofb_password: str = "rootpassword"
    misofb_database: str = "misofb"

    # ── FAISS ───────────────────────────────────────────────────
    faiss_persist_dir: str = "./faiss_index"
    faiss_collection_name: str = "ofb_documents"

    # ── Vectorizer ──────────────────────────────────────────────
    embed_batch_size: int = 16
    fetch_chunk_size: int = 500
    num_embed_workers: int = 2
    save_every_n: int = 2000

    # ── RAG ─────────────────────────────────────────────────────
    rag_top_k: int = 5
    rag_max_tokens: int = 2048
    rag_temperature: float = 0.1
    rag_rerank_enabled: bool = True

    # ── Sync ────────────────────────────────────────────────────
    sync_enabled: bool = True
    sync_interval_seconds: int = 120
    sync_full_rescan_hour: int = 2
    sync_state_file: str = "./sync_state.json"

    # ── Admin / Auth ─────────────────────────────────────────────
    super_admin_username: str = "admin"
    super_admin_password: str = "OfbAdmin@2026"
    jwt_secret: str = "ofb-jwt-secret-change-in-production"
    jwt_expiry_hours: int = 8
    encryption_key: str = ""
    sqlite_db_path: str = "./data/ofb_config.db"

    # ── Server ──────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    cors_origins: str = "*"

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    model_config = {
        "env_file": str(ENV_FILE),
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor — cached after first call."""
    return Settings()

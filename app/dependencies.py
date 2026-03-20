"""
app/dependencies.py
────────────────────
FastAPI dependency injection layer.

Uses module-level singletons set during lifespan.
Follows Dependency Inversion: routes depend on abstractions.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, Header, Query
from fastapi.security import APIKeyHeader

from app.config import Settings, get_settings
from app.exceptions import AuthenticationError, IndexNotReadyError

log = logging.getLogger("app.dependencies")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_search_engine = None
_vectorizer_config = None


def set_search_engine(engine) -> None:
    global _search_engine
    _search_engine = engine


def set_vectorizer_config(config) -> None:
    global _vectorizer_config
    _vectorizer_config = config


async def require_api_key(
    api_key: Optional[str] = Depends(_api_key_header),
    api_key_query: Optional[str] = Query(None, alias="api_key"),
    settings: Settings = Depends(get_settings),
) -> str:
    key = api_key or api_key_query
    if not key or key not in settings.api_key_list:
        raise AuthenticationError()
    return key


async def require_admin_or_api_key(
    api_key: Optional[str] = Depends(_api_key_header),
    authorization: Optional[str] = Header(None),
    settings: Settings = Depends(get_settings),
) -> str:
    """Accept either a valid API key or a valid JWT Bearer token."""
    if api_key and api_key in settings.api_key_list:
        return api_key

    if authorization and authorization.startswith("Bearer "):
        from app.core.admin_auth import verify_jwt
        token = authorization[7:]
        payload = verify_jwt(token, settings)
        if payload is not None:
            return f"jwt:{payload['sub']}"

    raise AuthenticationError("Valid API key or JWT token required.")


def get_search_engine():
    if _search_engine is None:
        raise IndexNotReadyError("Search engine has not been initialised yet.")
    return _search_engine


def get_vectorizer_config():
    if _vectorizer_config is None:
        raise IndexNotReadyError("Vectorizer configuration has not been initialised yet.")
    return _vectorizer_config


def build_vectorizer_config(settings: Settings):
    from app.core.vectorizer import VectorizerConfig, DBConnectionConfig
    from app.core.connection_store import get_connection_store

    store = get_connection_store()
    store.seed_defaults(settings)

    db_connections = {}
    for name, cred in store.load_all().items():
        db_connections[name] = DBConnectionConfig(
            host=cred.host,
            port=cred.port,
            user=cred.user,
            password=cred.password,
            database=cred.database,
        )

    return VectorizerConfig(
        db_connections=db_connections,
        faiss_persist_dir=settings.faiss_persist_dir,
        collection_name=settings.faiss_collection_name,
        embed_batch_size=settings.embed_batch_size,
        fetch_chunk_size=settings.fetch_chunk_size,
        num_embed_workers=settings.num_embed_workers,
        save_every_n=settings.save_every_n,
    )

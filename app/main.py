"""
app/main.py
────────────
FastAPI application factory.

Responsibilities:
  - Lifespan management (startup/shutdown)
  - Middleware stack (CORS, request logging)
  - Global exception handling (maps AppError hierarchy -> JSON)
  - Route registration

Run:
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.exceptions import AppError

log = logging.getLogger("app")


# ── Lifespan ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = get_settings()

    _configure_logging(settings.log_level)
    log.info("Starting OFB Document Discovery API...")

    from app.core.config_db import init_db, seed_defaults, migrate_from_json
    init_db(settings)
    seed_defaults(settings)

    migrated = migrate_from_json("./configs/connections.json")
    if migrated:
        log.info("Migrated %d connections from legacy JSON to SQLite.", migrated)

    _seed_handcrafted_table_configs()

    from app.dependencies import (
        build_vectorizer_config,
        set_search_engine,
        set_vectorizer_config,
    )

    vec_config = build_vectorizer_config(settings)
    set_vectorizer_config(vec_config)
    log.info("Vectorizer config built for databases: %s", list(vec_config.db_connections.keys()))

    search_engine = None
    try:
        from app.core.search_engine import DocumentSearchEngine

        search_engine = DocumentSearchEngine(
            faiss_persist_dir=settings.faiss_persist_dir,
            collection_name=settings.faiss_collection_name,
        )
        set_search_engine(search_engine)
        log.info("Search engine initialised with %d documents.", search_engine.index_count())
    except Exception as exc:
        log.warning("Search engine failed to load (index may not exist yet): %s", exc)

    scheduler = None
    if settings.sync_enabled:
        from app.services.scheduler import start_sync_scheduler
        scheduler = start_sync_scheduler(settings, vec_config, search_engine)

    log.info("API ready at http://%s:%s", settings.app_host, settings.app_port)

    yield

    if scheduler is not None and scheduler.running:
        scheduler.shutdown(wait=False)
        log.info("Background sync scheduler stopped.")

    log.info("Shutting down OFB Document Discovery API.")


# ── App factory ─────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    application = FastAPI(
        title="OFB Document Discovery API",
        description=(
            "Production-grade RAG and Vector Search API for the "
            "Ordnance Factory Board document ecosystem. "
            "LangChain/LangGraph-powered with FAISS, FlashRank reranking, "
            "and file vectorization with downloadable links."
        ),
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    _register_middleware(application, settings)
    _register_exception_handlers(application)
    _register_routes(application)

    return application


# ── Middleware ───────────────────────────────────────────────────

def _register_middleware(application: FastAPI, settings) -> None:
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        log.info(
            "%s %s -> %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response


# ── Exception Handlers ──────────────────────────────────────────

def _register_exception_handlers(application: FastAPI) -> None:
    @application.exception_handler(AppError)
    async def app_error_handler(_request: Request, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error_code": exc.error_code,
                "detail": exc.detail,
                "context": exc.context,
            },
        )

    @application.exception_handler(Exception)
    async def unhandled_error_handler(_request: Request, exc: Exception):
        log.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "INTERNAL_ERROR",
                "detail": "An unexpected internal error occurred.",
                "context": {},
            },
        )


# ── Routes ──────────────────────────────────────────────────────

def _register_routes(application: FastAPI) -> None:
    from app.routes.health import router as health_router
    from app.routes.search import router as search_router
    from app.routes.rag import router as rag_router
    from app.routes.index import router as index_router
    from app.routes.discover import router as discover_router
    from app.routes.status import router as status_router
    from app.routes.sync_status import router as sync_router
    from app.routes.files import router as files_router
    from app.routes.databases import router as databases_router
    from app.routes.admin import router as admin_router

    application.include_router(health_router)
    application.include_router(search_router)
    application.include_router(rag_router)
    application.include_router(index_router)
    application.include_router(discover_router)
    application.include_router(status_router)
    application.include_router(sync_router)
    application.include_router(files_router)
    application.include_router(databases_router)
    application.include_router(admin_router)


def _seed_handcrafted_table_configs() -> None:
    """Seed handcrafted OFBDB/MISOFB table configs into SQLite on first run."""
    from app.core.config_db import upsert_table_config, get_table_configs_for_db, is_initialized
    if not is_initialized():
        return

    from app.core.schema_config import OFBDB_CONFIGS, MISOFB_CONFIGS

    for cfgs, db_name in [(OFBDB_CONFIGS, "ofbdb"), (MISOFB_CONFIGS, "misofb")]:
        existing = get_table_configs_for_db(db_name)
        existing_tables = {c["table_name"] for c in existing}
        for cfg in cfgs:
            if cfg.table in existing_tables:
                continue
            upsert_table_config(
                db_name=db_name,
                table_name=cfg.table,
                text_columns=cfg.text_columns,
                metadata_columns=cfg.metadata_columns,
                pk_column=cfg.pk_column,
                label=cfg.label,
                description=cfg.description,
                date_column=cfg.date_column,
                file_columns=list(cfg.file_columns),
                source="handcrafted",
            )


# ── Logging config ──────────────────────────────────────────────

def _configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


# ── Module-level app instance (uvicorn entry point) ─────────────

app = create_app()

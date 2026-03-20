"""
app/routes/files.py
────────────────────
File processing and download endpoints.

POST /api/v1/files/process   -- trigger file text extraction and vectorization
GET  /api/v1/files/supported -- list supported file extensions
GET  /api/v1/files/download/{doc_id} -- download original file by doc_id
"""

from __future__ import annotations

import logging
import os
from typing import List

import mysql.connector
from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.config import Settings, get_settings
from app.core.database import get_db_connection
from app.dependencies import require_api_key, get_search_engine
from app.exceptions import DatabaseConnectionError, ResourceNotFoundError
from app.models.requests import FileProcessRequest
from app.models.responses import (
    FileProcessResponse,
    FileProcessResultItem,
)
from app.services.file_extractor import supported_extensions, resolve_file_path
from app.services.file_processor import FileProcessor, has_file_columns

log = logging.getLogger("app.routes.files")

router = APIRouter(prefix="/api/v1/files", tags=["files"], dependencies=[Depends(require_api_key)])


@router.post(
    "/process",
    response_model=FileProcessResponse,
    summary="Process files from DB table",
    description="Extract text from vendor-uploaded files referenced in the database. "
                "Uses multithreaded I/O for parallel file extraction.",
)
def process_files(
    body: FileProcessRequest,
    settings: Settings = Depends(get_settings),
):
    try:
        conn = get_db_connection(body.db_name, settings)
    except Exception as exc:
        raise DatabaseConnectionError(body.db_name, str(exc))

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(f"SELECT * FROM `{body.table}` LIMIT %s", (body.limit,))
        rows = cursor.fetchall()
        cursor.close()
    except Exception as exc:
        conn.close()
        raise DatabaseConnectionError(body.db_name, f"Query failed: {exc}")

    conn.close()

    if not rows:
        return FileProcessResponse(
            status="completed",
            total_files=0, extracted=0, failed=0, skipped=0,
            elapsed_ms=0.0,
            supported_extensions=supported_extensions(),
        )

    if not has_file_columns(body.table):
        return FileProcessResponse(
            status="completed",
            total_files=0, extracted=0, failed=0, skipped=0,
            elapsed_ms=0.0,
            supported_extensions=supported_extensions(),
            details=[],
        )

    processor = FileProcessor(max_workers=body.max_workers)
    result = processor.process_rows(body.table, rows, pk_column="id")

    details = [
        FileProcessResultItem(
            table=d.table,
            row_id=d.row_id,
            file_path=d.file_path,
            extracted=d.extracted,
            text_length=d.text_length,
            error=d.error,
        )
        for d in result.details
    ]

    return FileProcessResponse(
        status="completed",
        total_files=result.total_files,
        extracted=result.extracted,
        failed=result.failed,
        skipped=result.skipped,
        elapsed_ms=result.elapsed_ms,
        supported_extensions=supported_extensions(),
        details=details,
    )


@router.get(
    "/supported",
    summary="List supported file types",
    description="Returns the list of file extensions that can be extracted.",
)
def list_supported():
    return {"supported_extensions": supported_extensions()}


@router.get(
    "/download/{doc_id:path}",
    summary="Download original file",
    description="Serves the original file referenced by a doc_id from the vector store. "
                "Works for any indexed document that has a file_path in metadata.",
)
def download_file(doc_id: str):
    engine = get_search_engine()
    store = engine.store

    file_path = store.find_file_path_for_doc(doc_id)

    if not file_path and ".chunk_" in doc_id:
        base_id = doc_id.rsplit(".chunk_", 1)[0]
        file_path = store.find_file_path_for_doc(base_id)

    if not file_path:
        raise ResourceNotFoundError("File path", doc_id)

    resolved = resolve_file_path(file_path)
    if resolved is None or not os.path.isfile(resolved):
        raise ResourceNotFoundError("File on disk", file_path)

    meta = store.get_metadata_by_doc_id(doc_id) or {}
    filename = meta.get("file_name", os.path.basename(resolved))

    return FileResponse(
        path=resolved,
        filename=filename,
        media_type="application/octet-stream",
    )

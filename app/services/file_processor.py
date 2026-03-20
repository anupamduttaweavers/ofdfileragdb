"""
app/services/file_processor.py
───────────────────────────────
Multi-threaded file processing pipeline.

Uses ThreadPoolExecutor for I/O-bound file reads and text extraction,
and ProcessPoolExecutor for CPU-bound PDF parsing when the workload
is large enough to justify the overhead.

Single Responsibility: orchestrates parallel file text extraction and vectorization.
Interface Segregation: exposes only process_files() and process_table_files().
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.services.file_extractor import extract_text_from_file, resolve_file_path

log = logging.getLogger("app.services.file_processor")

_LEGACY_FILE_COLUMNS = {
    "file_master": [("file_path", "file_type")],
    "advertisement_items": [("item_file", ""), ("item_specification_file", "")],
    "addendum_items": [("item_file", ""), ("item_specification_file", "")],
    "renewal_items": [("item_file", ""), ("item_specification_file", "")],
    "item_master": [("item_file", ""), ("specification_files", "")],
    "item_master_diff": [("item_file", ""), ("specification_files", "")],
    "vendor_clarification": [("attach_file", "")],
    "vendor_debar": [("debar_file", "")],
}


@dataclass
class FileProcessResult:
    """Result of processing a single file reference."""
    table: str
    row_id: str
    file_path: str
    extracted: bool
    text_length: int
    error: Optional[str] = None


@dataclass
class BatchProcessResult:
    """Aggregate result of a batch file processing run."""
    total_files: int
    extracted: int
    failed: int
    skipped: int
    elapsed_ms: float
    details: List[FileProcessResult]


def get_file_columns_for_table(table_name: str, db_name: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Return list of (path_column, type_column) pairs for tables with file references.

    Resolution order:
      1. Dynamic lookup from TableConfig.file_columns (covers discovered + handcrafted configs)
      2. Legacy fallback dict for backward compatibility
    """
    try:
        from app.core.schema_config import get_all_configs
        for cfg in get_all_configs():
            if cfg.table == table_name and (db_name is None or cfg.db == db_name):
                if cfg.file_columns:
                    return list(cfg.file_columns)
    except Exception:
        pass

    return _LEGACY_FILE_COLUMNS.get(table_name, [])


def has_file_columns(table_name: str, db_name: Optional[str] = None) -> bool:
    return len(get_file_columns_for_table(table_name, db_name)) > 0


class FileProcessor:
    """
    Parallel file extraction and vectorization pipeline.

    For each row from a file-bearing table, resolves the file path,
    extracts text, and returns it for embedding.
    """

    def __init__(self, max_workers: int = 4):
        self._max_workers = max_workers

    def process_rows(
        self,
        table_name: str,
        rows: List[Dict[str, Any]],
        pk_column: Optional[str] = None,
    ) -> BatchProcessResult:
        """
        Extract text from files referenced in the given rows.

        Uses ThreadPoolExecutor for concurrent I/O-bound file reads.
        Returns a BatchProcessResult with per-file details.
        """
        start = time.perf_counter()
        file_cols = get_file_columns_for_table(table_name)
        if not file_cols:
            return BatchProcessResult(
                total_files=0, extracted=0, failed=0, skipped=0,
                elapsed_ms=0, details=[],
            )

        tasks: List[Tuple[str, str, str]] = []
        for row in rows:
            row_id = str(row.get(pk_column, "")) if pk_column else ""
            for path_col, type_col in file_cols:
                raw_path = row.get(path_col)
                if not raw_path or not str(raw_path).strip():
                    continue
                file_type = str(row.get(type_col, "")) if type_col else ""
                tasks.append((row_id, str(raw_path), file_type))

        if not tasks:
            elapsed = (time.perf_counter() - start) * 1000
            return BatchProcessResult(
                total_files=0, extracted=0, failed=0, skipped=0,
                elapsed_ms=round(elapsed, 2), details=[],
            )

        results: List[FileProcessResult] = []
        extracted_count = 0
        failed_count = 0
        skipped_count = 0

        worker_count = min(self._max_workers, len(tasks))

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(self._process_one, table_name, row_id, raw_path, file_type): (row_id, raw_path)
                for row_id, raw_path, file_type in tasks
            }

            for future in as_completed(future_map):
                row_id, raw_path = future_map[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result.extracted:
                        extracted_count += 1
                    elif result.error:
                        failed_count += 1
                    else:
                        skipped_count += 1
                except Exception as exc:
                    failed_count += 1
                    results.append(FileProcessResult(
                        table=table_name,
                        row_id=row_id,
                        file_path=raw_path,
                        extracted=False,
                        text_length=0,
                        error=str(exc),
                    ))

        elapsed = (time.perf_counter() - start) * 1000
        return BatchProcessResult(
            total_files=len(tasks),
            extracted=extracted_count,
            failed=failed_count,
            skipped=skipped_count,
            elapsed_ms=round(elapsed, 2),
            details=results,
        )

    def extract_file_text(self, file_path: str, file_type: str = "") -> Optional[str]:
        """Extract text from a single file. Returns None if not extractable."""
        return extract_text_from_file(file_path, file_type=file_type)

    @staticmethod
    def _process_one(
        table_name: str,
        row_id: str,
        raw_path: str,
        file_type: str,
    ) -> FileProcessResult:
        """Process a single file reference — called within thread pool."""
        resolved = resolve_file_path(raw_path)
        if resolved is None:
            return FileProcessResult(
                table=table_name,
                row_id=row_id,
                file_path=raw_path,
                extracted=False,
                text_length=0,
            )

        try:
            text = extract_text_from_file(resolved, file_type=file_type)
            if text:
                return FileProcessResult(
                    table=table_name,
                    row_id=row_id,
                    file_path=raw_path,
                    extracted=True,
                    text_length=len(text),
                )
            return FileProcessResult(
                table=table_name,
                row_id=row_id,
                file_path=raw_path,
                extracted=False,
                text_length=0,
            )
        except Exception as exc:
            return FileProcessResult(
                table=table_name,
                row_id=row_id,
                file_path=raw_path,
                extracted=False,
                text_length=0,
                error=str(exc),
            )

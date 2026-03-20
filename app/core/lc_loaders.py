"""
app.core.lc_loaders
────────────────────
Unified document loader using LangChain loaders.

Supports PDF, DOCX, TXT, CSV, XLSX, HTML.
Falls back to custom file_extractor when LangChain loader is unavailable.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from langchain_core.documents import Document

from app.services.file_extractor import resolve_file_path, get_file_extension

log = logging.getLogger("app.core.lc_loaders")

_LOADER_MAP = {}


def _register_loaders():
    """Lazy-register loaders to avoid import errors if packages are missing."""
    global _LOADER_MAP
    if _LOADER_MAP:
        return

    try:
        from langchain_community.document_loaders import PyPDFLoader
        _LOADER_MAP["pdf"] = PyPDFLoader
    except ImportError:
        pass

    try:
        from langchain_community.document_loaders import Docx2txtLoader
        _LOADER_MAP["docx"] = Docx2txtLoader
    except ImportError:
        pass

    try:
        from langchain_community.document_loaders import TextLoader
        _LOADER_MAP["txt"] = TextLoader
        _LOADER_MAP["text"] = TextLoader
        _LOADER_MAP["md"] = TextLoader
        _LOADER_MAP["rst"] = TextLoader
        _LOADER_MAP["log"] = TextLoader
    except ImportError:
        pass

    try:
        from langchain_community.document_loaders import CSVLoader
        _LOADER_MAP["csv"] = CSVLoader
    except ImportError:
        pass


def load_file(
    file_path: str,
    file_type: str = "",
    source_metadata: Optional[dict] = None,
) -> List[Document]:
    """
    Load a file and return LangChain Documents.

    Resolves potentially encoded paths, dispatches to the appropriate
    LangChain loader, and falls back to custom extraction.
    """
    _register_loaders()

    resolved = resolve_file_path(file_path)
    if resolved is None:
        log.debug("Cannot resolve file path: %s", file_path[:100])
        return []

    ext = get_file_extension(resolved, file_type)
    extra_meta = source_metadata or {}

    loader_cls = _LOADER_MAP.get(ext)
    if loader_cls is not None:
        try:
            if ext == "csv":
                loader = loader_cls(resolved, encoding="utf-8")
            elif ext in ("txt", "text", "md", "rst", "log"):
                loader = loader_cls(resolved, encoding="utf-8", autodetect_encoding=True)
            else:
                loader = loader_cls(resolved)

            docs = loader.load()
            for doc in docs:
                doc.metadata.update(extra_meta)
                doc.metadata["source_file"] = resolved
                doc.metadata["file_extension"] = ext
            return docs
        except Exception as exc:
            log.warning("LangChain loader failed for %s, falling back: %s", resolved, exc)

    from app.services.file_extractor import extract_text_from_file
    text = extract_text_from_file(resolved, file_type=file_type)
    if text:
        return [Document(
            page_content=text,
            metadata={
                **extra_meta,
                "source_file": resolved,
                "file_extension": ext,
            },
        )]

    return []


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Split documents into smaller chunks for better retrieval."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)

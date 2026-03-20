"""
app.core.interfaces
─────────────────────
Abstract base classes following Dependency Inversion Principle.

All concrete implementations depend on these abstractions,
not the other way around.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class BaseEmbedder(ABC):
    """Interface for embedding providers (Ollama, OpenAI, etc.)."""

    @abstractmethod
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        ...

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...


class BaseLLMProvider(ABC):
    """Interface for LLM generation providers."""

    @abstractmethod
    def generate(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...


class BaseVectorStore(ABC):
    """Interface for vector store backends (FAISS, ChromaDB, etc.)."""

    @abstractmethod
    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        ...

    @abstractmethod
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def save(self) -> None:
        ...

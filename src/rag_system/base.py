"""Base classes for RAG system components."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """Represents a document with content and metadata."""

    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None

    def __str__(self) -> str:
        """String representation."""
        return f"Document(id={self.doc_id}, content_length={len(self.content)})"


class BaseExtractor(ABC):
    """Base class for document extractors."""

    @abstractmethod
    def extract(self, file_path: str) -> List[Document]:
        """Extract documents from a file.

        Args:
            file_path: Path to the file

        Returns:
            List of extracted documents
        """
        pass

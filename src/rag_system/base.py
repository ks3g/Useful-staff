"""Base classes and interfaces for the RAG system."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Document:
    """Represents a document chunk with metadata."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if 'timestamp' not in self.metadata:
            self.metadata['timestamp'] = datetime.now().isoformat()


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""

    document: Document
    score: float
    rank: int


@dataclass
class RAGResponse:
    """Response from RAG query."""

    query: str
    answer: str
    source_documents: List[RetrievalResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if 'timestamp' not in self.metadata:
            self.metadata['timestamp'] = datetime.now().isoformat()


class BaseExtractor(ABC):
    """Abstract base class for document extractors."""

    @abstractmethod
    def extract(self, file_path: str) -> List[Document]:
        """Extract documents from file.

        Args:
            file_path: Path to the file to extract

        Returns:
            List of extracted documents
        """
        pass


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""

    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        """Chunk documents into smaller pieces.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents
        """
        pass


class BaseEmbedding(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        pass


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of documents with embeddings
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filters

        Returns:
            List of retrieval results
        """
        pass

    @abstractmethod
    def persist(self) -> None:
        """Persist the vector store to disk."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load the vector store from disk."""
        pass


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query.

        Args:
            query: Query string
            top_k: Number of results to return
            **kwargs: Additional retriever-specific arguments

        Returns:
            List of retrieval results
        """
        pass


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional LLM-specific arguments

        Returns:
            Generated text
        """
        pass


class BaseEvaluator(ABC):
    """Abstract base class for evaluation metrics."""

    @abstractmethod
    def evaluate(
        self,
        predictions: List[Any],
        references: List[Any],
        **kwargs
    ) -> Dict[str, float]:
        """Evaluate predictions against references.

        Args:
            predictions: List of predictions
            references: List of reference/ground truth
            **kwargs: Additional evaluator-specific arguments

        Returns:
            Dictionary of metric scores
        """
        pass

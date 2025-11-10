"""FAISS-based vector store implementation."""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..base import BaseVectorStore, Document, RetrievalResult
from ...utils.logger import get_logger

logger = get_logger()


class FAISSVectorStore(BaseVectorStore):
    """Vector store using FAISS for similarity search."""

    def __init__(
        self,
        embedding_dimension: int,
        index_type: str = "IndexFlatL2",
        persist_directory: Optional[str] = None
    ):
        """Initialize FAISS vector store.

        Args:
            embedding_dimension: Dimension of embeddings
            index_type: Type of FAISS index (IndexFlatL2, IndexFlatIP, IndexIVFFlat)
            persist_directory: Directory to persist the index
        """
        self.embedding_dimension = embedding_dimension
        self.index_type = index_type
        self.persist_directory = persist_directory

        # Initialize FAISS index
        self.index = self._create_index()

        # Store documents and metadata separately
        self.documents: List[Document] = []
        self.doc_ids: List[str] = []

        logger.info(f"Initialized FAISS vector store with {index_type}")

    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on type.

        Returns:
            FAISS index
        """
        if self.index_type == "IndexFlatL2":
            return faiss.IndexFlatL2(self.embedding_dimension)
        elif self.index_type == "IndexFlatIP":
            return faiss.IndexFlatIP(self.embedding_dimension)
        elif self.index_type == "IndexIVFFlat":
            # For IVF, we need a quantizer
            quantizer = faiss.IndexFlatL2(self.embedding_dimension)
            nlist = 100  # number of clusters
            return faiss.IndexIVFFlat(quantizer, self.embedding_dimension, nlist)
        else:
            logger.warning(f"Unknown index type {self.index_type}, using IndexFlatL2")
            return faiss.IndexFlatL2(self.embedding_dimension)

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of documents with embeddings
        """
        if not documents:
            logger.warning("No documents to add")
            return

        # Extract embeddings
        embeddings = []
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.doc_id} has no embedding")
            embeddings.append(doc.embedding)

        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')

        # Normalize for cosine similarity (if using IndexFlatIP)
        if self.index_type == "IndexFlatIP":
            faiss.normalize_L2(embeddings_array)

        # Train index if needed (for IVF)
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings_array)

        # Add to index
        self.index.add(embeddings_array)

        # Store documents
        self.documents.extend(documents)
        self.doc_ids.extend([doc.doc_id or f"doc_{i}" for i, doc in enumerate(documents)])

        logger.info(f"Added {len(documents)} documents to vector store. Total: {len(self.documents)}")

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
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []

        # Convert query to numpy array
        query_array = np.array([query_embedding]).astype('float32')

        # Normalize for cosine similarity
        if self.index_type == "IndexFlatIP":
            faiss.normalize_L2(query_array)

        # Search
        distances, indices = self.index.search(query_array, min(top_k, self.index.ntotal))

        # Convert to retrieval results
        results = []
        for rank, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.documents):  # Valid index
                doc = self.documents[idx]

                # Apply metadata filters if provided
                if filter_dict:
                    if not self._matches_filter(doc, filter_dict):
                        continue

                # Convert distance to similarity score
                # For L2 distance, smaller is better; convert to similarity
                if self.index_type == "IndexFlatL2":
                    score = 1.0 / (1.0 + distance)
                else:  # IndexFlatIP (cosine similarity)
                    score = float(distance)

                results.append(RetrievalResult(
                    document=doc,
                    score=score,
                    rank=rank
                ))

        logger.debug(f"Found {len(results)} results")
        return results[:top_k]

    def _matches_filter(self, doc: Document, filter_dict: Dict[str, Any]) -> bool:
        """Check if document matches metadata filters.

        Args:
            doc: Document to check
            filter_dict: Filter dictionary

        Returns:
            True if matches, False otherwise
        """
        for key, value in filter_dict.items():
            if key not in doc.metadata or doc.metadata[key] != value:
                return False
        return True

    def persist(self) -> None:
        """Persist the vector store to disk."""
        if not self.persist_directory:
            logger.warning("No persist directory specified")
            return

        persist_path = Path(self.persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = persist_path / "faiss.index"
        faiss.write_index(self.index, str(index_path))

        # Save documents and metadata
        metadata_path = persist_path / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'doc_ids': self.doc_ids,
                'embedding_dimension': self.embedding_dimension,
                'index_type': self.index_type
            }, f)

        logger.info(f"Persisted vector store to {self.persist_directory}")

    def load(self) -> None:
        """Load the vector store from disk."""
        if not self.persist_directory:
            raise ValueError("No persist directory specified")

        persist_path = Path(self.persist_directory)
        if not persist_path.exists():
            raise FileNotFoundError(f"Persist directory not found: {self.persist_directory}")

        # Load FAISS index
        index_path = persist_path / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        self.index = faiss.read_index(str(index_path))

        # Load documents and metadata
        metadata_path = persist_path / "metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.doc_ids = data['doc_ids']
            self.embedding_dimension = data['embedding_dimension']
            self.index_type = data['index_type']

        logger.info(f"Loaded vector store from {self.persist_directory} with {len(self.documents)} documents")

    def clear(self) -> None:
        """Clear the vector store."""
        self.index = self._create_index()
        self.documents = []
        self.doc_ids = []
        logger.info("Cleared vector store")

    def get_document_count(self) -> int:
        """Get the number of documents in the store.

        Returns:
            Document count
        """
        return len(self.documents)

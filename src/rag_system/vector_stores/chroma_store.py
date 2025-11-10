"""ChromaDB-based vector store implementation."""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..base import BaseVectorStore, Document, RetrievalResult
from ...utils.logger import get_logger

logger = get_logger()


class ChromaVectorStore(BaseVectorStore):
    """Vector store using ChromaDB."""

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: Optional[str] = None
    ):
        """Initialize Chroma vector store.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize ChromaDB client
        if persist_directory:
            persist_path = Path(persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=str(persist_path)
            )
        else:
            self.client = chromadb.Client()

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"Initialized Chroma vector store with collection: {collection_name}")

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of documents with embeddings
        """
        if not documents:
            logger.warning("No documents to add")
            return

        # Prepare data for Chroma
        ids = []
        embeddings = []
        metadatas = []
        documents_text = []

        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.doc_id} has no embedding")

            doc_id = doc.doc_id or f"doc_{len(ids)}"
            ids.append(doc_id)
            embeddings.append(doc.embedding)
            metadatas.append(doc.metadata)
            documents_text.append(doc.content)

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents_text
        )

        logger.info(f"Added {len(documents)} documents to Chroma vector store")

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
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict  # Chroma supports metadata filtering
        )

        # Convert to retrieval results
        retrieval_results = []

        if results['ids'] and len(results['ids']) > 0:
            for rank, (doc_id, distance, metadata, content) in enumerate(zip(
                results['ids'][0],
                results['distances'][0],
                results['metadatas'][0],
                results['documents'][0]
            )):
                # Create Document object
                doc = Document(
                    content=content,
                    metadata=metadata,
                    doc_id=doc_id
                )

                # Chroma returns distance; convert to similarity
                # Cosine distance: 0 = identical, 2 = opposite
                similarity_score = 1.0 - (distance / 2.0)

                retrieval_results.append(RetrievalResult(
                    document=doc,
                    score=similarity_score,
                    rank=rank
                ))

        logger.debug(f"Found {len(retrieval_results)} results")
        return retrieval_results

    def persist(self) -> None:
        """Persist the vector store to disk.

        Note: With PersistentClient, data is automatically persisted.
        """
        if self.persist_directory:
            logger.info(f"Chroma data persisted to {self.persist_directory}")
        else:
            logger.warning("Using in-memory Chroma, data will not persist")

    def load(self) -> None:
        """Load the vector store from disk.

        Note: With PersistentClient, data is automatically loaded on init.
        """
        count = self.collection.count()
        logger.info(f"Loaded Chroma collection with {count} documents")

    def clear(self) -> None:
        """Clear the vector store."""
        # Delete the collection and recreate it
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Cleared Chroma vector store")

    def get_document_count(self) -> int:
        """Get the number of documents in the store.

        Returns:
            Document count
        """
        return self.collection.count()

    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Get a document by its ID.

        Args:
            doc_id: Document ID

        Returns:
            Document if found, None otherwise
        """
        try:
            result = self.collection.get(
                ids=[doc_id],
                include=["metadatas", "documents"]
            )

            if result['ids']:
                return Document(
                    content=result['documents'][0],
                    metadata=result['metadatas'][0],
                    doc_id=result['ids'][0]
                )
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {e}")

        return None

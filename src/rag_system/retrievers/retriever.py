"""Retrieval implementations for RAG system."""

from typing import List, Optional, Dict, Any
from sentence_transformers import CrossEncoder
from ..base import BaseRetriever, RetrievalResult, BaseVectorStore, BaseEmbedding
from ...utils.logger import get_logger

logger = get_logger()


class VectorRetriever(BaseRetriever):
    """Simple vector similarity-based retriever."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_model: BaseEmbedding,
        score_threshold: float = 0.0
    ):
        """Initialize vector retriever.

        Args:
            vector_store: Vector store instance
            embedding_model: Embedding model instance
            score_threshold: Minimum similarity score threshold
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query.

        Args:
            query: Query string
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            **kwargs: Additional arguments

        Returns:
            List of retrieval results
        """
        logger.debug(f"Retrieving documents for query: {query[:50]}...")

        # Embed the query
        query_embedding = self.embedding_model.embed_query(query)

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict
        )

        # Filter by score threshold
        filtered_results = [
            result for result in results
            if result.score >= self.score_threshold
        ]

        logger.info(f"Retrieved {len(filtered_results)} documents (threshold: {self.score_threshold})")
        return filtered_results


class HybridRetriever(BaseRetriever):
    """Hybrid retriever with vector search + reranking."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_model: BaseEmbedding,
        reranker_model: Optional[str] = None,
        score_threshold: float = 0.0,
        initial_top_k_multiplier: int = 3
    ):
        """Initialize hybrid retriever.

        Args:
            vector_store: Vector store instance
            embedding_model: Embedding model instance
            reranker_model: Name of cross-encoder model for reranking
            score_threshold: Minimum similarity score threshold
            initial_top_k_multiplier: Multiply top_k for initial retrieval before reranking
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.score_threshold = score_threshold
        self.initial_top_k_multiplier = initial_top_k_multiplier

        # Initialize reranker if model name provided
        self.reranker = None
        if reranker_model:
            logger.info(f"Loading reranker model: {reranker_model}")
            self.reranker = CrossEncoder(reranker_model)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        use_reranking: bool = True,
        **kwargs
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents with optional reranking.

        Args:
            query: Query string
            top_k: Number of final results to return
            filter_dict: Optional metadata filters
            use_reranking: Whether to use reranking
            **kwargs: Additional arguments

        Returns:
            List of retrieval results
        """
        logger.debug(f"Retrieving documents with hybrid approach for: {query[:50]}...")

        # Initial retrieval with higher top_k
        initial_top_k = top_k * self.initial_top_k_multiplier

        # Embed the query
        query_embedding = self.embedding_model.embed_query(query)

        # Search vector store
        initial_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=initial_top_k,
            filter_dict=filter_dict
        )

        logger.debug(f"Initial retrieval: {len(initial_results)} documents")

        # Apply reranking if enabled and reranker is available
        if use_reranking and self.reranker and initial_results:
            logger.debug("Applying reranking...")
            results = self._rerank(query, initial_results)
        else:
            results = initial_results

        # Filter by score threshold and limit to top_k
        filtered_results = [
            result for result in results
            if result.score >= self.score_threshold
        ][:top_k]

        logger.info(f"Retrieved {len(filtered_results)} documents after hybrid retrieval")
        return filtered_results

    def _rerank(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Rerank results using cross-encoder.

        Args:
            query: Query string
            results: Initial retrieval results

        Returns:
            Reranked results
        """
        if not results:
            return results

        # Prepare query-document pairs
        pairs = [[query, result.document.content] for result in results]

        # Get reranking scores
        rerank_scores = self.reranker.predict(pairs)

        # Update scores and re-sort
        reranked_results = []
        for i, result in enumerate(results):
            # Create new result with reranked score
            reranked_result = RetrievalResult(
                document=result.document,
                score=float(rerank_scores[i]),
                rank=i
            )
            reranked_results.append(reranked_result)

        # Sort by new scores
        reranked_results.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for i, result in enumerate(reranked_results):
            result.rank = i

        logger.debug(f"Reranked {len(reranked_results)} results")
        return reranked_results


class BM25Retriever(BaseRetriever):
    """BM25-based retriever for keyword search.

    Note: This is a simplified implementation. For production use,
    consider using a dedicated BM25 library like rank_bm25.
    """

    def __init__(
        self,
        documents: List,
        k1: float = 1.5,
        b: float = 0.75
    ):
        """Initialize BM25 retriever.

        Args:
            documents: List of documents
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        self.documents = documents
        self.k1 = k1
        self.b = b
        logger.info("Initialized BM25 retriever (simplified implementation)")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """Retrieve documents using BM25.

        Args:
            query: Query string
            top_k: Number of results to return
            **kwargs: Additional arguments

        Returns:
            List of retrieval results
        """
        # This is a placeholder for BM25 implementation
        # For production, use rank_bm25 or similar library
        logger.warning("BM25Retriever is a placeholder implementation")
        return []

"""Embedding models for converting text to vectors."""

import os
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import openai
from ..base import BaseEmbedding
from ...utils.logger import get_logger

logger = get_logger()


class SentenceTransformerEmbedding(BaseEmbedding):
    """Embedding using Sentence Transformers (local models)."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize: bool = True,
        device: Optional[str] = None
    ):
        """Initialize Sentence Transformer embedding.

        Args:
            model_name: Name of the sentence transformer model
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings
            device: Device to use (cuda/cpu)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        logger.info(f"Loading Sentence Transformer model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        logger.debug(f"Embedding {len(texts)} documents")

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 100
        )

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        logger.debug(f"Embedding query: {text[:50]}...")

        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize
        )

        return embedding.tolist()

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings.

        Returns:
            Embedding dimension
        """
        return self.embedding_dim


class OpenAIEmbedding(BaseEmbedding):
    """Embedding using OpenAI's embedding API."""

    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        batch_size: int = 100
    ):
        """Initialize OpenAI embedding.

        Args:
            model_name: Name of the OpenAI embedding model
            api_key: OpenAI API key (if not set in environment)
            batch_size: Batch size for API calls
        """
        self.model_name = model_name
        self.batch_size = batch_size

        # Set API key
        if api_key:
            openai.api_key = api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key must be provided or set in environment")

        logger.info(f"Initialized OpenAI embedding with model: {model_name}")

        # Set embedding dimension based on model
        self.embedding_dim = self._get_model_dimension()

    def _get_model_dimension(self) -> int:
        """Get embedding dimension for the model.

        Returns:
            Embedding dimension
        """
        dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
        return dimensions.get(self.model_name, 1536)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        logger.debug(f"Embedding {len(texts)} documents with OpenAI")

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            try:
                response = openai.embeddings.create(
                    model=self.model_name,
                    input=batch
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(f"Error embedding batch {i//self.batch_size}: {e}")
                raise

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        logger.debug(f"Embedding query with OpenAI: {text[:50]}...")

        try:
            response = openai.embeddings.create(
                model=self.model_name,
                input=text
            )

            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings.

        Returns:
            Embedding dimension
        """
        return self.embedding_dim

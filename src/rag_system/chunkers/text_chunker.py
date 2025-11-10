"""Text chunking strategies for document processing."""

import re
import nltk
from typing import List, Optional
from ..base import BaseChunker, Document
from ...utils.logger import get_logger

logger = get_logger()

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class FixedSizeChunker(BaseChunker):
    """Chunk documents into fixed-size pieces with overlap."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n"
    ):
        """Initialize fixed-size chunker.

        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separator: Preferred separator for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

    def chunk(self, documents: List[Document]) -> List[Document]:
        """Chunk documents into fixed-size pieces.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents
        """
        chunked_docs = []

        for doc in documents:
            chunks = self._split_text(doc.content)

            for i, chunk_text in enumerate(chunks):
                if chunk_text.strip():
                    chunk_metadata = {
                        **doc.metadata,
                        "chunk_id": i,
                        "chunk_method": "fixed_size",
                        "parent_doc_id": doc.doc_id
                    }

                    chunked_docs.append(Document(
                        content=chunk_text,
                        metadata=chunk_metadata,
                        doc_id=f"{doc.doc_id}_chunk_{i}" if doc.doc_id else None
                    ))

        logger.info(f"Chunked {len(documents)} documents into {len(chunked_docs)} chunks")
        return chunked_docs

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # If this is not the last chunk, try to break at separator
            if end < len(text):
                # Look for separator within the chunk
                separator_pos = text.rfind(self.separator, start, end)
                if separator_pos != -1 and separator_pos > start:
                    end = separator_pos + len(self.separator)

            chunks.append(text[start:end])
            start = end - self.chunk_overlap

        return chunks


class SemanticChunker(BaseChunker):
    """Chunk documents based on semantic boundaries (sentences/paragraphs)."""

    def __init__(
        self,
        chunk_size: int = 1000,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000
    ):
        """Initialize semantic chunker.

        Args:
            chunk_size: Target size of each chunk in characters
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
        """
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk(self, documents: List[Document]) -> List[Document]:
        """Chunk documents based on semantic boundaries.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents
        """
        chunked_docs = []

        for doc in documents:
            chunks = self._split_by_semantics(doc.content)

            for i, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) >= self.min_chunk_size:
                    chunk_metadata = {
                        **doc.metadata,
                        "chunk_id": i,
                        "chunk_method": "semantic",
                        "parent_doc_id": doc.doc_id
                    }

                    chunked_docs.append(Document(
                        content=chunk_text,
                        metadata=chunk_metadata,
                        doc_id=f"{doc.doc_id}_chunk_{i}" if doc.doc_id else None
                    ))

        logger.info(f"Semantically chunked {len(documents)} documents into {len(chunked_docs)} chunks")
        return chunked_docs

    def _split_by_semantics(self, text: str) -> List[str]:
        """Split text by semantic boundaries.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        # First, try to split by paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            # If paragraph itself is too large, split by sentences
            if len(paragraph) > self.max_chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                # Split large paragraph into sentences
                sentences = nltk.sent_tokenize(paragraph)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= self.chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
            else:
                # Add paragraph to current chunk
                if len(current_chunk) + len(paragraph) <= self.chunk_size:
                    current_chunk += paragraph + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph + "\n\n"

        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks


class HybridChunker(BaseChunker):
    """Hybrid chunking using both fixed-size and semantic approaches."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000
    ):
        """Initialize hybrid chunker.

        Args:
            chunk_size: Target size of each chunk
            chunk_overlap: Overlap between fixed-size chunks
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk(self, documents: List[Document]) -> List[Document]:
        """Chunk documents using hybrid approach.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents
        """
        chunked_docs = []

        for doc in documents:
            chunks = self._hybrid_split(doc.content)

            for i, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) >= self.min_chunk_size:
                    chunk_metadata = {
                        **doc.metadata,
                        "chunk_id": i,
                        "chunk_method": "hybrid",
                        "parent_doc_id": doc.doc_id
                    }

                    chunked_docs.append(Document(
                        content=chunk_text,
                        metadata=chunk_metadata,
                        doc_id=f"{doc.doc_id}_chunk_{i}" if doc.doc_id else None
                    ))

        logger.info(f"Hybrid chunked {len(documents)} documents into {len(chunked_docs)} chunks")
        return chunked_docs

    def _hybrid_split(self, text: str) -> List[str]:
        """Split text using hybrid approach.

        Strategy:
        1. First try semantic boundaries (paragraphs)
        2. If chunks are too large, split by sentences
        3. If still too large, use fixed-size with overlap

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        chunks = []
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        for paragraph in paragraphs:
            if len(paragraph) <= self.chunk_size:
                chunks.append(paragraph)
            elif len(paragraph) <= self.max_chunk_size:
                # Split by sentences
                sentences = nltk.sent_tokenize(paragraph)
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= self.chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
                if current_chunk:
                    chunks.append(current_chunk.strip())
            else:
                # Use fixed-size splitting for very large paragraphs
                start = 0
                while start < len(paragraph):
                    end = start + self.chunk_size
                    # Try to break at sentence boundary
                    if end < len(paragraph):
                        sentence_end = paragraph.rfind('. ', start, end)
                        if sentence_end != -1:
                            end = sentence_end + 1
                    chunks.append(paragraph[start:end].strip())
                    start = end - self.chunk_overlap

        return chunks

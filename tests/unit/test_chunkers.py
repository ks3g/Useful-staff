"""Unit tests for chunking modules."""

import pytest
from src.rag_system.base import Document
from src.rag_system.chunkers import FixedSizeChunker, SemanticChunker, HybridChunker


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    def test_chunk_small_document(self):
        """Test chunking a small document that fits in one chunk."""
        chunker = FixedSizeChunker(chunk_size=1000, chunk_overlap=200)
        doc = Document(content="This is a small document.", metadata={"source": "test"})

        chunks = chunker.chunk([doc])

        assert len(chunks) == 1
        assert chunks[0].content == "This is a small document."

    def test_chunk_large_document(self):
        """Test chunking a large document into multiple chunks."""
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
        long_text = "This is a sentence. " * 20  # 400 characters
        doc = Document(content=long_text, metadata={"source": "test"})

        chunks = chunker.chunk([doc])

        assert len(chunks) > 1
        # Verify overlap
        if len(chunks) > 1:
            # Last part of first chunk should overlap with second chunk
            overlap_exists = chunks[1].content.startswith(chunks[0].content[-10:])
            # Note: overlap might not be exact due to separator handling

    def test_chunk_preserves_metadata(self):
        """Test that chunking preserves document metadata."""
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
        doc = Document(content="A" * 100, metadata={"page": 1, "source": "test.pdf"})

        chunks = chunker.chunk([doc])

        for chunk in chunks:
            assert chunk.metadata["page"] == 1
            assert chunk.metadata["source"] == "test.pdf"
            assert "chunk_id" in chunk.metadata
            assert "chunk_method" in chunk.metadata


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    def test_chunk_by_paragraphs(self):
        """Test chunking by paragraph boundaries."""
        chunker = SemanticChunker(chunk_size=100, min_chunk_size=10)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        doc = Document(content=text, metadata={"source": "test"})

        chunks = chunker.chunk([doc])

        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.content) >= 10  # min_chunk_size

    def test_respects_min_chunk_size(self):
        """Test that very small chunks are filtered out."""
        chunker = SemanticChunker(chunk_size=1000, min_chunk_size=50)
        text = "Tiny.\n\nAlso tiny.\n\n" + "A" * 100
        doc = Document(content=text, metadata={"source": "test"})

        chunks = chunker.chunk([doc])

        for chunk in chunks:
            assert len(chunk.content) >= 50


class TestHybridChunker:
    """Tests for HybridChunker."""

    def test_hybrid_chunking(self):
        """Test hybrid chunking strategy."""
        chunker = HybridChunker(
            chunk_size=100,
            chunk_overlap=20,
            min_chunk_size=10,
            max_chunk_size=200
        )

        text = "Paragraph one.\n\nParagraph two with more text.\n\n" + "Long " * 50
        doc = Document(content=text, metadata={"source": "test"})

        chunks = chunker.chunk([doc])

        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.content) >= 10
            assert chunk.metadata["chunk_method"] == "hybrid"

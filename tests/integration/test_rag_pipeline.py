"""Integration tests for RAG pipeline."""

import pytest
import tempfile
import shutil
from pathlib import Path
from src.rag_system.rag_pipeline import RAGPipeline
from src.rag_system.config import RAGSystemConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def test_config(temp_dir):
    """Create test configuration."""
    config = RAGSystemConfig()
    config.vector_store.persist_directory = temp_dir
    config.embeddings.provider = "sentence-transformers"
    config.embeddings.model_name = "all-MiniLM-L6-v2"
    return config


class TestRAGPipeline:
    """Integration tests for RAG pipeline."""

    def test_pipeline_setup(self, test_config):
        """Test pipeline setup."""
        pipeline = RAGPipeline(config=test_config)
        pipeline.setup()

        assert pipeline.extractor is not None
        assert pipeline.chunker is not None
        assert pipeline.embedding_model is not None
        assert pipeline.vector_store is not None
        assert pipeline.retriever is not None

    def test_ingest_and_query_workflow(self, test_config, temp_dir):
        """Test complete ingest and query workflow.

        Note: This test requires a PDF file and LLM API keys.
        Skip if not available.
        """
        pytest.skip("Requires PDF file and API keys for full integration test")

        # This is a template for integration testing:
        # 1. Initialize pipeline
        pipeline = RAGPipeline(config=test_config)
        pipeline.setup()

        # 2. Ingest a test PDF
        # pdf_path = "path/to/test.pdf"
        # chunk_count = pipeline.ingest_pdf(pdf_path)
        # assert chunk_count > 0

        # 3. Query the system
        # response = pipeline.query("What is the main topic?")
        # assert response.answer
        # assert len(response.source_documents) > 0

    def test_vector_store_persistence(self, test_config, temp_dir):
        """Test vector store save and load."""
        # Create and setup pipeline
        pipeline = RAGPipeline(config=test_config)
        pipeline.setup()

        # Add some test documents (without actual PDF)
        # This would require mocking or test data
        pytest.skip("Requires test documents for persistence testing")

    def test_get_stats(self, test_config):
        """Test retrieving pipeline statistics."""
        pipeline = RAGPipeline(config=test_config)
        pipeline.setup()

        stats = pipeline.get_stats()

        assert "document_count" in stats
        assert "embedding_model" in stats
        assert "vector_store" in stats
        assert stats["embedding_model"] == "all-MiniLM-L6-v2"

"""Main RAG pipeline orchestrating all components."""

from typing import List, Optional, Dict, Any
from pathlib import Path
from .base import Document, RAGResponse, RetrievalResult, BaseLLM, BaseRetriever
from .extractors import PDFExtractor
from .chunkers import FixedSizeChunker, SemanticChunker, HybridChunker
from .embeddings import SentenceTransformerEmbedding, OpenAIEmbedding
from .vector_stores import FAISSVectorStore, ChromaVectorStore
from .retrievers import VectorRetriever, HybridRetriever
from .llm import OpenAILLM, AnthropicLLM
from .config import RAGSystemConfig, load_config
from ..utils.logger import get_logger, setup_logger

logger = get_logger()


class RAGPipeline:
    """Complete RAG pipeline for PDF question answering."""

    def __init__(self, config: Optional[RAGSystemConfig] = None):
        """Initialize RAG pipeline.

        Args:
            config: RAG system configuration
        """
        self.config = config or load_config()

        # Setup logger
        setup_logger(
            log_file=self.config.logging.log_file,
            level=self.config.logging.level
        )

        # Initialize components
        self.extractor: Optional[PDFExtractor] = None
        self.chunker = None
        self.embedding_model = None
        self.vector_store = None
        self.retriever: Optional[BaseRetriever] = None
        self.llm: Optional[BaseLLM] = None

        logger.info("RAG Pipeline initialized")

    def setup(self) -> None:
        """Setup all components of the RAG pipeline."""
        logger.info("Setting up RAG pipeline components...")

        # 1. Setup PDF extractor
        self.extractor = PDFExtractor(
            extract_images=self.config.pdf_extraction.extract_images,
            extract_tables=self.config.pdf_extraction.extract_tables,
            method="hybrid"  # Use hybrid method for best results
        )

        # 2. Setup chunker
        chunking_strategy = self.config.chunking.strategy
        if chunking_strategy == "fixed":
            self.chunker = FixedSizeChunker(
                chunk_size=self.config.chunking.chunk_size,
                chunk_overlap=self.config.chunking.chunk_overlap
            )
        elif chunking_strategy == "semantic":
            self.chunker = SemanticChunker(
                chunk_size=self.config.chunking.chunk_size,
                min_chunk_size=self.config.chunking.min_chunk_size,
                max_chunk_size=self.config.chunking.max_chunk_size
            )
        else:  # hybrid
            self.chunker = HybridChunker(
                chunk_size=self.config.chunking.chunk_size,
                chunk_overlap=self.config.chunking.chunk_overlap,
                min_chunk_size=self.config.chunking.min_chunk_size,
                max_chunk_size=self.config.chunking.max_chunk_size
            )

        # 3. Setup embedding model
        if self.config.embeddings.provider == "sentence-transformers":
            self.embedding_model = SentenceTransformerEmbedding(
                model_name=self.config.embeddings.model_name,
                batch_size=self.config.embeddings.batch_size,
                normalize=self.config.embeddings.normalize
            )
        elif self.config.embeddings.provider == "openai":
            self.embedding_model = OpenAIEmbedding(
                model_name=self.config.embeddings.model_name,
                batch_size=self.config.embeddings.batch_size
            )
        else:
            raise ValueError(f"Unknown embedding provider: {self.config.embeddings.provider}")

        # 4. Setup vector store
        embedding_dim = self.embedding_model.get_embedding_dimension()

        if self.config.vector_store.provider == "faiss":
            self.vector_store = FAISSVectorStore(
                embedding_dimension=embedding_dim,
                index_type=self.config.vector_store.index_type,
                persist_directory=self.config.vector_store.persist_directory
            )
        elif self.config.vector_store.provider == "chroma":
            self.vector_store = ChromaVectorStore(
                collection_name="rag_documents",
                persist_directory=self.config.vector_store.persist_directory
            )
        else:
            raise ValueError(f"Unknown vector store provider: {self.config.vector_store.provider}")

        # 5. Setup retriever
        if self.config.retrieval.reranking:
            self.retriever = HybridRetriever(
                vector_store=self.vector_store,
                embedding_model=self.embedding_model,
                reranker_model=self.config.retrieval.reranker_model,
                score_threshold=self.config.retrieval.score_threshold
            )
        else:
            self.retriever = VectorRetriever(
                vector_store=self.vector_store,
                embedding_model=self.embedding_model,
                score_threshold=self.config.retrieval.score_threshold
            )

        # 6. Setup LLM
        if self.config.llm.provider == "openai":
            self.llm = OpenAILLM(
                model_name=self.config.llm.model_name,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                timeout=self.config.llm.timeout
            )
        elif self.config.llm.provider == "anthropic":
            self.llm = AnthropicLLM(
                model_name=self.config.llm.model_name,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                timeout=self.config.llm.timeout
            )
        else:
            raise ValueError(f"Unknown LLM provider: {self.config.llm.provider}")

        logger.info("RAG pipeline setup complete")

    def ingest_pdf(self, pdf_path: str) -> int:
        """Ingest a PDF file into the RAG system.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Number of chunks ingested
        """
        logger.info(f"Ingesting PDF: {pdf_path}")

        # 1. Extract documents from PDF
        documents = self.extractor.extract(pdf_path)
        logger.info(f"Extracted {len(documents)} pages")

        # 2. Chunk documents
        chunked_docs = self.chunker.chunk(documents)
        logger.info(f"Created {len(chunked_docs)} chunks")

        # 3. Generate embeddings
        texts = [doc.content for doc in chunked_docs]
        embeddings = self.embedding_model.embed_documents(texts)

        # Add embeddings to documents
        for doc, embedding in zip(chunked_docs, embeddings):
            doc.embedding = embedding

        # 4. Add to vector store
        self.vector_store.add_documents(chunked_docs)

        # 5. Persist vector store
        self.vector_store.persist()

        logger.info(f"Successfully ingested {len(chunked_docs)} chunks from {pdf_path}")
        return len(chunked_docs)

    def ingest_multiple_pdfs(self, pdf_paths: List[str]) -> Dict[str, int]:
        """Ingest multiple PDF files.

        Args:
            pdf_paths: List of PDF file paths

        Returns:
            Dictionary mapping file paths to chunk counts
        """
        results = {}
        for pdf_path in pdf_paths:
            try:
                count = self.ingest_pdf(pdf_path)
                results[pdf_path] = count
            except Exception as e:
                logger.error(f"Error ingesting {pdf_path}: {e}")
                results[pdf_path] = 0

        return results

    def query(
        self,
        question: str,
        top_k: int = None,
        return_sources: bool = True
    ) -> RAGResponse:
        """Query the RAG system.

        Args:
            question: User question
            top_k: Number of documents to retrieve (uses config default if None)
            return_sources: Whether to return source documents

        Returns:
            RAG response with answer and sources
        """
        if top_k is None:
            top_k = self.config.retrieval.top_k

        logger.info(f"Processing query: {question}")

        # 1. Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query=question, top_k=top_k)
        logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")

        if not retrieved_docs:
            logger.warning("No relevant documents found")
            return RAGResponse(
                query=question,
                answer="I couldn't find any relevant information to answer your question.",
                source_documents=[],
                metadata={"retrieved_docs": 0}
            )

        # 2. Prepare context from retrieved documents
        context = self._prepare_context(retrieved_docs)

        # 3. Generate prompt
        prompt = self._create_prompt(question, context)

        # 4. Generate answer
        answer = self.llm.generate(prompt)

        # 5. Create response
        response = RAGResponse(
            query=question,
            answer=answer,
            source_documents=retrieved_docs if return_sources else [],
            metadata={
                "retrieved_docs": len(retrieved_docs),
                "context_length": len(context),
                "prompt_length": len(prompt)
            }
        )

        logger.info("Query processed successfully")
        return response

    def _prepare_context(self, retrieved_docs: List[RetrievalResult]) -> str:
        """Prepare context from retrieved documents.

        Args:
            retrieved_docs: Retrieved documents

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, result in enumerate(retrieved_docs, 1):
            doc = result.document
            context_parts.append(f"[Document {i}]")
            context_parts.append(doc.content)

            # Add metadata if available
            if "page" in doc.metadata:
                context_parts.append(f"(Source: {doc.metadata.get('filename', 'Unknown')}, Page {doc.metadata['page']})")

            context_parts.append("")  # Empty line between documents

        context = "\n".join(context_parts)

        # Truncate if too long
        max_length = self.config.rag.max_context_length
        if len(context) > max_length:
            logger.warning(f"Context truncated from {len(context)} to {max_length} characters")
            context = context[:max_length] + "..."

        return context

    def _create_prompt(self, question: str, context: str) -> str:
        """Create prompt for LLM.

        Args:
            question: User question
            context: Context from retrieved documents

        Returns:
            Formatted prompt
        """
        template = self.config.rag.prompt_template
        prompt = template.format(context=context, query=question)
        return prompt

    def load_vector_store(self) -> None:
        """Load existing vector store from disk."""
        logger.info("Loading vector store...")
        self.vector_store.load()
        logger.info(f"Loaded vector store with {self.vector_store.get_document_count()} documents")

    def clear_vector_store(self) -> None:
        """Clear the vector store."""
        logger.warning("Clearing vector store...")
        self.vector_store.clear()
        logger.info("Vector store cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system.

        Returns:
            Dictionary of statistics
        """
        return {
            "document_count": self.vector_store.get_document_count(),
            "embedding_model": self.config.embeddings.model_name,
            "vector_store": self.config.vector_store.provider,
            "llm_model": self.config.llm.model_name,
            "chunking_strategy": self.config.chunking.strategy
        }

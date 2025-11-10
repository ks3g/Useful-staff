"""Configuration management for RAG system."""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class PDFExtractionConfig(BaseModel):
    """PDF extraction configuration."""

    extract_images: bool = False
    extract_tables: bool = True
    ocr_enabled: bool = False


class ChunkingConfig(BaseModel):
    """Chunking configuration."""

    strategy: str = "semantic"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000


class EmbeddingsConfig(BaseModel):
    """Embeddings configuration."""

    provider: str = "sentence-transformers"
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize: bool = True


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""

    provider: str = "faiss"
    index_type: str = "IndexFlatL2"
    similarity_metric: str = "cosine"
    persist_directory: str = "./data/vector_store"


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""

    top_k: int = 5
    score_threshold: float = 0.7
    reranking: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class LLMConfig(BaseModel):
    """LLM configuration."""

    provider: str = "openai"
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_tokens: int = 1000
    timeout: int = 30


class RAGConfig(BaseModel):
    """RAG pipeline configuration."""

    max_context_length: int = 4000
    prompt_template: str = Field(
        default="""Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query}
Answer:"""
    )


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    metrics: list = Field(
        default=[
            "retrieval_precision",
            "retrieval_recall",
            "mrr",
            "ndcg",
            "rouge",
            "bert_score",
            "answer_relevancy",
            "faithfulness"
        ]
    )
    test_dataset_path: str = "./data/eval/test_questions.json"


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    log_file: str = "./logs/rag_system.log"


class CacheConfig(BaseModel):
    """Cache configuration."""

    enabled: bool = True
    cache_dir: str = "./data/cache"
    ttl: int = 3600


class RAGSystemConfig(BaseModel):
    """Complete RAG system configuration."""

    pdf_extraction: PDFExtractionConfig = Field(default_factory=PDFExtractionConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)


class Settings(BaseSettings):
    """Environment settings."""

    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    huggingface_token: Optional[str] = None
    environment: str = "development"
    debug: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = False


def load_config(config_path: str = "config.yaml") -> RAGSystemConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        RAGSystemConfig object
    """
    config_file = Path(config_path)

    if not config_file.exists():
        # Return default configuration
        return RAGSystemConfig()

    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)

    return RAGSystemConfig(**config_dict)


def get_settings() -> Settings:
    """Get environment settings.

    Returns:
        Settings object
    """
    return Settings()

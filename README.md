# RAG System for PDF Data Extraction

A comprehensive Retrieval-Augmented Generation (RAG) system for extracting and querying information from PDF documents. This system features modular architecture, multiple model options, and built-in evaluation metrics.

## Features

### Core Capabilities
- **PDF Extraction**: Support for text, tables, and metadata extraction using PyMuPDF and PDFPlumber
- **Flexible Chunking**: Multiple strategies (fixed-size, semantic, hybrid) for optimal document segmentation
- **Multiple Embedding Models**: Support for Sentence Transformers and OpenAI embeddings
- **Vector Stores**: FAISS and ChromaDB backends for efficient similarity search
- **Advanced Retrieval**: Vector similarity search with optional cross-encoder reranking
- **LLM Integration**: OpenAI and Anthropic Claude support for answer generation
- **Comprehensive Evaluation**: Built-in metrics for retrieval and generation quality

### Key Components

1. **PDF Extraction** (`src/rag_system/extractors/`)
   - PyMuPDF for fast text extraction
   - PDFPlumber for table extraction
   - Hybrid mode combining both approaches

2. **Document Chunking** (`src/rag_system/chunkers/`)
   - Fixed-size chunking with overlap
   - Semantic chunking based on paragraphs/sentences
   - Hybrid approach for optimal results

3. **Embeddings** (`src/rag_system/embeddings/`)
   - Sentence Transformers (local, free)
   - OpenAI embeddings (API-based)

4. **Vector Stores** (`src/rag_system/vector_stores/`)
   - FAISS (fast, in-memory or persistent)
   - ChromaDB (feature-rich, persistent)

5. **Retrieval** (`src/rag_system/retrievers/`)
   - Vector similarity search
   - Hybrid retrieval with reranking
   - Configurable score thresholds

6. **LLM Providers** (`src/rag_system/llm/`)
   - OpenAI (GPT-3.5, GPT-4)
   - Anthropic Claude

7. **Evaluation** (`src/rag_system/evaluation/`)
   - Retrieval metrics: Precision@K, Recall@K, MRR, NDCG
   - Generation metrics: ROUGE, BERTScore
   - Answer relevancy and faithfulness

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Useful-staff
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
python -c "import nltk; nltk.download('punkt')"
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

## Configuration

The system is configured via `config.yaml`. Key settings:

```yaml
# Embedding model selection
embeddings:
  provider: "sentence-transformers"  # or "openai"
  model_name: "all-MiniLM-L6-v2"

# Vector store selection
vector_store:
  provider: "faiss"  # or "chroma"
  persist_directory: "./data/vector_store"

# LLM selection
llm:
  provider: "openai"  # or "anthropic"
  model_name: "gpt-3.5-turbo"

# Retrieval settings
retrieval:
  top_k: 5
  reranking: true
```

See `config.yaml` for all available options.

## Quick Start

### Basic Usage

```python
from src.rag_system.rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()
pipeline.setup()

# Ingest PDF
pipeline.ingest_pdf("your_document.pdf")

# Query the system
response = pipeline.query("What is this document about?")
print(response.answer)

# View sources
for result in response.source_documents:
    print(f"Page {result.document.metadata['page']}: Score {result.score}")
```

### Running Examples

1. **Basic usage**:
```bash
python examples/basic_usage.py
```

2. **Evaluation**:
```bash
python examples/evaluation_example.py
```

## Usage Guide

### 1. Ingesting Documents

```python
from src.rag_system.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.setup()

# Single PDF
chunk_count = pipeline.ingest_pdf("document.pdf")

# Multiple PDFs
pdf_paths = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
results = pipeline.ingest_multiple_pdfs(pdf_paths)
```

### 2. Querying

```python
# Simple query
response = pipeline.query("What are the main findings?")
print(response.answer)

# Query with custom parameters
response = pipeline.query(
    question="Explain the methodology",
    top_k=10,  # Retrieve more documents
    return_sources=True
)

# Access metadata
print(f"Retrieved {response.metadata['retrieved_docs']} documents")
```

### 3. Vector Store Management

```python
# Save vector store
pipeline.vector_store.persist()

# Load existing vector store
pipeline.load_vector_store()

# Clear vector store
pipeline.clear_vector_store()

# Get statistics
stats = pipeline.get_stats()
print(f"Document count: {stats['document_count']}")
```

### 4. Evaluation

```python
from src.rag_system.evaluation import RAGEvaluator

evaluator = RAGEvaluator()

# Evaluate retrieval
retrieval_metrics = evaluator.evaluate_retrieval(
    retrieved_results,
    relevant_docs
)

# Evaluate generation
generation_metrics = evaluator.evaluate_generation(
    generated_answers,
    reference_answers
)
```

## Testing

Run the test suite:

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# With coverage
pytest --cov=src --cov-report=html
```

## Project Structure

```
Useful-staff/
├── src/
│   ├── rag_system/
│   │   ├── __init__.py
│   │   ├── base.py                 # Base classes and interfaces
│   │   ├── config.py               # Configuration management
│   │   ├── rag_pipeline.py         # Main RAG pipeline
│   │   ├── extractors/             # PDF extraction
│   │   ├── chunkers/               # Document chunking
│   │   ├── embeddings/             # Embedding models
│   │   ├── vector_stores/          # Vector store implementations
│   │   ├── retrievers/             # Retrieval mechanisms
│   │   ├── llm/                    # LLM providers
│   │   └── evaluation/             # Evaluation metrics
│   └── utils/
│       └── logger.py               # Logging utilities
├── tests/
│   ├── unit/                       # Unit tests
│   └── integration/                # Integration tests
├── examples/
│   ├── basic_usage.py             # Basic usage example
│   └── evaluation_example.py      # Evaluation example
├── data/
│   ├── pdfs/                      # Input PDFs
│   ├── vector_store/              # Persisted vector stores
│   ├── cache/                     # Cache directory
│   └── eval/                      # Evaluation results
├── logs/                          # Log files
├── config.yaml                    # Configuration file
├── requirements.txt               # Python dependencies
├── pytest.ini                     # Pytest configuration
└── README.md                      # This file
```

## API Keys

The system requires API keys for certain providers:

- **OpenAI**: Required if using OpenAI embeddings or LLM
- **Anthropic**: Required if using Claude LLM
- **HuggingFace**: Optional, for private models

Set these in your `.env` file:

```bash
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
HUGGINGFACE_TOKEN=your_token_here
```

## Model Selection Guide

### Embeddings

**Sentence Transformers (Recommended for starting)**
- Pros: Free, runs locally, good quality
- Cons: Requires GPU for best performance
- Models: `all-MiniLM-L6-v2` (fast), `all-mpnet-base-v2` (better quality)

**OpenAI**
- Pros: High quality, no local compute needed
- Cons: Costs money, API dependency
- Models: `text-embedding-ada-002`, `text-embedding-3-small`

### Vector Stores

**FAISS (Recommended for starting)**
- Pros: Fast, memory efficient, battle-tested
- Cons: Less features than ChromaDB

**ChromaDB**
- Pros: Feature-rich, easy metadata filtering
- Cons: Slightly slower for large datasets

### LLMs

**OpenAI**
- `gpt-3.5-turbo`: Fast, cheap, good for most tasks
- `gpt-4`: Best quality, slower, more expensive

**Anthropic Claude**
- `claude-3-sonnet`: Balanced performance
- `claude-3-opus`: Highest quality

## Performance Tips

1. **Use appropriate chunk size**: 500-1000 chars for general content
2. **Enable reranking**: Significantly improves retrieval quality
3. **Adjust top_k**: Start with 5, increase for complex queries
4. **Use FAISS for speed**: Especially with large document collections
5. **Cache embeddings**: Vector store persistence avoids recomputation

## Evaluation Metrics

### Retrieval Metrics
- **Precision@K**: Proportion of retrieved documents that are relevant
- **Recall@K**: Proportion of relevant documents that are retrieved
- **MRR**: Mean Reciprocal Rank of first relevant document
- **NDCG**: Normalized Discounted Cumulative Gain

### Generation Metrics
- **ROUGE**: N-gram overlap with reference answers
- **BERTScore**: Semantic similarity using BERT embeddings
- **Answer Relevancy**: Query-answer term overlap
- **Faithfulness**: Context-answer consistency

## Troubleshooting

### Common Issues

**Issue**: Out of memory errors
- Solution: Reduce batch size in config, use smaller embedding model

**Issue**: Slow performance
- Solution: Use FAISS instead of Chroma, enable GPU for embeddings

**Issue**: Poor retrieval quality
- Solution: Enable reranking, adjust chunk size, use better embedding model

**Issue**: API rate limits
- Solution: Add retry logic, reduce batch sizes, use local models

## Contributing

Contributions are welcome! Areas for improvement:

- Additional embedding models
- More sophisticated chunking strategies
- Additional evaluation metrics
- Support for more document formats
- Query optimization techniques

## Acknowledgments

Built with:
- LangChain for RAG components
- Sentence Transformers for embeddings
- FAISS for vector search
- ChromaDB for vector storage
- PyMuPDF and PDFPlumber for PDF processing

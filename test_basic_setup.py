"""Quick test to verify RAG system setup."""

import sys
from pathlib import Path

print("Testing RAG System Setup...")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    from src.rag_system.base import Document, RetrievalResult, RAGResponse
    print("   ✓ Base classes imported successfully")
except Exception as e:
    print(f"   ✗ Error importing base classes: {e}")
    sys.exit(1)

try:
    from src.rag_system.config import load_config
    print("   ✓ Config module imported successfully")
except Exception as e:
    print(f"   ✗ Error importing config: {e}")
    sys.exit(1)

try:
    from src.rag_system.extractors import PDFExtractor
    print("   ✓ PDF extractor imported successfully")
except Exception as e:
    print(f"   ✗ Error importing extractor: {e}")
    sys.exit(1)

try:
    from src.rag_system.chunkers import FixedSizeChunker, SemanticChunker, HybridChunker
    print("   ✓ Chunkers imported successfully")
except Exception as e:
    print(f"   ✗ Error importing chunkers: {e}")
    sys.exit(1)

try:
    from src.rag_system.embeddings import SentenceTransformerEmbedding
    print("   ✓ Embeddings imported successfully")
except Exception as e:
    print(f"   ✗ Error importing embeddings: {e}")
    sys.exit(1)

try:
    from src.rag_system.vector_stores import FAISSVectorStore, ChromaVectorStore
    print("   ✓ Vector stores imported successfully")
except Exception as e:
    print(f"   ✗ Error importing vector stores: {e}")
    sys.exit(1)

try:
    from src.rag_system.retrievers import VectorRetriever, HybridRetriever
    print("   ✓ Retrievers imported successfully")
except Exception as e:
    print(f"   ✗ Error importing retrievers: {e}")
    sys.exit(1)

try:
    from src.rag_system.evaluation import RetrievalMetrics, GenerationMetrics, RAGEvaluator
    print("   ✓ Evaluation metrics imported successfully")
except Exception as e:
    print(f"   ✗ Error importing evaluation: {e}")
    sys.exit(1)

try:
    from src.rag_system.rag_pipeline import RAGPipeline
    print("   ✓ RAG Pipeline imported successfully")
except Exception as e:
    print(f"   ✗ Error importing RAG pipeline: {e}")
    sys.exit(1)

# Test configuration loading
print("\n2. Testing configuration...")
try:
    config = load_config("config.yaml")
    print(f"   ✓ Configuration loaded successfully")
    print(f"   - Embedding provider: {config.embeddings.provider}")
    print(f"   - Vector store: {config.vector_store.provider}")
    print(f"   - LLM provider: {config.llm.provider}")
except Exception as e:
    print(f"   ✗ Error loading configuration: {e}")
    sys.exit(1)

# Test basic document operations
print("\n3. Testing basic operations...")
try:
    doc = Document(
        content="This is a test document.",
        metadata={"source": "test"},
        doc_id="test_doc_1"
    )
    print(f"   ✓ Document created: {doc.doc_id}")
except Exception as e:
    print(f"   ✗ Error creating document: {e}")
    sys.exit(1)

# Test chunking
print("\n4. Testing chunking...")
try:
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk([doc])
    print(f"   ✓ Chunking works: created {len(chunks)} chunk(s)")
except Exception as e:
    print(f"   ✗ Error testing chunker: {e}")
    sys.exit(1)

# Test PDF extraction (if PDF exists)
print("\n5. Testing PDF extraction...")
pdf_path = Path("gerics_klimaausblick_hessen_version1.2_deutsch.pdf")
if pdf_path.exists():
    try:
        extractor = PDFExtractor(method="pymupdf")
        docs = extractor.extract(str(pdf_path))
        print(f"   ✓ PDF extraction works: extracted {len(docs)} pages")
    except Exception as e:
        print(f"   ✗ Error extracting PDF: {e}")
else:
    print(f"   - PDF file not found (skipping): {pdf_path}")

print("\n" + "=" * 60)
print("All basic tests passed! ✓")
print("=" * 60)
print("\nNext steps:")
print("1. Install dependencies: pip install -r requirements.txt")
print("2. Set up API keys in .env file")
print("3. Run examples: python examples/basic_usage.py")
print("4. Run tests: pytest")

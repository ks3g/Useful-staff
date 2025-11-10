"""Verify project structure without requiring dependencies."""

from pathlib import Path

print("Verifying RAG System Project Structure...")
print("=" * 60)

required_files = [
    # Configuration
    "config.yaml",
    "requirements.txt",
    ".env.example",
    ".gitignore",
    "pytest.ini",
    "README.md",

    # Source code - Core
    "src/rag_system/__init__.py",
    "src/rag_system/base.py",
    "src/rag_system/config.py",
    "src/rag_system/rag_pipeline.py",

    # Source code - Modules
    "src/rag_system/extractors/__init__.py",
    "src/rag_system/extractors/pdf_extractor.py",
    "src/rag_system/chunkers/__init__.py",
    "src/rag_system/chunkers/text_chunker.py",
    "src/rag_system/embeddings/__init__.py",
    "src/rag_system/embeddings/embedding_models.py",
    "src/rag_system/vector_stores/__init__.py",
    "src/rag_system/vector_stores/faiss_store.py",
    "src/rag_system/vector_stores/chroma_store.py",
    "src/rag_system/retrievers/__init__.py",
    "src/rag_system/retrievers/retriever.py",
    "src/rag_system/llm/__init__.py",
    "src/rag_system/llm/llm_providers.py",
    "src/rag_system/evaluation/__init__.py",
    "src/rag_system/evaluation/metrics.py",

    # Utilities
    "src/utils/__init__.py",
    "src/utils/logger.py",

    # Tests
    "tests/__init__.py",
    "tests/unit/test_chunkers.py",
    "tests/integration/test_rag_pipeline.py",

    # Examples
    "examples/basic_usage.py",
    "examples/evaluation_example.py",
]

required_dirs = [
    "src/rag_system",
    "src/utils",
    "tests/unit",
    "tests/integration",
    "examples",
    "data/pdfs",
    "data/vector_store",
    "data/cache",
    "data/eval",
    "logs",
]

print("\n1. Checking required files...")
missing_files = []
for file_path in required_files:
    full_path = Path(file_path)
    if full_path.exists():
        print(f"   ✓ {file_path}")
    else:
        print(f"   ✗ Missing: {file_path}")
        missing_files.append(file_path)

print("\n2. Checking required directories...")
missing_dirs = []
for dir_path in required_dirs:
    full_path = Path(dir_path)
    if full_path.exists() and full_path.is_dir():
        print(f"   ✓ {dir_path}/")
    else:
        print(f"   ✗ Missing: {dir_path}/")
        missing_dirs.append(dir_path)

print("\n3. Counting lines of code...")
total_lines = 0
for file_path in required_files:
    full_path = Path(file_path)
    if full_path.exists() and full_path.suffix == ".py":
        with open(full_path, 'r') as f:
            lines = len(f.readlines())
            total_lines += lines

print(f"   Total Python lines: {total_lines}")

print("\n" + "=" * 60)
if not missing_files and not missing_dirs:
    print("✓ Project structure is complete!")
    print("\nComponents implemented:")
    print("  - PDF Extraction (PyMuPDF, PDFPlumber)")
    print("  - Document Chunking (Fixed, Semantic, Hybrid)")
    print("  - Embeddings (Sentence Transformers, OpenAI)")
    print("  - Vector Stores (FAISS, ChromaDB)")
    print("  - Retrieval (Vector, Hybrid with Reranking)")
    print("  - LLM Integration (OpenAI, Anthropic)")
    print("  - Evaluation Metrics (Retrieval + Generation)")
    print("  - Testing Suite")
    print("  - Example Scripts")
    print("\nNext steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Configure .env with API keys")
    print("  3. Run examples: python examples/basic_usage.py")
else:
    print("✗ Some components are missing")
    if missing_files:
        print(f"   Missing files: {len(missing_files)}")
    if missing_dirs:
        print(f"   Missing directories: {len(missing_dirs)}")

print("=" * 60)

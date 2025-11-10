"""Basic usage example for RAG system."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_system.rag_pipeline import RAGPipeline
from src.rag_system.config import load_config


def main():
    """Run basic RAG pipeline example."""

    # 1. Load configuration
    print("Loading configuration...")
    config = load_config("config.yaml")

    # 2. Initialize and setup pipeline
    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline(config=config)
    pipeline.setup()

    # 3. Ingest a PDF file
    pdf_path = "gerics_klimaausblick_hessen_version1.2_deutsch.pdf"

    print(f"\nIngesting PDF: {pdf_path}")
    try:
        chunk_count = pipeline.ingest_pdf(pdf_path)
        print(f"Successfully ingested {chunk_count} chunks")
    except Exception as e:
        print(f"Error ingesting PDF: {e}")
        return

    # 4. Get pipeline statistics
    stats = pipeline.get_stats()
    print(f"\nPipeline Statistics:")
    print(f"  - Documents: {stats['document_count']}")
    print(f"  - Embedding Model: {stats['embedding_model']}")
    print(f"  - Vector Store: {stats['vector_store']}")
    print(f"  - LLM Model: {stats['llm_model']}")

    # 5. Query the system
    print("\n" + "="*60)
    print("Querying the RAG system...")
    print("="*60)

    questions = [
        "What is the main topic of this document?",
        "What information is provided about climate?",
        "Summarize the key findings."
    ]

    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 60)

        try:
            response = pipeline.query(question, top_k=3)

            print(f"Answer: {response.answer}\n")
            print(f"Source Documents: {len(response.source_documents)}")

            # Show source document details
            for j, result in enumerate(response.source_documents, 1):
                doc = result.document
                print(f"  [{j}] Score: {result.score:.3f} | "
                      f"Page: {doc.metadata.get('page', 'N/A')} | "
                      f"Source: {doc.metadata.get('filename', 'N/A')}")

        except Exception as e:
            print(f"Error processing query: {e}")

    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)


if __name__ == "__main__":
    main()

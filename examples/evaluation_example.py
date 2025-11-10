"""Example of evaluating the RAG system."""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_system.rag_pipeline import RAGPipeline
from src.rag_system.evaluation import RAGEvaluator
from src.rag_system.config import load_config


def create_test_dataset():
    """Create a sample test dataset for evaluation.

    Returns:
        List of test cases with queries, answers, and relevant docs
    """
    # This is a template - replace with actual test data
    test_dataset = [
        {
            "query": "What is the document about?",
            "answer": "This document discusses climate outlook for Hessen.",
            "relevant_docs": ["doc_1_chunk_0", "doc_1_chunk_1"]
        },
        {
            "query": "What are the main findings?",
            "answer": "The main findings include temperature and precipitation projections.",
            "relevant_docs": ["doc_1_chunk_2", "doc_1_chunk_3"]
        }
    ]

    return test_dataset


def main():
    """Run RAG system evaluation."""

    print("="*60)
    print("RAG System Evaluation")
    print("="*60)

    # 1. Setup pipeline
    print("\nSetting up RAG pipeline...")
    config = load_config("config.yaml")
    pipeline = RAGPipeline(config=config)
    pipeline.setup()

    # 2. Load or ingest documents
    pdf_path = "gerics_klimaausblick_hessen_version1.2_deutsch.pdf"
    if pipeline.vector_store.get_document_count() == 0:
        print(f"Ingesting PDF: {pdf_path}")
        pipeline.ingest_pdf(pdf_path)
    else:
        print(f"Using existing vector store with {pipeline.vector_store.get_document_count()} documents")

    # 3. Create test dataset
    print("\nCreating test dataset...")
    test_dataset = create_test_dataset()
    print(f"Test dataset size: {len(test_dataset)}")

    # 4. Run queries and collect responses
    print("\nRunning queries...")
    rag_responses = []

    for test_case in test_dataset:
        query = test_case["query"]
        print(f"  - {query}")

        response = pipeline.query(query, top_k=5)
        rag_responses.append(response)

    # 5. Evaluate
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)

    evaluator = RAGEvaluator()

    # Evaluate retrieval
    print("\nRetrieval Metrics:")
    print("-" * 60)

    retrieved_results = [resp.source_documents for resp in rag_responses]
    relevant_docs = [test_case["relevant_docs"] for test_case in test_dataset]

    try:
        retrieval_metrics = evaluator.evaluate_retrieval(retrieved_results, relevant_docs)
        for metric, score in retrieval_metrics.items():
            print(f"  {metric}: {score:.4f}")
    except Exception as e:
        print(f"  Error evaluating retrieval: {e}")

    # Evaluate generation
    print("\nGeneration Metrics:")
    print("-" * 60)

    generated_answers = [resp.answer for resp in rag_responses]
    reference_answers = [test_case["answer"] for test_case in test_dataset]

    try:
        generation_metrics = evaluator.evaluate_generation(generated_answers, reference_answers)
        for metric, score in generation_metrics.items():
            print(f"  {metric}: {score:.4f}")
    except Exception as e:
        print(f"  Error evaluating generation: {e}")

    # Evaluate answer relevancy and faithfulness
    print("\nAnswer Quality Metrics:")
    print("-" * 60)

    for i, (response, test_case) in enumerate(zip(rag_responses, test_dataset), 1):
        print(f"\nQuery {i}: {test_case['query']}")

        # Get context from retrieved documents
        context = "\n".join([doc.document.content for doc in response.source_documents])

        relevancy = evaluator.evaluate_answer_relevancy(
            test_case['query'],
            response.answer,
            context
        )
        faithfulness = evaluator.evaluate_faithfulness(
            response.answer,
            context
        )

        print(f"  Answer Relevancy: {relevancy:.4f}")
        print(f"  Faithfulness: {faithfulness:.4f}")

    # 6. Save results
    results = {
        "test_dataset_size": len(test_dataset),
        "retrieval_metrics": retrieval_metrics if 'retrieval_metrics' in locals() else {},
        "generation_metrics": generation_metrics if 'generation_metrics' in locals() else {}
    }

    output_path = Path("data/eval/evaluation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print(f"Evaluation complete! Results saved to {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()

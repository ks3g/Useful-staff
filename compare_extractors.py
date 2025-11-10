"""Compare PyMuPDF and Docling PDF extractors."""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_system.extractors.pdf_extractor import PDFExtractor
from src.rag_system.extractors.docling_extractor import DoclingExtractor
from src.rag_system.base import Document


def extract_with_timer(extractor, pdf_path: str, method_name: str) -> tuple:
    """Extract PDF and measure time.

    Args:
        extractor: Extractor instance
        pdf_path: Path to PDF file
        method_name: Name of extraction method

    Returns:
        Tuple of (documents, extraction_time, error)
    """
    print(f"\n{'='*60}")
    print(f"Extracting with {method_name}...")
    print(f"{'='*60}")

    start_time = time.time()
    error = None
    documents = []

    try:
        documents = extractor.extract(pdf_path)
        extraction_time = time.time() - start_time
        print(f"✓ Extracted {len(documents)} pages in {extraction_time:.2f} seconds")
    except Exception as e:
        extraction_time = time.time() - start_time
        error = str(e)
        print(f"✗ Error: {error}")

    return documents, extraction_time, error


def analyze_documents(documents: List[Document]) -> Dict[str, Any]:
    """Analyze extracted documents.

    Args:
        documents: List of extracted documents

    Returns:
        Analysis results
    """
    if not documents:
        return {
            "num_documents": 0,
            "total_chars": 0,
            "avg_chars_per_doc": 0,
            "num_tables": 0,
            "pages_with_tables": 0,
            "metadata_fields": []
        }

    total_chars = sum(len(doc.content) for doc in documents)
    num_tables = sum(len(doc.metadata.get("tables", [])) for doc in documents)
    pages_with_tables = sum(1 for doc in documents if doc.metadata.get("tables"))

    # Get all unique metadata fields
    metadata_fields = set()
    for doc in documents:
        metadata_fields.update(doc.metadata.keys())

    return {
        "num_documents": len(documents),
        "total_chars": total_chars,
        "avg_chars_per_doc": total_chars / len(documents) if documents else 0,
        "num_tables": num_tables,
        "pages_with_tables": pages_with_tables,
        "metadata_fields": sorted(list(metadata_fields))
    }


def compare_content(docs1: List[Document], docs2: List[Document],
                   name1: str, name2: str) -> Dict[str, Any]:
    """Compare content from two extractors.

    Args:
        docs1: Documents from first extractor
        docs2: Documents from second extractor
        name1: Name of first extractor
        name2: Name of second extractor

    Returns:
        Comparison results
    """
    comparison = {
        "num_pages_match": len(docs1) == len(docs2),
        f"{name1}_pages": len(docs1),
        f"{name2}_pages": len(docs2),
        "content_differences": []
    }

    # Compare first page content (sample)
    if docs1 and docs2:
        content1 = docs1[0].content
        content2 = docs2[0].content

        comparison["first_page_comparison"] = {
            f"{name1}_length": len(content1),
            f"{name2}_length": len(content2),
            f"{name1}_preview": content1[:200] + "..." if len(content1) > 200 else content1,
            f"{name2}_preview": content2[:200] + "..." if len(content2) > 200 else content2,
        }

    return comparison


def save_sample_pages(documents: List[Document], output_dir: Path, prefix: str):
    """Save sample pages for manual inspection.

    Args:
        documents: Extracted documents
        output_dir: Output directory
        prefix: File prefix
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save first 3 pages
    for i, doc in enumerate(documents[:3], 1):
        output_file = output_dir / f"{prefix}_page_{i}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Page {doc.metadata.get('page', i)}\n")
            f.write("=" * 60 + "\n\n")
            f.write(doc.content)
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("METADATA:\n")
            f.write(json.dumps(doc.metadata, indent=2, default=str))


def main():
    """Run comparison between PyMuPDF and Docling."""

    print("\n" + "=" * 60)
    print("PDF EXTRACTOR COMPARISON: PyMuPDF vs Docling")
    print("=" * 60)

    # PDF file to test
    pdf_path = "gerics_klimaausblick_hessen_version1.2_deutsch.pdf"

    if not Path(pdf_path).exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return

    # 1. Extract with PyMuPDF
    pymupdf_extractor = PDFExtractor(
        extract_tables=True,
        extract_metadata=True,
        method="pymupdf"
    )
    pymupdf_docs, pymupdf_time, pymupdf_error = extract_with_timer(
        pymupdf_extractor, pdf_path, "PyMuPDF"
    )

    # 2. Extract with Docling
    docling_extractor = DoclingExtractor(
        extract_tables=True,
        extract_metadata=True,
        preserve_structure=True
    )
    docling_docs, docling_time, docling_error = extract_with_timer(
        docling_extractor, pdf_path, "Docling"
    )

    # 3. Analyze results
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    pymupdf_analysis = analyze_documents(pymupdf_docs)
    docling_analysis = analyze_documents(docling_docs)

    print("\nPyMuPDF Analysis:")
    print(json.dumps(pymupdf_analysis, indent=2))

    print("\nDocling Analysis:")
    print(json.dumps(docling_analysis, indent=2))

    # 4. Compare
    if pymupdf_docs and docling_docs:
        print("\n" + "=" * 60)
        print("CONTENT COMPARISON")
        print("=" * 60)

        comparison = compare_content(
            pymupdf_docs, docling_docs,
            "PyMuPDF", "Docling"
        )
        print(json.dumps(comparison, indent=2))

    # 5. Save sample pages
    output_dir = Path("comparison_results")
    save_sample_pages(pymupdf_docs, output_dir, "pymupdf")
    save_sample_pages(docling_docs, output_dir, "docling")

    print(f"\n✓ Sample pages saved to {output_dir}/")

    # 6. Generate summary report
    report = {
        "pdf_file": pdf_path,
        "extraction_results": {
            "pymupdf": {
                "time_seconds": pymupdf_time,
                "error": pymupdf_error,
                "analysis": pymupdf_analysis
            },
            "docling": {
                "time_seconds": docling_time,
                "error": docling_error,
                "analysis": docling_analysis
            }
        },
        "comparison": comparison if pymupdf_docs and docling_docs else None,
        "summary": {
            "speed_winner": "PyMuPDF" if pymupdf_time < docling_time else "Docling",
            "pymupdf_faster_by": abs(docling_time - pymupdf_time),
            "table_extraction": {
                "pymupdf_tables": pymupdf_analysis.get("num_tables", 0),
                "docling_tables": docling_analysis.get("num_tables", 0),
            }
        }
    }

    # Save report
    report_file = output_dir / "comparison_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✓ Full report saved to {report_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nSpeed:")
    print(f"  PyMuPDF: {pymupdf_time:.2f}s")
    print(f"  Docling:  {docling_time:.2f}s")
    print(f"  Winner: {report['summary']['speed_winner']} "
          f"(faster by {report['summary']['pymupdf_faster_by']:.2f}s)")

    print(f"\nTable Extraction:")
    print(f"  PyMuPDF: {pymupdf_analysis.get('num_tables', 0)} tables")
    print(f"  Docling:  {docling_analysis.get('num_tables', 0)} tables")

    print(f"\nContent Quality:")
    print(f"  PyMuPDF: {pymupdf_analysis.get('total_chars', 0):,} chars")
    print(f"  Docling:  {docling_analysis.get('total_chars', 0):,} chars")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

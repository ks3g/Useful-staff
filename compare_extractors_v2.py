"""Compare PyMuPDF and Docling PDF extractors - IMPROVED VERSION.

Improvements over v1:
- ✅ Uses correct Docling API (v2)
- ✅ Adds memory profiling
- ✅ CLI arguments support
- ✅ Progress indicators
- ✅ Better error handling
- ✅ Type hints
- ✅ Configurable options
"""

import sys
import time
import json
import argparse
import tracemalloc
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Better imports - avoid sys.path hacks
try:
    from src.rag_system.extractors.pdf_extractor import PDFExtractor
    from src.rag_system.extractors.docling_extractor_v2 import DoclingExtractorV2
    from src.rag_system.base import Document
except ImportError:
    # Fallback for different execution contexts
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from src.rag_system.extractors.pdf_extractor import PDFExtractor
    from src.rag_system.extractors.docling_extractor_v2 import DoclingExtractorV2
    from src.rag_system.base import Document

# Optional: progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  Install tqdm for progress bars: pip install tqdm")


class ExtractionResult:
    """Results from PDF extraction with timing and memory info."""

    def __init__(
        self,
        documents: List[Document],
        extraction_time: float,
        memory_peak_mb: float,
        error: Optional[str] = None
    ):
        self.documents = documents
        self.extraction_time = extraction_time
        self.memory_peak_mb = memory_peak_mb
        self.error = error


def extract_with_profiling(
    extractor: Any,
    pdf_path: str,
    method_name: str,
    show_progress: bool = True
) -> ExtractionResult:
    """Extract PDF and measure time + memory.

    Args:
        extractor: Extractor instance
        pdf_path: Path to PDF file
        method_name: Name of extraction method
        show_progress: Show progress indicator

    Returns:
        ExtractionResult with documents, timing, and memory usage
    """
    print(f"\n{'='*60}")
    print(f"Extracting with {method_name}...")
    print(f"{'='*60}")

    # Start memory tracking
    tracemalloc.start()
    start_time = time.time()

    error = None
    documents = []

    try:
        if show_progress and HAS_TQDM:
            with tqdm(total=1, desc=f"{method_name} extraction") as pbar:
                documents = extractor.extract(pdf_path)
                pbar.update(1)
        else:
            documents = extractor.extract(pdf_path)

        extraction_time = time.time() - start_time

        # Get peak memory usage
        current, peak = tracemalloc.get_traced_memory()
        memory_peak_mb = peak / 1024 / 1024  # Convert to MB

        print(f"✓ Extracted {len(documents)} pages in {extraction_time:.2f}s")
        print(f"  Memory: {memory_peak_mb:.2f} MB (peak)")

    except Exception as e:
        extraction_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        memory_peak_mb = peak / 1024 / 1024

        error = str(e)
        print(f"✗ Error: {error}")
        print(f"  Traceback:\n{traceback.format_exc()}")

    finally:
        tracemalloc.stop()

    return ExtractionResult(
        documents=documents,
        extraction_time=extraction_time,
        memory_peak_mb=memory_peak_mb,
        error=error
    )


def analyze_documents(documents: List[Document]) -> Dict[str, Any]:
    """Analyze extracted documents."""
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

    metadata_fields = set()
    for doc in documents:
        metadata_fields.update(doc.metadata.keys())

    return {
        "num_documents": len(documents),
        "total_chars": total_chars,
        "avg_chars_per_doc": total_chars / len(documents) if documents else 0,
        "num_tables": num_tables,
        "pages_with_tables": pages_with_tables,
        "metadata_fields": sorted(list(metadata_fields)),
        "total_words": sum(len(doc.content.split()) for doc in documents),
    }


def compare_content(
    docs1: List[Document],
    docs2: List[Document],
    name1: str,
    name2: str
) -> Dict[str, Any]:
    """Compare content from two extractors with similarity metrics."""
    comparison = {
        "num_pages_match": len(docs1) == len(docs2),
        f"{name1}_pages": len(docs1),
        f"{name2}_pages": len(docs2),
    }

    if docs1 and docs2:
        content1 = docs1[0].content
        content2 = docs2[0].content

        # Calculate simple similarity (character overlap)
        common_chars = len(set(content1) & set(content2))
        total_chars = len(set(content1) | set(content2))
        char_similarity = common_chars / total_chars if total_chars > 0 else 0

        comparison["first_page_comparison"] = {
            f"{name1}_length": len(content1),
            f"{name2}_length": len(content2),
            "length_diff_pct": abs(len(content1) - len(content2)) / max(len(content1), len(content2), 1) * 100,
            "char_similarity": round(char_similarity, 3),
            f"{name1}_preview": content1[:200] + "..." if len(content1) > 200 else content1,
            f"{name2}_preview": content2[:200] + "..." if len(content2) > 200 else content2,
        }

    return comparison


def save_sample_pages(documents: List[Document], output_dir: Path, prefix: str, max_pages: int = 3):
    """Save sample pages for manual inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, doc in enumerate(documents[:max_pages], 1):
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
    parser = argparse.ArgumentParser(
        description="Compare PyMuPDF and Docling PDF extractors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        nargs="?",
        default="gerics_klimaausblick_hessen_version1.2_deutsch.pdf",
        help="Path to PDF file to process"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--sample-pages",
        type=int,
        default=3,
        help="Number of sample pages to save"
    )
    parser.add_argument(
        "--skip-pymupdf",
        action="store_true",
        help="Skip PyMuPDF extraction"
    )
    parser.add_argument(
        "--skip-docling",
        action="store_true",
        help="Skip Docling extraction"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("PDF EXTRACTOR COMPARISON: PyMuPDF vs Docling (v2)")
    print("=" * 60)
    print(f"PDF: {args.pdf_path}")
    print(f"Output: {args.output_dir}")

    if not Path(args.pdf_path).exists():
        print(f"❌ Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    show_progress = not args.no_progress

    # Extract with PyMuPDF
    pymupdf_result = None
    if not args.skip_pymupdf:
        pymupdf_extractor = PDFExtractor(
            extract_tables=True,
            extract_metadata=True,
            method="pymupdf"
        )
        pymupdf_result = extract_with_profiling(
            pymupdf_extractor, args.pdf_path, "PyMuPDF", show_progress
        )

    # Extract with Docling
    docling_result = None
    if not args.skip_docling:
        try:
            docling_extractor = DoclingExtractorV2(
                extract_tables=True,
                extract_metadata=True,
                preserve_structure=True
            )
            docling_result = extract_with_profiling(
                docling_extractor, args.pdf_path, "Docling", show_progress
            )
        except ImportError as e:
            print(f"⚠️  Docling not available: {e}")
            print("   Install with: pip install docling")

    # Analyze results
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    pymupdf_analysis = analyze_documents(pymupdf_result.documents) if pymupdf_result else {}
    docling_analysis = analyze_documents(docling_result.documents) if docling_result else {}

    if pymupdf_result:
        print("\n📊 PyMuPDF Analysis:")
        print(json.dumps(pymupdf_analysis, indent=2))

    if docling_result:
        print("\n📊 Docling Analysis:")
        print(json.dumps(docling_analysis, indent=2))

    # Compare
    comparison = None
    if pymupdf_result and docling_result and pymupdf_result.documents and docling_result.documents:
        print("\n" + "=" * 60)
        print("CONTENT COMPARISON")
        print("=" * 60)

        comparison = compare_content(
            pymupdf_result.documents, docling_result.documents,
            "PyMuPDF", "Docling"
        )
        print(json.dumps(comparison, indent=2))

    # Save sample pages
    if pymupdf_result:
        save_sample_pages(pymupdf_result.documents, output_dir, "pymupdf", args.sample_pages)
    if docling_result:
        save_sample_pages(docling_result.documents, output_dir, "docling", args.sample_pages)

    print(f"\n✓ Sample pages saved to {output_dir}/")

    # Generate summary report
    report = {
        "pdf_file": args.pdf_path,
        "extraction_results": {
            "pymupdf": {
                "time_seconds": pymupdf_result.extraction_time if pymupdf_result else None,
                "memory_mb": pymupdf_result.memory_peak_mb if pymupdf_result else None,
                "error": pymupdf_result.error if pymupdf_result else None,
                "analysis": pymupdf_analysis
            } if pymupdf_result else None,
            "docling": {
                "time_seconds": docling_result.extraction_time if docling_result else None,
                "memory_mb": docling_result.memory_peak_mb if docling_result else None,
                "error": docling_result.error if docling_result else None,
                "analysis": docling_analysis
            } if docling_result else None,
        },
        "comparison": comparison,
    }

    # Add summary if both succeeded
    if pymupdf_result and docling_result and not pymupdf_result.error and not docling_result.error:
        report["summary"] = {
            "speed_winner": "PyMuPDF" if pymupdf_result.extraction_time < docling_result.extraction_time else "Docling",
            "pymupdf_faster_by_sec": abs(docling_result.extraction_time - pymupdf_result.extraction_time),
            "pymupdf_faster_by_pct": ((docling_result.extraction_time - pymupdf_result.extraction_time) /
                                      pymupdf_result.extraction_time * 100),
            "memory_winner": "PyMuPDF" if pymupdf_result.memory_peak_mb < docling_result.memory_peak_mb else "Docling",
            "memory_diff_mb": abs(docling_result.memory_peak_mb - pymupdf_result.memory_peak_mb),
            "table_extraction": {
                "pymupdf_tables": pymupdf_analysis.get("num_tables", 0),
                "docling_tables": docling_analysis.get("num_tables", 0),
                "winner": "Docling" if docling_analysis.get("num_tables", 0) > pymupdf_analysis.get("num_tables", 0) else "PyMuPDF"
            }
        }

    # Save report
    report_file = output_dir / "comparison_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✓ Full report saved to {report_file}")

    # Print summary
    if pymupdf_result and docling_result:
        print("\n" + "=" * 60)
        print("📈 SUMMARY")
        print("=" * 60)
        print(f"\n⏱️  Speed:")
        print(f"  PyMuPDF: {pymupdf_result.extraction_time:.2f}s")
        print(f"  Docling:  {docling_result.extraction_time:.2f}s")
        if report.get("summary"):
            print(f"  Winner: {report['summary']['speed_winner']} "
                  f"({report['summary']['pymupdf_faster_by_pct']:.1f}% faster)")

        print(f"\n💾 Memory:")
        print(f"  PyMuPDF: {pymupdf_result.memory_peak_mb:.2f} MB")
        print(f"  Docling:  {docling_result.memory_peak_mb:.2f} MB")
        if report.get("summary"):
            print(f"  Winner: {report['summary']['memory_winner']} "
                  f"({report['summary']['memory_diff_mb']:.2f} MB difference)")

        print(f"\n📊 Table Extraction:")
        print(f"  PyMuPDF: {pymupdf_analysis.get('num_tables', 0)} tables")
        print(f"  Docling:  {docling_analysis.get('num_tables', 0)} tables")

        print(f"\n📝 Content Quality:")
        print(f"  PyMuPDF: {pymupdf_analysis.get('total_chars', 0):,} chars, {pymupdf_analysis.get('total_words', 0):,} words")
        print(f"  Docling:  {docling_analysis.get('total_chars', 0):,} chars, {docling_analysis.get('total_words', 0):,} words")

    print("\n" + "=" * 60)
    print("✅ Comparison complete!")


if __name__ == "__main__":
    main()

# How to Run PyMuPDF vs Docling Benchmarks

## Prerequisites

Wait for docling installation to complete:
```bash
# Check if installed
python3 -c "import docling; print('✓ Docling ready!')"
```

The installation is currently in progress (installing ~2-3 GB of dependencies including PyTorch and CUDA libraries).

## Running the Comparison

Once docling is installed, run:

```bash
python compare_extractors.py
```

This will:
1. Extract your PDF with both PyMuPDF and Docling
2. Measure extraction time for each
3. Analyze extraction quality (tables, structure, content)
4. Save sample pages to `comparison_results/`
5. Generate a detailed JSON report

## What You'll Get

### Console Output
- Extraction time for each method
- Number of pages extracted
- Table count
- Content statistics

### Files Created
- `comparison_results/pymupdf_page_1.txt` - PyMuPDF extraction sample
- `comparison_results/docling_page_1.txt` - Docling extraction sample
- `comparison_results/comparison_report.json` - Full benchmark data

## Quick Manual Test

If you want to test just one extractor:

### PyMuPDF Test
```python
from src.rag_system.extractors.pdf_extractor import PDFExtractor

extractor = PDFExtractor(method="pymupdf")
docs = extractor.extract("gerics_klimaausblick_hessen_version1.2_deutsch.pdf")

print(f"Extracted {len(docs)} pages")
print(f"First page preview: {docs[0].content[:200]}")
```

### Docling Test
```python
from src.rag_system.extractors.docling_extractor import DoclingExtractor

extractor = DoclingExtractor()
docs = extractor.extract("gerics_klimaausblick_hessen_version1.2_deutsch.pdf")

print(f"Extracted {len(docs)} pages")
print(f"First page preview: {docs[0].content[:200]}")
```

## Expected Results

Based on the analysis in `COMPARISON_PYMUPDF_VS_DOCLING.md`:

- **PyMuPDF**: Faster (~1-2 sec), less accurate structure
- **Docling**: Slower (~10-20 sec), superior quality for RAG

## Recommendation

**Use Docling** for your RAG system because:
1. Your PDF has complex layouts and tables
2. Quality > Speed for document indexing
3. Better retrieval accuracy = Better answers
4. One-time cost (indexing) vs continuous benefit (queries)

## Troubleshooting

### If docling import fails:
```bash
# Check installation status
pip list | grep docling

# If not installed, wait or restart:
pip install docling
```

### If comparison script fails:
```bash
# Install missing dependencies
pip install pymupdf pdfplumber
```

## Next Steps After Benchmarks

1. Review the comparison report
2. Check sample extractions in `comparison_results/`
3. Decide on final extractor choice
4. Integrate into your RAG pipeline
5. Enjoy better document understanding! 🚀

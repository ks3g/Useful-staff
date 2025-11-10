"""PDF extraction using Docling - CORRECTED VERSION based on actual API."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.document import TextItem, TableItem
except ImportError:
    print("Docling not installed. Install with: pip install docling")
    sys.exit(1)

from ..base import BaseExtractor, Document
from ...utils.logger import get_logger

logger = get_logger()


class DoclingExtractorV2(BaseExtractor):
    """Extract text and metadata from PDF files using Docling (Corrected API).

    Based on ACTUAL Docling documentation:
    - github.com/docling-project/docling
    - docling-project.github.io/docling

    Key differences from v1:
    - Uses result.document.iterate_items() for proper iteration
    - Uses result.pages for page-level extraction
    - Correctly handles TableItem.export_to_dataframe()
    - Properly structures metadata
    """

    def __init__(
        self,
        extract_images: bool = False,
        extract_tables: bool = True,
        extract_metadata: bool = True,
        preserve_structure: bool = True,
    ):
        """Initialize Docling extractor.

        Args:
            extract_images: Whether to extract images
            extract_tables: Whether to extract tables
            extract_metadata: Whether to extract document metadata
            preserve_structure: Whether to preserve document structure
        """
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.extract_metadata = extract_metadata
        self.preserve_structure = preserve_structure
        self.converter = DocumentConverter()

    def extract(self, file_path: str) -> List[Document]:
        """Extract documents from PDF file using Docling.

        Args:
            file_path: Path to PDF file

        Returns:
            List of extracted documents (one per page)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        logger.info(f"Extracting PDF with Docling: {file_path}")

        try:
            # Convert document - this is the correct API
            conv_result = self.converter.convert(str(file_path))

            documents = []

            # Extract document-level metadata
            doc_metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "extraction_method": "docling_v2",
                "status": str(conv_result.status) if hasattr(conv_result, 'status') else None,
            }

            # CORRECT: conv_result has .pages attribute
            if hasattr(conv_result, 'pages') and conv_result.pages:
                logger.info(f"Processing {len(conv_result.pages)} pages")

                for page_num, page in enumerate(conv_result.pages, 1):
                    page_content = []
                    page_tables = []

                    page_metadata = {
                        **doc_metadata,
                        "page": page_num,
                        "page_number": page_num,
                    }

                    # Extract page dimensions if available
                    if hasattr(page, 'size'):
                        page_metadata["page_width"] = page.size.width if hasattr(page.size, 'width') else None
                        page_metadata["page_height"] = page.size.height if hasattr(page.size, 'height') else None

                    # Collect content from this page
                    # Note: We need to use document.iterate_items() and filter by page
                    for item, level in conv_result.document.iterate_items():
                        # Check if item belongs to this page
                        item_page = getattr(item, 'page', None) or getattr(item, 'page_no', None)
                        if item_page != page_num:
                            continue

                        if isinstance(item, TextItem):
                            # Add text with hierarchy level
                            indent = "  " * level
                            page_content.append(f"{indent}{item.text}")

                        elif isinstance(item, TableItem) and self.extract_tables:
                            # Extract table to DataFrame
                            try:
                                table_df = item.export_to_dataframe()
                                table_md = table_df.to_markdown(index=False)
                                page_content.append(f"\n{table_md}\n")

                                # Store table metadata
                                page_tables.append({
                                    "table_id": len(page_tables) + 1,
                                    "rows": len(table_df),
                                    "cols": len(table_df.columns),
                                    "data": table_df.to_dict('records')
                                })
                            except Exception as e:
                                logger.warning(f"Failed to export table on page {page_num}: {e}")

                    # Combine content
                    content = "\n".join(page_content)

                    # Add table metadata
                    if page_tables:
                        page_metadata["tables"] = page_tables
                        page_metadata["num_tables"] = len(page_tables)

                    if content.strip():
                        documents.append(Document(
                            content=content,
                            metadata=page_metadata,
                            doc_id=f"{file_path.stem}_page_{page_num}"
                        ))

            # Fallback: Use full document if pages not available
            else:
                logger.warning("No pages found, using full document")
                content_parts = []
                all_tables = []

                for item, level in conv_result.document.iterate_items():
                    if isinstance(item, TextItem):
                        content_parts.append(item.text)
                    elif isinstance(item, TableItem) and self.extract_tables:
                        try:
                            table_df = item.export_to_dataframe()
                            content_parts.append(table_df.to_markdown(index=False))
                            all_tables.append({
                                "table_id": len(all_tables) + 1,
                                "data": table_df.to_dict('records')
                            })
                        except Exception as e:
                            logger.warning(f"Failed to export table: {e}")

                content = "\n\n".join(content_parts)

                if all_tables:
                    doc_metadata["tables"] = all_tables
                    doc_metadata["num_tables"] = len(all_tables)

                if content.strip():
                    documents.append(Document(
                        content=content,
                        metadata=doc_metadata,
                        doc_id=f"{file_path.stem}_full"
                    ))

            logger.info(f"Extracted {len(documents)} documents from {file_path.name}")

        except Exception as e:
            logger.error(f"Error extracting PDF with Docling: {e}", exc_info=True)
            raise

        return documents

    def export_to_markdown(self, file_path: str) -> str:
        """Export PDF directly to Markdown using Docling.

        Args:
            file_path: Path to PDF file

        Returns:
            Markdown string
        """
        conv_result = self.converter.convert(str(file_path))
        return conv_result.document.export_to_markdown()

    def export_to_html(self, file_path: str) -> str:
        """Export PDF directly to HTML using Docling.

        Args:
            file_path: Path to PDF file

        Returns:
            HTML string
        """
        conv_result = self.converter.convert(str(file_path))
        return conv_result.document.export_to_html()

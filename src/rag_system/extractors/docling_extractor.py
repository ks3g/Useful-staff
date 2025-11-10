"""PDF extraction using Docling - IBM's RAG-optimized library."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

try:
    from docling.document_converter import DocumentConverter
except ImportError:
    print("Docling not installed. Install with: pip install docling")
    sys.exit(1)

from ..base import BaseExtractor, Document
from ...utils.logger import get_logger

logger = get_logger()


class DoclingExtractor(BaseExtractor):
    """Extract text and metadata from PDF files using Docling.

    Docling is IBM's library specifically designed for document understanding
    and RAG applications. It provides:
    - Superior table extraction
    - Document structure preservation
    - Semantic-aware chunking
    - Better handling of complex layouts
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
            preserve_structure: Whether to preserve document structure (sections, paragraphs)
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
            List of extracted documents
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        logger.info(f"Extracting PDF with Docling: {file_path}")

        try:
            # Convert document
            result = self.converter.convert(str(file_path))

            documents = []

            # Extract document-level metadata
            doc_metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "extraction_method": "docling",
            }

            if self.extract_metadata and hasattr(result, 'metadata'):
                doc_metadata.update({
                    "title": getattr(result.metadata, 'title', ''),
                    "author": getattr(result.metadata, 'author', ''),
                    "num_pages": getattr(result.metadata, 'num_pages', 0),
                })

            # Process pages
            if hasattr(result, 'pages'):
                for page_num, page in enumerate(result.pages, 1):
                    content_parts = []
                    page_metadata = {
                        **doc_metadata,
                        "page": page_num,
                    }

                    # Extract structured content
                    if self.preserve_structure:
                        # Docling maintains document structure
                        content_parts.append(self._extract_structured_content(page))
                    else:
                        # Simple text extraction
                        content_parts.append(page.get_text() if hasattr(page, 'get_text') else str(page))

                    # Extract tables
                    if self.extract_tables and hasattr(page, 'tables'):
                        tables = self._extract_tables(page)
                        if tables:
                            page_metadata["tables"] = tables
                            content_parts.append(self._format_tables_as_text(tables))

                    content = "\n\n".join(filter(None, content_parts))

                    if content.strip():
                        documents.append(Document(
                            content=content,
                            metadata=page_metadata,
                            doc_id=f"{file_path.stem}_page_{page_num}"
                        ))

            # Fallback: if no pages, try to get full document text
            elif hasattr(result, 'document'):
                doc = result.document
                content = str(doc)

                if content.strip():
                    documents.append(Document(
                        content=content,
                        metadata=doc_metadata,
                        doc_id=f"{file_path.stem}_full"
                    ))

            logger.info(f"Extracted {len(documents)} pages from {file_path.name} using Docling")

        except Exception as e:
            logger.error(f"Error extracting PDF with Docling: {e}")
            raise

        return documents

    def _extract_structured_content(self, page) -> str:
        """Extract content while preserving document structure.

        Args:
            page: Docling page object

        Returns:
            Structured text content
        """
        # Docling preserves sections, headers, paragraphs, lists
        if hasattr(page, 'get_structured_text'):
            return page.get_structured_text()
        elif hasattr(page, 'get_text'):
            return page.get_text()
        else:
            return str(page)

    def _extract_tables(self, page) -> List[Dict[str, Any]]:
        """Extract tables from page.

        Args:
            page: Docling page object

        Returns:
            List of table dictionaries
        """
        tables = []

        if hasattr(page, 'tables'):
            for i, table in enumerate(page.tables, 1):
                table_data = {
                    "table_id": i,
                    "data": []
                }

                # Extract table structure
                if hasattr(table, 'to_dict'):
                    table_data["data"] = table.to_dict()
                elif hasattr(table, 'rows'):
                    table_data["rows"] = len(table.rows)
                    table_data["data"] = [
                        [cell.text if hasattr(cell, 'text') else str(cell)
                         for cell in row]
                        for row in table.rows
                    ]
                else:
                    table_data["data"] = str(table)

                tables.append(table_data)

        return tables

    def _format_tables_as_text(self, tables: List[Dict[str, Any]]) -> str:
        """Format tables as text for inclusion in content.

        Args:
            tables: List of table dictionaries

        Returns:
            Text representation of tables
        """
        text_parts = []

        for table in tables:
            text_parts.append(f"\n[Table {table['table_id']}]")

            if isinstance(table['data'], list):
                for row in table['data']:
                    if isinstance(row, list):
                        row_text = " | ".join([str(cell) for cell in row])
                        text_parts.append(row_text)
            else:
                text_parts.append(str(table['data']))

        return "\n".join(text_parts)

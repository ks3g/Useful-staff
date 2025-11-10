"""PDF extraction using multiple libraries for robust text extraction."""

import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..base import BaseExtractor, Document
from ...utils.logger import get_logger

logger = get_logger()


class PDFExtractor(BaseExtractor):
    """Extract text and metadata from PDF files using PyMuPDF and PDFPlumber."""

    def __init__(
        self,
        extract_images: bool = False,
        extract_tables: bool = True,
        extract_metadata: bool = True,
        method: str = "pymupdf"  # Options: pymupdf, pdfplumber, hybrid
    ):
        """Initialize PDF extractor.

        Args:
            extract_images: Whether to extract images
            extract_tables: Whether to extract tables
            extract_metadata: Whether to extract document metadata
            method: Extraction method to use
        """
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.extract_metadata = extract_metadata
        self.method = method

    def extract(self, file_path: str) -> List[Document]:
        """Extract documents from PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            List of extracted documents (one per page)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        logger.info(f"Extracting PDF: {file_path}")

        if self.method == "pymupdf":
            return self._extract_with_pymupdf(file_path)
        elif self.method == "pdfplumber":
            return self._extract_with_pdfplumber(file_path)
        elif self.method == "hybrid":
            return self._extract_hybrid(file_path)
        else:
            raise ValueError(f"Unknown extraction method: {self.method}")

    def _extract_with_pymupdf(self, file_path: Path) -> List[Document]:
        """Extract using PyMuPDF (faster, good for standard PDFs).

        Args:
            file_path: Path to PDF file

        Returns:
            List of documents
        """
        documents = []

        try:
            pdf_document = fitz.open(str(file_path))

            # Extract document-level metadata
            doc_metadata = {}
            if self.extract_metadata:
                doc_metadata = {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "total_pages": len(pdf_document),
                    "title": pdf_document.metadata.get("title", ""),
                    "author": pdf_document.metadata.get("author", ""),
                    "subject": pdf_document.metadata.get("subject", ""),
                    "creator": pdf_document.metadata.get("creator", ""),
                    "producer": pdf_document.metadata.get("producer", ""),
                    "creation_date": pdf_document.metadata.get("creationDate", ""),
                }

            # Extract text from each page
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                text = page.get_text()

                if text.strip():  # Only add non-empty pages
                    page_metadata = {
                        **doc_metadata,
                        "page": page_num + 1,
                        "extraction_method": "pymupdf"
                    }

                    # Extract tables if requested
                    if self.extract_tables:
                        tables = self._extract_tables_pymupdf(page)
                        if tables:
                            page_metadata["tables"] = tables

                    documents.append(Document(
                        content=text,
                        metadata=page_metadata,
                        doc_id=f"{file_path.stem}_page_{page_num + 1}"
                    ))

            pdf_document.close()
            logger.info(f"Extracted {len(documents)} pages from {file_path.name}")

        except Exception as e:
            logger.error(f"Error extracting PDF with PyMuPDF: {e}")
            raise

        return documents

    def _extract_with_pdfplumber(self, file_path: Path) -> List[Document]:
        """Extract using PDFPlumber (better for tables and complex layouts).

        Args:
            file_path: Path to PDF file

        Returns:
            List of documents
        """
        documents = []

        try:
            with pdfplumber.open(str(file_path)) as pdf:
                # Extract document-level metadata
                doc_metadata = {}
                if self.extract_metadata:
                    doc_metadata = {
                        "source": str(file_path),
                        "filename": file_path.name,
                        "total_pages": len(pdf.pages),
                        "title": pdf.metadata.get("Title", ""),
                        "author": pdf.metadata.get("Author", ""),
                        "subject": pdf.metadata.get("Subject", ""),
                        "creator": pdf.metadata.get("Creator", ""),
                        "producer": pdf.metadata.get("Producer", ""),
                    }

                # Extract text from each page
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()

                    if text and text.strip():
                        page_metadata = {
                            **doc_metadata,
                            "page": page_num + 1,
                            "extraction_method": "pdfplumber"
                        }

                        # Extract tables if requested
                        if self.extract_tables:
                            tables = page.extract_tables()
                            if tables:
                                page_metadata["tables"] = self._format_tables(tables)
                                # Append table content to text
                                text += "\n\n" + self._tables_to_text(tables)

                        documents.append(Document(
                            content=text,
                            metadata=page_metadata,
                            doc_id=f"{file_path.stem}_page_{page_num + 1}"
                        ))

            logger.info(f"Extracted {len(documents)} pages from {file_path.name}")

        except Exception as e:
            logger.error(f"Error extracting PDF with PDFPlumber: {e}")
            raise

        return documents

    def _extract_hybrid(self, file_path: Path) -> List[Document]:
        """Use hybrid approach: PyMuPDF for text, PDFPlumber for tables.

        Args:
            file_path: Path to PDF file

        Returns:
            List of documents
        """
        documents = []

        try:
            # Use PyMuPDF for basic extraction
            pdf_mupdf = fitz.open(str(file_path))

            # Use PDFPlumber for tables
            with pdfplumber.open(str(file_path)) as pdf_plumber:
                doc_metadata = {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "total_pages": len(pdf_mupdf),
                }

                for page_num in range(len(pdf_mupdf)):
                    # Extract text with PyMuPDF
                    page_mupdf = pdf_mupdf[page_num]
                    text = page_mupdf.get_text()

                    if text.strip():
                        page_metadata = {
                            **doc_metadata,
                            "page": page_num + 1,
                            "extraction_method": "hybrid"
                        }

                        # Extract tables with PDFPlumber
                        if self.extract_tables and page_num < len(pdf_plumber.pages):
                            page_plumber = pdf_plumber.pages[page_num]
                            tables = page_plumber.extract_tables()
                            if tables:
                                page_metadata["tables"] = self._format_tables(tables)
                                text += "\n\n" + self._tables_to_text(tables)

                        documents.append(Document(
                            content=text,
                            metadata=page_metadata,
                            doc_id=f"{file_path.stem}_page_{page_num + 1}"
                        ))

            pdf_mupdf.close()
            logger.info(f"Extracted {len(documents)} pages from {file_path.name} (hybrid)")

        except Exception as e:
            logger.error(f"Error extracting PDF with hybrid method: {e}")
            raise

        return documents

    def _extract_tables_pymupdf(self, page) -> List[Dict[str, Any]]:
        """Extract tables from a PyMuPDF page.

        Args:
            page: PyMuPDF page object

        Returns:
            List of table dictionaries
        """
        # PyMuPDF doesn't have built-in table extraction
        # This is a placeholder for future enhancement
        return []

    def _format_tables(self, tables: List[List[List[str]]]) -> List[Dict[str, Any]]:
        """Format extracted tables.

        Args:
            tables: Raw table data

        Returns:
            Formatted table dictionaries
        """
        formatted_tables = []
        for i, table in enumerate(tables):
            if table and len(table) > 0:
                formatted_tables.append({
                    "table_id": i + 1,
                    "rows": len(table),
                    "cols": len(table[0]) if table else 0,
                    "data": table
                })
        return formatted_tables

    def _tables_to_text(self, tables: List[List[List[str]]]) -> str:
        """Convert tables to text format.

        Args:
            tables: List of tables

        Returns:
            Text representation of tables
        """
        text_parts = []
        for i, table in enumerate(tables):
            text_parts.append(f"\n[Table {i + 1}]")
            for row in table:
                if row:
                    # Filter out None values and join
                    row_text = " | ".join([str(cell) if cell else "" for cell in row])
                    text_parts.append(row_text)
        return "\n".join(text_parts)

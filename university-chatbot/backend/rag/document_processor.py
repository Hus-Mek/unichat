"""Document text extraction for PDF, Excel, and Word files."""

import io

import pandas as pd
from pypdf import PdfReader


def extract_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def extract_excel(file_bytes: bytes) -> str:
    """Extract text from an Excel file, preserving sheet and row structure."""
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    parts: list[str] = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        if df.empty:
            continue

        parts.append(f"--- Sheet: {sheet_name} ---")
        headers = list(df.columns)

        for _, row in df.iterrows():
            cells = [
                f"{col}: {row[col]}"
                for col in headers
                if pd.notna(row[col])
            ]
            if cells:
                parts.append(" | ".join(cells))

    return "\n".join(parts)


def extract_docx(file_bytes: bytes) -> str:
    """Extract text from a Word (.docx) file."""
    from docx import Document

    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


EXTRACTORS = {
    ".pdf": extract_pdf,
    ".xlsx": extract_excel,
    ".xls": extract_excel,
    ".docx": extract_docx,
}

SUPPORTED_EXTENSIONS = set(EXTRACTORS.keys())


def process_document(file_bytes: bytes, file_extension: str) -> str:
    """
    Extract text from a document based on its file extension.

    Raises ValueError for unsupported file types.
    """
    ext = file_extension.lower()
    if ext not in EXTRACTORS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")
    return EXTRACTORS[ext](file_bytes)

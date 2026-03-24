"""Tests for document processing."""

from backend.rag.document_processor import SUPPORTED_EXTENSIONS, process_document
import pytest


def test_supported_extensions():
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".xlsx" in SUPPORTED_EXTENSIONS
    assert ".xls" in SUPPORTED_EXTENSIONS
    assert ".docx" in SUPPORTED_EXTENSIONS


def test_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file type"):
        process_document(b"data", ".txt")


def test_unsupported_extension_message():
    with pytest.raises(ValueError) as exc_info:
        process_document(b"data", ".mp3")
    assert ".mp3" in str(exc_info.value)

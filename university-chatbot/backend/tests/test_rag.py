"""Tests for the RAG chunker."""

from backend.rag.chunker import chunk_text


def test_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   ") == []
    assert chunk_text(None) == []


def test_short_text():
    text = "Hello world"
    chunks = chunk_text(text, chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0] == "Hello world"


def test_chunking_produces_overlap():
    # Create text that should produce at least 2 chunks
    text = "A" * 500 + "\n\n" + "B" * 500 + "\n\n" + "C" * 500
    chunks = chunk_text(text, chunk_size=600, chunk_overlap=100)
    assert len(chunks) >= 2

    # Check overlap exists: the end of chunk N should appear at the start of chunk N+1
    for i in range(len(chunks) - 1):
        tail = chunks[i][-100:]
        assert chunks[i + 1].startswith(tail) or len(tail.strip()) == 0


def test_chunk_size_respected():
    text = "word " * 1000  # 5000 chars
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    # Chunks should be roughly within bounds (with some tolerance for overlap)
    for chunk in chunks:
        assert len(chunk) <= 600  # some tolerance


def test_paragraph_boundaries():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = chunk_text(text, chunk_size=1000)
    assert len(chunks) == 1  # Should fit in a single chunk
    assert "Paragraph one." in chunks[0]
    assert "Paragraph three." in chunks[0]

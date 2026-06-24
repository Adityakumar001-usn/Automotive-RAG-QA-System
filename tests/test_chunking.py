from src.chunking import fixed_chunking, overlap_chunking, recursive_chunking
import pytest

def test_fixed_chunking():
    text = "A" * 1000
    chunks = fixed_chunking(text, chunk_size=500)
    assert len(chunks) == 2
    assert len(chunks[0]) == 500

def test_overlap_chunking():
    text = "A" * 1000
    chunks = overlap_chunking(text, chunk_size=500, chunk_overlap=100)
    # (1000 - 500) / 400 = 1.something -> 3 chunks
    assert len(chunks) == 3

def test_recursive_chunking():
    text = "Paragraph one.\n\nParagraph two."
    chunks = recursive_chunking(text, chunk_size=15)
    # Should split by paragraph
    assert "Paragraph one." in chunks
    assert "Paragraph two." in chunks

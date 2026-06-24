import pytest
import os
from src.ingestion import ingest_document

def test_extract_txt():
    text, pages = ingest_document("test_assets/sample.txt")
    assert "Test TXT Document" in text
    assert pages == 1

def test_extract_pdf():
    text, pages = ingest_document("test_assets/sample.pdf")
    assert "Test PDF Document" in text
    assert pages == 1

def test_extract_csv():
    text, pages = ingest_document("test_assets/sample.csv")
    assert "Value1" in text
    assert pages == 1

def test_extract_docx():
    text, pages = ingest_document("test_assets/sample.docx")
    assert "Test DOCX Document" in text
    assert pages == 1

def test_extract_xlsx():
    text, pages = ingest_document("test_assets/sample.xlsx")
    assert "Header1" in text
    assert pages == 1

def test_extract_jpg():
    text, pages = ingest_document("test_assets/sample.jpg")
    # OCR might not be perfect with generic fonts, but should extract something or just not crash
    assert isinstance(text, str)
    assert pages == 1

def test_ingest_missing_file():
    with pytest.raises(FileNotFoundError):
        ingest_document("test_assets/does_not_exist.txt")

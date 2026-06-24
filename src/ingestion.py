import os
import fitz  # PyMuPDF
import pandas as pd
from docx import Document
import openpyxl
from PIL import Image
import pytesseract
from typing import Dict, Any, Tuple
from src.utils import get_logger, clean_text

logger = get_logger(__name__)

def extract_pdf(file_path: str) -> Tuple[str, int]:
    """
    Extracts text from a PDF file.

    Why it exists:
    PDFs are standard in automotive repair manuals. We need to convert visual pages into raw text strings
    so the AI can read them later.

    Inputs:
    file_path (str): The exact location of the PDF file on the computer.

    Outputs:
    Tuple[str, int]: A package containing the full text combined into one long string, and the total number of pages.
    """
    text = ""
    try:
        # Open the PDF using the PyMuPDF library (fitz). It's fast and handles complex layouts well.
        doc = fitz.open(file_path)
        num_pages = len(doc)

        # Go through the document one page at a time.
        for page in doc:
            page_text = page.get_text("text") # Pull out just the readable text
            if page_text:
                text += page_text + "\n" # Add it to our running total with a newline for spacing

        doc.close() # Always close files to free up computer memory
        return text, num_pages
    except Exception as e:
        logger.error(f"Error extracting PDF {file_path}: {e}")
        raise

def extract_csv(file_path: str) -> Tuple[str, int]:
    """
    Extracts text from a CSV file.

    Why it exists:
    Automotive data often comes in tabular formats (like parts lists). We need to read these tables
    and flatten them into strings so the LLM can interpret them.

    Inputs:
    file_path (str): The exact location of the CSV file.
    Outputs:
    Tuple[str, int]: A string of all row data, and a page count of 1 (since CSVs aren't paginated).
    """
    try:
        df = pd.read_csv(file_path)
        # Convert all rows to a string representation
        text = df.to_string(index=False)
        return text, 1
    except Exception as e:
        logger.error(f"Error extracting CSV {file_path}: {e}")
        raise

def extract_docx(file_path: str) -> Tuple[str, int]:
    """
    Extracts text from a DOCX file.

    Why it exists:
    Used to parse Microsoft Word files like generated maintenance schedules.
    """
    try:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text, 1
    except Exception as e:
        logger.error(f"Error extracting DOCX {file_path}: {e}")
        raise

def extract_xlsx(file_path: str) -> Tuple[str, int]:
    """
    Extracts text from an XLSX file using pandas.

    Why it exists:
    Multi-sheet Excel files are common in large repair matrices. We extract all sheets into a single long string.
    """
    try:
        # Read all sheets
        dict_df = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
        text = ""
        for sheet_name, df in dict_df.items():
            text += f"--- Sheet: {sheet_name} ---\n"
            text += df.to_string(index=False) + "\n"
        return text, 1
    except Exception as e:
        logger.error(f"Error extracting XLSX {file_path}: {e}")
        raise

def extract_txt(file_path: str) -> Tuple[str, int]:
    """
    Extracts text from a TXT file.

    Why it exists:
    For simple, raw text diagnostic logs.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text, 1
    except Exception as e:
        logger.error(f"Error extracting TXT {file_path}: {e}")
        raise

def extract_image_ocr(file_path: str) -> Tuple[str, int]:
    """Extracts text from an image using Tesseract OCR."""
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return text, 1
    except pytesseract.TesseractNotFoundError:
        error_msg = "Tesseract OCR is not installed or not in your PATH. Please install it (e.g., 'sudo apt-get install tesseract-ocr' on Ubuntu or via installer on Windows)."
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    except Exception as e:
        logger.error(f"Error extracting Image OCR {file_path}: {e}")
        raise

def ingest_document(file_path: str) -> Tuple[str, int]:
    """
    Ingests a document based on its extension.
    Returns the extracted raw text and the number of pages (or 1 for non-paginated formats).
    """
    logger.info(f"[PHASE 1] Dataset Ingestion - Processing document: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_pdf(file_path)
    elif ext == ".csv":
        return extract_csv(file_path)
    elif ext == ".docx":
        return extract_docx(file_path)
    elif ext == ".xlsx":
        return extract_xlsx(file_path)
    elif ext == ".txt":
        return extract_txt(file_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        return extract_image_ocr(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

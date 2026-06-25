import logging
import re
import unicodedata

def get_logger(name: str) -> logging.Logger:
    """
    Creates and configures a generic logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def clean_text(text: str) -> str:
    """
    Cleans raw text extracted from documents to make it readable for the AI.

    Why it exists:
    PDFs and documents contain weird symbols, random page numbers like "Page 1 of 50",
    and giant blank spaces. If we don't remove them, the AI gets confused.

    How it works:
    1. Normalizes weird Unicode symbols (like converting fancy quotes to standard quotes).
    2. Uses regular expressions (regex) to search for and delete page number patterns.
    3. Shrinks multiple spaces and multiple blank lines down to single spaces/newlines.
    """
    if not text:
        return ""

    # Normalize unicode (e.g., replace smart quotes, normalize spaces)
    text = unicodedata.normalize("NFKC", text)

    lines = text.split('\n')
    cleaned_lines = []

    # Very basic header/footer removal: if lines are repeated at top/bottom across chunks
    # This is a bit tricky on raw text without page boundaries.
    # We will mainly rely on regex for common page number patterns.

    # Regex to catch "Page X", "Page X of Y", or lines that are just numbers
    page_num_pattern = re.compile(r'^\s*(page\s+\d+(\s+of\s+\d+)?|\d+)\s*$', re.IGNORECASE)

    for line in lines:
        if page_num_pattern.match(line):
            continue
        cleaned_lines.append(line)

    cleaned_text = '\n'.join(cleaned_lines)

    # Remove multiple spaces
    cleaned_text = re.compile(r'[ \t]+').sub(' ', cleaned_text)

    # Remove duplicate blank lines
    cleaned_text = re.compile(r'\n{3,}').sub('\n\n', cleaned_text)

    return cleaned_text.strip()

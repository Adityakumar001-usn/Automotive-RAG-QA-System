import re
from typing import List
from src.utils import get_logger

logger = get_logger(__name__)

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def fixed_chunking(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Splits text into fixed size chunks (by character count).
    """
    if not text:
        return []

    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def overlap_chunking(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Splits text into overlapping chunks (by character count).
    """
    if not text:
        return []

    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    step = chunk_size - chunk_overlap
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), step)]
    return chunks

def _split_by_regex(text: str, pattern: str) -> List[str]:
    """Helper to split text by regex, preserving the delimiter if possible, and cleaning empty strings."""
    # Split by pattern. We use re.split with a capture group if we wanted to keep delimiters.
    # For simplicity, we split and then strip empty spaces.
    splits = re.split(pattern, text)
    return [s.strip() for s in splits if s.strip()]

def recursive_chunking(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Recursive chunking using native Python and regex.
    Priority: Paragraph -> Sentence -> Word
    """
    if not text:
        return []

    def _recursive_split(sub_text: str, level: int) -> List[str]:
        # Levels: 0 = Paragraph, 1 = Sentence, 2 = Word, 3 = Characters
        if len(sub_text) <= chunk_size:
            return [sub_text]

        if level == 0:
            # Split by double newline (Paragraphs)
            splits = _split_by_regex(sub_text, r'\n\s*\n')
        elif level == 1:
            # Split by sentence boundaries (. ? !) followed by space
            splits = _split_by_regex(sub_text, r'(?<=[.!?])\s+')
        elif level == 2:
            # Split by whitespace (Words)
            splits = _split_by_regex(sub_text, r'\s+')
        else:
            # Fallback to fixed chunking
            return fixed_chunking(sub_text, chunk_size)

        # Recombine splits greedily up to chunk_size
        chunks = []
        current_chunk = ""

        # If the level was unable to produce more than 1 split (e.g. no paragraphs), go deeper
        if len(splits) == 1 and level < 3:
            return _recursive_split(sub_text, level + 1)

        separator = "\n\n" if level == 0 else (" " if level in [1, 2] else "")

        for s in splits:
            if not current_chunk:
                if len(s) > chunk_size:
                    # Element itself is too large, recurse deeper on it
                    chunks.extend(_recursive_split(s, level + 1))
                else:
                    current_chunk = s
            else:
                new_chunk = current_chunk + separator + s
                if len(new_chunk) <= chunk_size:
                    current_chunk = new_chunk
                else:
                    chunks.append(current_chunk)
                    if len(s) > chunk_size:
                        # The single split is larger than max_size, need to go deeper
                        chunks.extend(_recursive_split(s, level + 1))
                        current_chunk = ""
                    else:
                        current_chunk = s

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    return _recursive_split(text, 0)

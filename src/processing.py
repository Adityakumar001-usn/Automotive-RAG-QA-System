from typing import Dict, Any, Tuple
from src.ingestion import ingest_document
from src.utils import get_logger, clean_text
from src.metadata_generator import generate_metadata

logger = get_logger(__name__)

def process_document(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Acts as the master controller for Phase 1 (Ingesting a single file).

    Why it exists:
    We need one unified function that takes a raw file on disk and runs it through
    the entire ingestion pipeline (Reading -> Cleaning -> Tagging) in the correct order.

    Inputs:
    file_path (str): The exact file location.

    Outputs:
    Tuple[str, Dict]: The fully cleaned text ready for chunking, and the metadata dictionary.
    """
    import time
    start = time.time()
    logger.info(f"[PHASE 2] Document Processing - Starting processing pipeline for: {file_path}")

    # 1. Ingestion
    raw_text, num_pages = ingest_document(file_path)

    # 2. Cleaning
    cleaned_text = clean_text(raw_text)

    # 3. Metadata Generation
    metadata = generate_metadata(file_path, num_pages=num_pages, num_chunks=0)

    logger.info(f"[PHASE 2] Document Processing - Completed in {time.time()-start:.2f}s for: {file_path}")
    return cleaned_text, metadata

from typing import Dict, Any, Tuple
from src.ingestion import ingest_document
from src.utils import get_logger, clean_text
from src.metadata_generator import generate_metadata

logger = get_logger(__name__)

def process_document(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Full processing pipeline for a single document:
    1. Ingestion (extraction)
    2. Cleaning
    3. Metadata Generation
    Returns the cleaned text and the generated metadata.
    Note: num_chunks is initialized to 0 and should be updated after chunking.
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

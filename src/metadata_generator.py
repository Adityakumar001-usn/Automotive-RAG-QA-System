import os
import time
from typing import Dict, Any
from src.utils import get_logger

logger = get_logger(__name__)

def generate_metadata(file_path: str, num_pages: int = 1, num_chunks: int = 0) -> Dict[str, Any]:
    """
    Auto-infers metadata from the file path.
    Rules:
    - directory dictates category
    - filename dictates document_name and document_type (extension)
    """
    file_path = os.path.normpath(file_path)
    parts = file_path.split(os.sep)

    # Determine category based on the immediate parent directory if it's not the root
    # e.g., data/service_manuals/doc.pdf -> category = service_manuals (then we normalize)
    parent_dir = parts[-2] if len(parts) > 1 else ""

    category_map = {
        "service_manuals": "service_manual",
        "repair_procedures": "repair_procedure",
        "diagnostic_flowcharts": "diagnostic_flowchart",
        "wiring_descriptions": "wiring_description",
        "maintenance_schedules": "maintenance_schedule",
        "tsb": "technical_service_bulletin"
    }

    category = category_map.get(parent_dir, "unknown")

    filename = os.path.basename(file_path)
    document_name, extension = os.path.splitext(filename)
    document_type = extension.lstrip('.').lower()

    metadata = {
        "document_id": filename, # Using filename as a simple ID
        "document_name": document_name,
        "document_type": document_type,
        "category": category,
        "source": file_path,
        "num_pages": num_pages,
        "num_chunks": num_chunks,
        "ingestion_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    logger.info(f"Generated metadata for {filename}: category={category}")
    return metadata

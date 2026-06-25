import os
import time
from typing import Dict, Any
from src.utils import get_logger

logger = get_logger(__name__)

def generate_metadata(file_path: str, num_pages: int = 1, num_chunks: int = 0) -> Dict[str, Any]:
    """
    Auto-infers tags and categorization information (metadata) from the file path.

    Why it exists:
    When an AI retrieves a chunk of text (like "Change the oil every 5,000 miles"), we need to know
    where that fact came from so we can provide a source citation to the user.

    How it works:
    1. Looks at the folder the file is inside (e.g., if it's in a 'tsb' folder, it tags it as a TSB).
    2. If the folder is unknown, it reads the filename itself (e.g., 'engine_manual.pdf').
    3. It generates a unique fingerprint (SHA-256) for the file so we can track it exactly.
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

    import hashlib
    filename = os.path.basename(file_path)
    document_name, extension = os.path.splitext(filename)
    document_type = extension.lstrip('.').lower()

    # Priority 2: Fallback to keyword matching if folder structure is flat/unknown
    if category == "unknown":
        name_lower = document_name.lower()
        if "manual" in name_lower or "service" in name_lower:
            category = "service_manual"
        elif "repair" in name_lower or "procedure" in name_lower:
            category = "repair_procedure"
        elif "diag" in name_lower or "flowchart" in name_lower:
            category = "diagnostic_flowchart"
        elif "wiring" in name_lower or "diagram" in name_lower:
            category = "wiring_description"
        elif "maintenance" in name_lower or "schedule" in name_lower:
            category = "maintenance_schedule"
        elif "tsb" in name_lower or "bulletin" in name_lower:
            category = "technical_service_bulletin"

    # Guarantee uniqueness using SHA256 of the absolute path
    abs_path = os.path.abspath(file_path)
    doc_id = hashlib.sha256(abs_path.encode('utf-8')).hexdigest()

    metadata = {
        "document_id": doc_id,
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

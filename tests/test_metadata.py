from src.metadata_generator import generate_metadata
import os

def test_generate_metadata():
    path = os.path.join("data", "service_manuals", "engine_manual.pdf")
    meta = generate_metadata(path, num_pages=5, num_chunks=10)

    assert meta["category"] == "service_manual"
    assert meta["document_name"] == "engine_manual"
    assert meta["document_type"] == "pdf"
    assert meta["num_pages"] == 5
    assert meta["num_chunks"] == 10

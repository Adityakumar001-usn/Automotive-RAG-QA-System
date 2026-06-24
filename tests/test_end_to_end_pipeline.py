import os
import shutil
import pytest
from src.processing import process_document
from src.chunking import recursive_chunking
from src.embeddings import generate_embeddings
from src.vector_store import VectorStore
from src.retriever import Retriever

def test_end_to_end_pipeline():
    test_file = "test_assets/sample.txt"
    index_dir = "tests/test_faiss_index_output"

    # 1. Extraction, Cleaning, Metadata
    text, metadata = process_document(test_file)
    assert "Test TXT Document" in text
    assert metadata["document_type"] == "txt"

    # 2. Chunking
    chunks = recursive_chunking(text, chunk_size=50)
    assert len(chunks) > 0
    metadata["num_chunks"] = len(chunks)

    # Generate identical metadatas for each chunk
    metadatas = [metadata.copy() for _ in chunks]

    # 3. Embedding (this is mocked by conftest.py)
    embeddings = generate_embeddings(chunks)
    assert embeddings.shape == (len(chunks), 384)

    # 4. FAISS Index
    vs = VectorStore()
    vs.build_index(embeddings, chunks, metadatas)
    vs.save_index(index_dir)

    assert os.path.exists(os.path.join(index_dir, "index.faiss"))

    # 5. Top-K Retrieval
    vs_loaded = VectorStore()
    vs_loaded.load_index(index_dir)

    retriever = Retriever(vs_loaded)
    results = retriever.retrieve("Testing query", top_k=1)

    assert len(results) == 1
    assert "chunk" in results[0]
    assert "metadata" in results[0]

    # Cleanup
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)

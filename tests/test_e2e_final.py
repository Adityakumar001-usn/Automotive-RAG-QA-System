import pytest
import os
import shutil
from src.processing import process_document
from src.chunking import recursive_chunking
from src.embeddings import generate_embeddings
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.rag_engine import AutomotiveRAG
from src.rag_evaluator import RAGEvaluator

def test_full_pipeline_e2e():
    """
    End-to-End system validation spanning from Document -> Output.
    Validates No Exceptions, Expected Outputs, and Traceability preservation.
    """
    # Phase 1: Ingestion & Metadata
    text, metadata = process_document("test_assets/sample.txt")
    assert "Test TXT Document" in text

    # Phase 2: Chunking
    chunks = recursive_chunking(text)
    metadatas = [metadata.copy() for _ in chunks]
    assert len(chunks) > 0

    # Phase 3: Embedding (Offline Mocked)
    embeddings = generate_embeddings(chunks)
    assert embeddings.shape[0] == len(chunks)

    # Phase 4: Vector Store FAISS
    vs = VectorStore()
    vs.build_index(embeddings, chunks, metadatas)
    assert vs.index.ntotal == len(chunks)

    # Phase 5: RAG Initialization
    rag = AutomotiveRAG(Retriever(vs))
    evaluator = RAGEvaluator()

    # Phase 6 & 7 & 8: Retrieval, LLM Engine, Evaluation
    result = rag.ask("Testing end to end.")

    # Traceability Validation
    assert result["question"] == "Testing end to end."
    assert "Mocked" in result["answer"]
    assert len(result["sources"]) > 0

    source = result["sources"][0]
    assert source["document_name"] == metadata["document_name"]
    assert source["category"] == metadata["category"]
    assert "chunk_id" in source
    assert "distance" in source

    assert len(result["retrieved_chunks"]) > 0
    assert len(result["retrieval_scores"]) > 0
    assert result["retrieval_time_ms"] >= 0.0

    # Evaluation Validation
    assert evaluator.answer_found(result) is True
    assert evaluator.source_count(result) > 0
    assert evaluator.retrieval_count(result) > 0
    assert evaluator.average_distance(result) >= 0.0
    assert evaluator.average_retrieval_score(result) >= 0.0

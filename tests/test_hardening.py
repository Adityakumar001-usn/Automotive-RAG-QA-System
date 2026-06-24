import pytest
import os
import shutil
import json
from src.metadata_generator import generate_metadata
from src.prompt_builder import PromptBuilder
import src.config
from src.rag_engine import AutomotiveRAG
from src.rag_evaluator import RAGEvaluator

def test_metadata_uniqueness():
    path1 = os.path.join("data", "service_manuals", "manual_a.pdf")
    path2 = os.path.join("data", "service_manuals", "manual_b.pdf")

    meta1 = generate_metadata(path1)
    meta2 = generate_metadata(path2)

    assert meta1["document_id"] != meta2["document_id"]
    assert len(meta1["document_id"]) == 64 # SHA256 length

def test_context_truncation():
    pb = PromptBuilder()

    src.config.MAX_CONTEXT_CHARS = 50
    chunks = [{"chunk": "A" * 100}]

    prompt = pb.build_prompt("Q?", chunks)
    # The context should be truncated to 50 chars.
    assert "A" * 50 in prompt
    assert "A" * 51 not in prompt

def test_retrieval_metrics_and_logging():
    src.config.ENABLE_RETRIEVAL_LOGGING = True

    class MockRetriever:
        def retrieve(self, query, top_k):
            return [
                {
                    "chunk": "First chunk.",
                    "metadata": {"document_name": "doc1", "category": "cat1", "source": "s1"},
                    "index_id": 1,
                    "distance": 0.2
                },
                {
                    "chunk": "Second chunk.",
                    "metadata": {"document_name": "doc2", "category": "cat2", "source": "s2"},
                    "index_id": 2,
                    "distance": 0.4
                }
            ]

    # Needs LLM engine mock via conftest.py
    rag = AutomotiveRAG(MockRetriever())
    result = rag.ask("Testing logging.")

    # Check evaluator metrics
    evaluator = RAGEvaluator()
    import math
    assert math.isclose(evaluator.average_distance(result), 0.3)
    assert math.isclose(evaluator.average_retrieval_score(result), 0.3)

    # Check logging file
    log_dir = "outputs/retrieval_logs"
    assert os.path.exists(log_dir)
    files = os.listdir(log_dir)
    assert len(files) > 0

    with open(os.path.join(log_dir, files[0]), "r") as f:
        log_data = json.load(f)
        assert "timestamp" in log_data
        assert log_data["question"] == "Testing logging."

    shutil.rmtree(log_dir)

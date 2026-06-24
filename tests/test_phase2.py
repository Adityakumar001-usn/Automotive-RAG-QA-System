import pytest
from src.prompt_builder import PromptBuilder
from src.llm_engine import LLMEngine
from src.rag_engine import AutomotiveRAG
from src.rag_evaluator import RAGEvaluator

def test_prompt_builder():
    pb = PromptBuilder()
    chunks = [
        {"chunk": "Context block 1."},
        {"chunk": "Context block 2."}
    ]
    prompt = pb.build_prompt("What is this?", chunks)

    assert "Context block 1." in prompt
    assert "Context block 2." in prompt
    assert "What is this?" in prompt
    assert "Information not found in provided documents." in prompt

def test_llm_engine_offline():
    # Because of conftest.py, this will not download anything.
    engine = LLMEngine()
    answer = engine.generate_response("Testing prompt")
    assert "Mocked deterministic answer based on context." in answer

def test_rag_engine_pipeline_and_evaluator():
    # Create a mock retriever
    class MockRetriever:
        def retrieve(self, query, top_k):
            return [
                {
                    "chunk": "Engine is broken.",
                    "metadata": {"document_name": "doc1", "category": "repair_procedure", "source": "s1"},
                    "index_id": 10,
                    "distance": 0.5
                },
                {
                    "chunk": "Brakes are fine.",
                    "metadata": {"document_name": "doc2", "category": "service_manual", "source": "s2"},
                    "index_id": 11,
                    "distance": 0.9
                }
            ]

    retriever = MockRetriever()
    rag = AutomotiveRAG(retriever)

    result = rag.ask("What is wrong with the car?")

    # 1. Pipeline Verification
    assert result["question"] == "What is wrong with the car?"
    assert "Mocked" in result["answer"]

    # 2. Transparency Verification
    assert "retrieved_chunks" in result
    assert len(result["retrieved_chunks"]) == 2
    assert "retrieval_scores" in result
    assert result["retrieval_scores"] == [0.5, 0.9]
    assert "retrieval_time_ms" in result
    assert result["retrieval_time_ms"] >= 0

    # 3. Traceability Verification
    sources = result["sources"]
    assert len(sources) == 2
    assert sources[0]["document_name"] == "doc1"
    assert sources[0]["category"] == "repair_procedure"
    assert sources[0]["source"] == "s1"
    assert sources[0]["chunk_id"] == 10
    assert sources[0]["distance"] == 0.5

    # 4. Evaluator Verification
    evaluator = RAGEvaluator()
    assert evaluator.answer_found(result) is True
    assert evaluator.source_count(result) == 2
    assert evaluator.retrieval_count(result) == 2

    # Check evaluator edge case
    result_not_found = {"answer": "Information not found in provided documents."}
    assert evaluator.answer_found(result_not_found) is False

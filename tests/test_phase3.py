import pytest
import os
import shutil
from src.context_window_experiment import ContextWindowExperiment
from src.benchmark_runner import BenchmarkRunner
from src.rag_engine import AutomotiveRAG

class MockRetriever:
    def retrieve(self, query, top_k):
        return [
            {
                "chunk": "First chunk " * 50, # approx 100 tokens
                "metadata": {"document_name": "doc1", "category": "cat1", "source": "s1"},
                "index_id": 1,
                "distance": 0.2
            },
            {
                "chunk": "Second chunk " * 200, # approx 400 tokens
                "metadata": {"document_name": "doc2", "category": "cat2", "source": "s2"},
                "index_id": 2,
                "distance": 0.4
            }
        ]

@pytest.fixture
def clean_results_dir():
    # Setup
    if os.path.exists("results"):
        shutil.rmtree("results")
    yield
    # Teardown
    if os.path.exists("results"):
        shutil.rmtree("results")

def test_context_window_truncation_via_tokenizer(clean_results_dir):
    # Relies on conftest.py mocking the Tokenizer and Model entirely.
    # The MockTokenizer in conftest.py has an __call__ that returns a constant length input_ids,
    # but for ContextWindowExperiment we used add_special_tokens=False, so let's check basic logic.
    rag = AutomotiveRAG(MockRetriever())

    # We will test the Benchmark Runner end-to-end which exercises the token truncator
    runner = BenchmarkRunner(rag)

    # Validate execution for ALL required windows per the prompt requirements
    runner.windows = [512, 1024, 2048, 4096]
    runner.questions = ["Q1?", "Q2?"]

    runner.run()
    runner.generate_outputs()

    # Verify files
    assert os.path.exists("results/latency_results.csv")
    assert os.path.exists("results/raw_benchmark_results.csv")
    assert os.path.exists("results/latency_plot.png")
    assert os.path.exists("results/memory_plot.png")
    assert os.path.exists("results/answer_quality_plot.png")
    assert os.path.exists("results/phase3_analysis.md")

    # Verify contents of markdown
    with open("results/phase3_analysis.md", "r") as f:
        md = f.read()
        assert "Overall Recommended Window:" in md
        assert "Best Latency Window:" in md

from tests.conftest import MockAutoModel, MockTokenizer, MockSentenceTransformer
import sys
import src.llm_engine
import src.embeddings
src.llm_engine.AutoTokenizer = type("MockAutoTokenizer", (), {"from_pretrained": lambda *a, **kw: MockTokenizer()})
src.llm_engine.AutoModelForCausalLM = type("MockCausalLM", (), {"from_pretrained": lambda *a, **kw: MockAutoModel()})
src.embeddings.SentenceTransformer = MockSentenceTransformer

from tests.test_phase3 import MockRetriever
from src.rag_engine import AutomotiveRAG
from src.benchmark_runner import BenchmarkRunner

runner = BenchmarkRunner(AutomotiveRAG(MockRetriever()))
runner.run()
runner.generate_outputs()

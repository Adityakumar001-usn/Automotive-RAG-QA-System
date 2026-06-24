import pytest
import numpy as np

class MockSentenceTransformer:
    def __init__(self, model_name, device):
        pass

    def encode(self, texts, *args, **kwargs):
        return np.random.rand(len(texts), 384).astype(np.float32)

@pytest.fixture(autouse=True)
def mock_sentence_transformer(monkeypatch):
    """Mocks SentenceTransformer to avoid downloading models during tests."""
    import src.embeddings
    monkeypatch.setattr(src.embeddings, "SentenceTransformer", MockSentenceTransformer)

class MockTokenizer:
    def __init__(self, *args, **kwargs):
        self.eos_token_id = 0
    def __call__(self, text, *args, **kwargs):
        class Inputs:
            def __init__(self):
                import torch
                self.input_ids = torch.tensor([[1, 2, 3]])
                self.attention_mask = torch.tensor([[1, 1, 1]])
            def to(self, device):
                return self
            def keys(self):
                return ["input_ids", "attention_mask"]
            def __getitem__(self, key):
                if key == "input_ids":
                    return self.input_ids
                if key == "attention_mask":
                    return self.attention_mask
        return Inputs()
    def decode(self, tokens, skip_special_tokens=True):
        return "Mocked deterministic answer based on context."

class MockAutoModel:
    def __init__(self, *args, **kwargs):
        pass
    def eval(self):
        pass
    def generate(self, **kwargs):
        import torch
        return torch.tensor([[1, 2, 3, 4, 5, 6, 7]])

@pytest.fixture(autouse=True)
def mock_llm_dependencies(monkeypatch):
    """Mocks AutoModelForCausalLM and AutoTokenizer to avoid downloads during tests."""
    import src.llm_engine
    monkeypatch.setattr(src.llm_engine, "AutoTokenizer", type("MockAutoTokenizer", (), {"from_pretrained": lambda *a, **kw: MockTokenizer()}))
    monkeypatch.setattr(src.llm_engine, "AutoModelForCausalLM", type("MockCausalLM", (), {"from_pretrained": lambda *a, **kw: MockAutoModel()}))

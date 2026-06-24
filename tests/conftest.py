import pytest
import numpy as np

class MockSentenceTransformer:
    def __init__(self, model_name, device):
        pass

    def encode(self, texts, convert_to_numpy=True, convert_to_tensor=False):
        return np.random.rand(len(texts), 384).astype(np.float32)

@pytest.fixture(autouse=True)
def mock_sentence_transformer(monkeypatch):
    """Mocks SentenceTransformer to avoid downloading models during tests."""
    import src.embeddings
    monkeypatch.setattr(src.embeddings, "SentenceTransformer", MockSentenceTransformer)

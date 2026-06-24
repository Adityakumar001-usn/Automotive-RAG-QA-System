import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from src.utils import get_logger
from src.config import EMBEDDING_MODEL

logger = get_logger(__name__)

# Global model cache to avoid reloading
_MODEL_INSTANCE = None

def get_model() -> SentenceTransformer:
    """Loads and caches the SentenceTransformer model on CPU."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        logger.info(f"Loading SentenceTransformer model: {EMBEDDING_MODEL}")
        # Explicitly enforce CPU per constraints
        _MODEL_INSTANCE = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
    return _MODEL_INSTANCE

def generate_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generates embeddings for a list of strings.
    Returns a NumPy array of embeddings.
    Embeddings are strictly stored and computed on CPU.
    """
    if not texts:
        # Return empty numpy array with the expected embedding dimension (384 for all-MiniLM-L6-v2)
        return np.empty((0, 384), dtype=np.float32)

    model = get_model()
    logger.info(f"Generating embeddings for {len(texts)} chunks.")

    # encode returns numpy arrays by default if convert_to_tensor is False
    embeddings = model.encode(texts, convert_to_numpy=True, convert_to_tensor=False)

    # Ensure standard numpy float32 array
    return np.array(embeddings, dtype=np.float32)

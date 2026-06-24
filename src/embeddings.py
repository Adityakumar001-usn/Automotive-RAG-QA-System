import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from src.utils import get_logger
from tqdm import tqdm
from src.config import EMBEDDING_MODEL

logger = get_logger(__name__)

# Global model cache to avoid reloading
_MODEL_INSTANCE = None

def get_model() -> SentenceTransformer:
    """
    Loads and caches the AI model that turns text into numbers (embeddings).

    Why it exists:
    Loading a machine learning model takes several seconds. If we loaded it every time we needed to process
    a text chunk, our code would be incredibly slow. This 'Singleton' pattern loads it once, saves it in
    memory, and reuses it.
    """
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None: # Only load if we haven't already!
        logger.info(f"[PHASE 3] Embedding Generation - Loading SentenceTransformer model: {EMBEDDING_MODEL}")
        # Explicitly enforce CPU per constraints so we don't accidentally blow up Colab's GPU memory
        _MODEL_INSTANCE = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
    return _MODEL_INSTANCE

def generate_embeddings(texts: List[str]) -> np.ndarray:
    """
    Translates normal text strings into dense number arrays (vectors/embeddings).

    Why it exists:
    Computers cannot understand raw words. They understand numbers. An embedding model turns a sentence
    into a list of 384 floating-point numbers. Sentences with similar meanings will have similar numbers!

    Inputs:
    texts (List[str]): A list of text chunks.

    Outputs:
    np.ndarray: A matrix (grid) of numbers representing the text.
    """
    if not texts:
        # Return an empty matrix with 384 columns (the size our specific model outputs)
        return np.empty((0, 384), dtype=np.float32)

    model = get_model()
    logger.info(f"Generating embeddings for {len(texts)} chunks.")

    # We use show_progress_bar=True natively built into SentenceTransformer
    # to show a nice tqdm progress bar while it processes lists of chunks!
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        convert_to_tensor=False,
        show_progress_bar=True
    )

    # Ensure it's a standard numpy float32 array for memory efficiency
    return np.array(embeddings, dtype=np.float32)

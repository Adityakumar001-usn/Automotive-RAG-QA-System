import numpy as np
from typing import List, Dict, Any, Tuple
from src.embeddings import generate_embeddings
from src.vector_store import VectorStore
from src.utils import get_logger

logger = get_logger(__name__)

class Retriever:
    def __init__(self, vector_store: VectorStore):
        """
        Initializes the retriever with a VectorStore.
        """
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Encodes the query, performs top-k similarity search on the FAISS index,
        and returns the ranked chunks and metadata.
        """
        if not query.strip():
            logger.warning("Empty query provided. Returning empty results.")
            return []

        # 1. Encode query
        query_embedding = generate_embeddings([query])

        # 2. Search FAISS index
        if self.vector_store.index.ntotal == 0:
            logger.warning("Vector store is empty. Returning empty results.")
            return []

        # Ensure float32 contiguous array
        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)

        distances, indices = self.vector_store.index.search(query_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1: # FAISS returns -1 if there aren't enough vectors
                continue

            chunk_text = self.vector_store.chunk_store.get(idx, "")
            metadata = self.vector_store.metadata_store.get(idx, {})

            result_item = {
                "chunk": chunk_text,
                "metadata": metadata,
                "distance": float(dist),
                "index_id": int(idx)
            }
            results.append(result_item)

        logger.info(f"Retrieved {len(results)} chunks for query: '{query}'")
        return results

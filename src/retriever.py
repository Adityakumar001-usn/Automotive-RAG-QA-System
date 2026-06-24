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
        Takes a human question and finds the most relevant document chunks.

        Why it exists:
        Before we ask the AI to answer a question, we must find the facts. This 'Retriever' translates
        the question into numbers (embeddings) and asks FAISS (the fast database) to find the closest
        matching numbers in our documents.

        Inputs:
        query (str): The user's question.
        top_k (int): How many chunks of text we want to bring back.

        Outputs:
        List[Dict[str, Any]]: A list of dictionaries containing the text chunk, its metadata, and how closely it matched.
        """
        import time
        start = time.time()
        logger.info(f"[PHASE 5] Retrieval - Start searching for top {top_k} results")

        if not query.strip():
            logger.warning("Empty query provided. Returning empty results.")
            return []

        # 1. Encode query into numbers so FAISS can understand it
        query_embedding = generate_embeddings([query])

        # 2. Search FAISS index
        if self.vector_store.index.ntotal == 0:
            logger.warning("Vector store is empty. Returning empty results.")
            return []

        # Ensure float32 contiguous array for memory safety
        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)

        # 'search' asks the database for the closest points.
        # It returns the 'distances' (how close they are) and 'indices' (the IDs of the chunks)
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

        logger.info(f"[PHASE 5] Retrieval - Completed in {time.time()-start:.2f}s. Retrieved {len(results)} chunks.")
        return results

import os
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from src.utils import get_logger

logger = get_logger(__name__)

class VectorStore:
    def __init__(self, dimension: int = 384):
        """
        Initializes the VectorStore with a FAISS IndexFlatL2.
        Default dimension corresponds to all-MiniLM-L6-v2.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        # Store metadata associated with each vector index
        self.metadata_store: Dict[int, Dict[str, Any]] = {}
        # Store chunk text
        self.chunk_store: Dict[int, str] = {}
        logger.info(f"Initialized FAISS IndexFlatL2 with dimension {self.dimension}")

    def build_index(self, embeddings: np.ndarray, chunks: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """
        Adds embeddings, chunks, and metadata to the FAISS index.
        """
        if embeddings.shape[0] == 0:
            logger.warning("No embeddings provided to build index.")
            return

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected embedding dimension {self.dimension}, got {embeddings.shape[1]}")

        if len(chunks) != embeddings.shape[0] or len(metadatas) != embeddings.shape[0]:
            raise ValueError("Number of chunks and metadata items must match number of embeddings.")

        # Ensure type is float32 for FAISS
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        from tqdm import tqdm
        logger.info(f"[PHASE 4] FAISS Indexing - Adding vectors to database")

        start_id = self.index.ntotal
        self.index.add(embeddings) # This puts the math arrays into the fast search engine

        # We must also save the actual readable text and metadata, because FAISS only stores numbers.
        # We loop through, matching the index ID to the actual text chunk.
        for i in tqdm(range(embeddings.shape[0]), desc="Mapping vectors to text"):
            idx = start_id + i
            self.chunk_store[idx] = chunks[i]
            self.metadata_store[idx] = metadatas[i]

        logger.info(f"Completed FAISS Indexing! Added {embeddings.shape[0]} vectors. Total vectors: {self.index.ntotal}")

    def save_index(self, output_dir: str = "faiss_index") -> None:
        """
        Saves the FAISS index, chunks, and metadata to disk.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        index_path = os.path.join(output_dir, "index.faiss")
        faiss.write_index(self.index, index_path)

        # Save metadata and chunks
        import pickle
        with open(os.path.join(output_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata_store, f)

        with open(os.path.join(output_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunk_store, f)

        logger.info(f"Saved FAISS index and metadata to {output_dir}")

    def load_index(self, index_dir: str = "faiss_index") -> None:
        """
        Loads the FAISS index, chunks, and metadata from disk.
        """
        index_path = os.path.join(index_dir, "index.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")

        self.index = faiss.read_index(index_path)

        import pickle
        with open(os.path.join(index_dir, "metadata.pkl"), "rb") as f:
            self.metadata_store = pickle.load(f)

        with open(os.path.join(index_dir, "chunks.pkl"), "rb") as f:
            self.chunk_store = pickle.load(f)

        logger.info(f"Loaded FAISS index from {index_dir}. Total vectors: {self.index.ntotal}")

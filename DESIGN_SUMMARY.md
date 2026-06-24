# Phase 1 Design Summary

## Objective
Establish a robust, memory-efficient Document Processing Pipeline tailored for large Automotive datasets (Service Manuals, TSBs, Wiring diagrams, etc.), laying the foundation for a future Retrieval-Augmented Generation (RAG) system.

## Constraints Adhered To
1. **CPU/NumPy Storage Only**: The `src/embeddings.py` enforces device mapping to CPU, using NumPy float32 contiguous arrays to strictly satisfy memory footprint rules.
2. **Native Python Chunking**: Rather than relying on LangChain or LlamaIndex, `src/chunking.py` implements pure regex-based recursive chunking. This maximizes portability across environments like Colab and reduces the dependency tree footprint.
3. **No LLM Integration**: The scope intentionally omits LLM and prompt template implementations. The pipeline stops exactly at FAISS IndexFlatL2 Top-K retrieval, per requirements.
4. **Offline Testing**: `SentenceTransformer` calls are mocked via pytest fixtures (`tests/conftest.py`) enabling fast, offline testing pipelines.

## Component Breakdown
*   **Ingestion Pipeline (`ingestion.py`, `processing.py`)**: Extensible handlers for PDFs (PyMuPDF for speed/reliability), tabular data (Pandas), Docs, and OCR (Tesseract).
*   **Metadata Generator (`metadata_generator.py`)**: Rule-based inference eliminates the need for manual tagging. The category is directly inferred from the `data/` subdirectory structure.
*   **Vector Database (`vector_store.py`)**: A lightweight wrapper around `faiss.IndexFlatL2` tracking vectors and storing chunks + metadata alongside it via standard python persistence.
*   **Colab Optimization**: All modules operate iteratively to be safe for typical T4 runtimes (e.g. streaming vectors to FAISS rather than buffering them unnecessarily).

# Automotive RAG Question Answering System

This repository contains the foundation and execution engine for an Automotive Retrieval-Augmented Generation (RAG) Question Answering System. The system spans multiple phases from raw dataset preparation to robust LLM inference and tracing.

## Architecture

The system is currently composed of two primary phases, integrating ingestion, chunking, and deterministic LLM inference seamlessly:

### Phase 1: Dataset Preparation & Document Pipeline
1. **Ingestion (`src/ingestion.py`)**: Supports reading PDF (PyMuPDF), CSV/XLSX (Pandas/openpyxl), DOCX (python-docx), TXT, and Images (pytesseract/Pillow). Handles OCR dependency fallbacks explicitly.
2. **Cleaning & Metadata (`src/utils.py`, `src/metadata_generator.py`)**:
   - Uses lightweight generic heuristics (regex and string ops) to remove headers/footers, normalize text, and remove blank lines.
   - Generates unique SHA256 IDs for documents to prevent metadata collision.
3. **Chunking (`src/chunking.py`)**: Implements Native Python algorithms for Fixed, Overlapping, and Recursive (Paragraph -> Sentence -> Word) chunking.
4. **Embeddings (`src/embeddings.py`)**: Uses `sentence-transformers/all-MiniLM-L6-v2` loaded strictly on CPU and outputs memory-efficient NumPy arrays.
5. **Vector Store (`src/vector_store.py`)**: Manages a CPU-only FAISS `IndexFlatL2` vector store.

### Phase 2: RAG QA System
1. **Retriever (`src/retriever.py`)**: Top-K retrieval generating chunk texts and precise FAISS similarity distances.
2. **Prompt Builder (`src/prompt_builder.py`)**: Rigid prompt templates preventing hallucination, featuring strict `MAX_CONTEXT_CHARS` protections against token overflow.
3. **LLM Engine (`src/llm_engine.py`)**: Singleton, lazy-loaded inference interface. Uses 4-bit `BitsAndBytesConfig` quantization dynamically assigned via `device_map="auto"` on GPUs with full CPU fallback mapping.
4. **Automotive RAG & Transparency (`src/rag_engine.py`)**: The pipeline controller. Embeds detailed retrieval logs (chunks, latencies, and explicit distances) into the output dictionary and optional `outputs/retrieval_logs/*.json` hashes.
5. **RAG Evaluator (`src/rag_evaluator.py`)**: Computes analytical checks (answer validity, distances, source bounds) natively.

#### Phase 2 Architecture Flow

```
Question
↓
Retriever
↓
Retrieved Chunks
↓
Prompt Builder
↓
LLM Engine
↓
Grounded Answer
↓
Source Traceability
↓
RAG Evaluator
```

## Folder Structure

```text
Automotive-RAG-QA-System/
├── data/                       # Categorized source documents (manuals, TSBs, etc)
├── metadata/
├── faiss_index/                # Persistent FAISS store
├── outputs/                    # Output logs & traces (e.g. outputs/retrieval_logs/)
├── src/                        # Core application modules
│   ├── config.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── ingestion.py
│   ├── llm_engine.py
│   ├── metadata_generator.py
│   ├── processing.py
│   ├── prompt_builder.py
│   ├── rag_engine.py
│   ├── rag_evaluator.py
│   ├── retriever.py
│   ├── utils.py
│   └── vector_store.py
├── tests/                      # Pytest suite
├── test_assets/
├── notebook/
│   ├── phase1_colab.ipynb
│   └── phase2_rag_demo.ipynb
├── requirements.txt
├── DESIGN_SUMMARY.md
└── README.md
```

## Setup & Installation

**Prerequisites:** Python 3.11+ and Tesseract OCR.

*Note: The environment specifically pins `transformers==4.41.2` and `accelerate==0.32.0` because newer versions have known instantiation conflicts with `Phi-3-mini-4k-instruct` tensor layers, and older versions of accelerate have `clear_device_cache` import errors.*

```bash
# Ubuntu dependency for OCR
sudo apt-get install tesseract-ocr

# Install Python requirements
pip install -r requirements.txt
```

## Running Tests

The test suite ensures offline reliability. `SentenceTransformer` and HuggingFace Tokenizers/Models are strictly mocked to prevent network bandwidth leakage.

```bash
python3 -m pytest tests/
```

## Quick Start (Recommended)

The easiest way to execute, demonstrate, and evaluate the Automotive RAG Question Answering System is via the unified Master Notebook. This notebook is highly recommended for Colab execution, supervisor demonstrations, and dissertation evaluations.

1. Go to [colab.research.google.com](https://colab.research.google.com/).
2. Select **File > Open notebook > GitHub**.
3. Paste the URL to this repository.
4. Open the **`notebook/automotive_rag_master.ipynb`** file.
5. Set your Runtime to **T4 GPU** (`Runtime > Change runtime type > T4 GPU`).
6. Run the cells sequentially! The notebook will automatically clone the backend, allow you to upload your own files, index them, ask RAG questions, and generate the final Phase 3 analytics dashboards natively.

*Recommendation: For the best evaluation experience, upload multiple documents from different categories together (e.g., Service Manuals, TSBs, Wiring Diagrams, Maintenance Schedules) during the upload prompt in Section 2. The pipeline handles multi-category processing transparently, mapping chunks appropriately.*

*(Note: The individual phase notebooks remain available in the `notebook/` folder for detailed module-by-module study).*

### Phase 3: Context Window Study
Phase 3 expands the QA system by introducing an evaluation framework designed to benchmark different context window sizes.
1. **Benchmark Runner (`src/benchmark_runner.py`)**: Automates executions over multiple configurations, saving granular and aggregate datasets via CSVs.
2. **Metrics Collector (`src/metrics_collector.py`)**: Employs `psutil` and PyTorch profiling to actively monitor CPU and GPU memory loads without altering underlying mechanisms.
3. **Context Window Experiments (`src/context_window_experiment.py`)**: Intercepts retrieved document chunks and trims them accurately at specific Token boundaries (using the exact active Tokenizer) preventing context overflow inherently while providing real analytical constraints.

Running the evaluation experiment produces full analytical graphics (latency, memory, quality) directly into the `results/` folder, completing Task 6 research metrics.

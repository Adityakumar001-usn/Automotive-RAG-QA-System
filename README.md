# Automotive RAG Question Answering System - Phase 1

This repository contains the Phase 1 Foundation for an Automotive Retrieval-Augmented Generation (RAG) Question Answering System. The focus of this phase is on the **Dataset Preparation** and **Document Processing Pipeline**.

## System Architecture

Phase 1 encompasses the ingestion of unstructured automotive documents, cleaning, metadata inference, chunking, and vectorization:

1. **Ingestion (`src/ingestion.py`)**: Supports reading PDF (PyMuPDF), CSV/XLSX (Pandas/openpyxl), DOCX (python-docx), TXT, and Images (pytesseract/Pillow).
2. **Cleaning & Metadata (`src/utils.py`, `src/metadata_generator.py`)**:
   - Uses lightweight generic heuristics (regex and string ops) to remove headers/footers, normalize text, and remove blank lines.
   - Infers structured metadata based on the directory layout (e.g., categorizing `service_manuals` automatically).
3. **Chunking (`src/chunking.py`)**: Implements Native Python algorithms for Fixed, Overlapping, and Recursive (Paragraph -> Sentence -> Word) chunking.
4. **Embeddings (`src/embeddings.py`)**: Uses `sentence-transformers/all-MiniLM-L6-v2` loaded strictly on CPU and outputs memory-efficient NumPy arrays.
5. **Vector Store & Retriever (`src/vector_store.py`, `src/retriever.py`)**:
   - Manages a FAISS `IndexFlatL2` vector store.
   - Provides Top-K similarity search functionality for downstream RAG queries.

## Folder Structure

```
Automotive-RAG-QA-System/
├── data/                       # Contains all document datasets categorized by subfolder
│   ├── service_manuals/
│   ├── repair_procedures/
│   ├── diagnostic_flowcharts/
│   ├── wiring_descriptions/
│   ├── maintenance_schedules/
│   └── tsb/
├── metadata/                   # Output directory for structured metadata files (if needed)
├── faiss_index/                # Persistent storage for FAISS indices and chunk pickles
├── outputs/                    # General directory for arbitrary pipeline outputs
├── src/                        # Source code modules
│   ├── ingestion.py
│   ├── processing.py
│   ├── chunking.py
│   ├── metadata_generator.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── utils.py
├── tests/                      # Unit tests suite (pytest)
├── test_assets/                # Dummy files to facilitate unit tests and notebook demos
├── notebook/
│   └── phase1_colab.ipynb      # Google Colab compatible demonstration
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation
```

## Setup & Installation

**Prerequisites:** Python 3.11+ and Tesseract OCR installed on the system (for image extraction).

```bash
# Ubuntu system dependency for OCR
sudo apt-get install tesseract-ocr

# Install Python requirements
pip install -r requirements.txt
```

## Running the Tests

To ensure the pipeline works and does not attempt to download models from HuggingFace (they are mocked during testing), execute:

```bash
python3 -m pytest tests/
```

## Google Colab Usage

The notebook is configured to run smoothly on Google Colab T4 environments.
1. Upload the repository contents to Colab.
2. Ensure you have the `data/` layout and any test assets present.
3. Open `notebook/phase1_colab.ipynb`.
4. Install requirements and execute the end-to-end pipeline cells.

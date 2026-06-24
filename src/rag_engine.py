from typing import Dict, Any, List
import time
from src.retriever import Retriever
from src.prompt_builder import PromptBuilder
from src.llm_engine import LLMEngine
from src.config import TOP_K
from src.utils import get_logger

logger = get_logger(__name__)

class AutomotiveRAG:
    def __init__(self, retriever: Retriever):
        """
        Initializes the full Automotive RAG pipeline.
        """
        self.retriever = retriever
        self.prompt_builder = PromptBuilder()
        self.llm_engine = LLMEngine()
        logger.info("Initialized AutomotiveRAG engine.")

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Executes the QA pipeline:
        1. Retrieves Top-K chunks (measures latency)
        2. Builds prompt
        3. Generates answer using LLM
        4. Extracts and formats sources with complete traceability
        """
        logger.info(f"Processing question: '{question}'")

        # 1. Retrieve with Latency Measurement
        start_time = time.time()
        retrieved_chunks = self.retriever.retrieve(question, top_k=TOP_K)
        retrieval_time_ms = (time.time() - start_time) * 1000

        retrieved_texts = []
        retrieval_scores = []

        # 4. Compile Sources
        sources = []
        # We don't deduplicate here because we need exact chunks per the traceability requirement
        for item in retrieved_chunks:
            meta = item.get("metadata", {})
            doc_name = meta.get("document_name", "Unknown")
            cat = meta.get("category", "Unknown")
            source = meta.get("source", "Unknown")

            retrieved_texts.append(item.get("chunk", ""))
            retrieval_scores.append(item.get("distance", 0.0))

            sources.append({
                "document_name": doc_name,
                "category": cat,
                "source": source,
                "chunk_id": item.get("index_id", -1),
                "distance": item.get("distance", 0.0)
            })

        # 2. Build Prompt
        prompt = self.prompt_builder.build_prompt(question, retrieved_chunks)

        # 3. Generate Answer
        answer = self.llm_engine.generate_response(prompt)

        result = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_texts,
            "retrieval_scores": retrieval_scores,
            "retrieval_time_ms": retrieval_time_ms
        }

        from src.config import ENABLE_RETRIEVAL_LOGGING
        if ENABLE_RETRIEVAL_LOGGING:
            import os
            import json
            import uuid

            log_dir = "outputs/retrieval_logs"
            os.makedirs(log_dir, exist_ok=True)

            log_entry = {
                **result,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }

            log_file = os.path.join(log_dir, f"log_{uuid.uuid4().hex}.json")
            with open(log_file, "w") as f:
                json.dump(log_entry, f, indent=2)
            logger.info(f"Retrieval log written to {log_file}")

        logger.info("Successfully completed QA pipeline.")
        return result

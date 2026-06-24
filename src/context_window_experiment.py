import torch
from typing import List, Dict, Any
from src.prompt_builder import PromptBuilder
from src.utils import get_logger

logger = get_logger(__name__)

class ContextWindowExperiment:
    """
    Engine for running the RAG pipeline dynamically limited by strict Token Window bounds
    rather than character bounds.
    """
    def __init__(self, rag_engine):
        self.rag = rag_engine
        # Ensure tokenizer is loaded by initializing LLMEngine safely
        self.rag.llm_engine._load_model()

    def _truncate_chunks_by_tokens(self, chunks: List[Dict[str, Any]], max_tokens: int) -> tuple[List[Dict[str, Any]], int]:
        """
        Takes retrieved chunks and truncates their text cumulatively to strictly
        adhere to the max_tokens limit using the underlying HuggingFace tokenizer.
        Returns the truncated chunks and the total number of tokens used.
        """
        import src.llm_engine as llm_engine
        if llm_engine._TOKENIZER is None:
            logger.warning("Tokenizer not loaded, skipping strict token truncation.")
            return chunks, 0

        truncated_chunks = []
        current_tokens = 0

        for item in chunks:
            text = item.get("chunk", "")

            # Count tokens of current chunk
            encoded = llm_engine._TOKENIZER(text, add_special_tokens=False)["input_ids"]
            if hasattr(encoded, 'tolist'): # handle torch tensors if tokenizer returns them by default
                encoded = encoded.tolist()
                if isinstance(encoded[0], list):
                    encoded = encoded[0]
            chunk_token_count = len(encoded)

            if current_tokens + chunk_token_count <= max_tokens:
                truncated_chunks.append(item)
                current_tokens += chunk_token_count
            else:
                # We have reached the boundary, we need to partially truncate this chunk
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 0:
                    truncated_ids = encoded[:remaining_tokens]
                    truncated_text = llm_engine._TOKENIZER.decode(truncated_ids, skip_special_tokens=True)
                    new_item = item.copy()
                    new_item["chunk"] = truncated_text
                    truncated_chunks.append(new_item)
                    current_tokens += remaining_tokens
                break # Limit reached

        return truncated_chunks, current_tokens

    def ask_with_window(self, question: str, max_tokens: int) -> Dict[str, Any]:
        """
        Intercepts the normal RAG flow to enforce token boundaries on the retrieved context.
        """
        from src.config import TOP_K
        import time

        # 1. Retrieve normally
        start_time = time.time()
        retrieved_chunks = self.rag.retriever.retrieve(question, top_k=TOP_K)
        retrieval_time_ms = (time.time() - start_time) * 1000

        # 2. Apply explicit Token Window truncation to chunks
        truncated_chunks, tokens_used = self._truncate_chunks_by_tokens(retrieved_chunks, max_tokens)

        # Calculate Utilization Metric
        context_utilization_percent = (tokens_used / max_tokens) * 100 if max_tokens > 0 else 0.0

        retrieved_texts = []
        retrieval_scores = []
        sources = []

        for item in truncated_chunks:
            meta = item.get("metadata", {})
            sources.append({
                "document_name": meta.get("document_name", "Unknown"),
                "category": meta.get("category", "Unknown"),
                "source": meta.get("source", "Unknown"),
                "chunk_id": item.get("index_id", -1),
                "distance": item.get("distance", 0.0)
            })
            retrieved_texts.append(item.get("chunk", ""))
            retrieval_scores.append(item.get("distance", 0.0))

        # 3. Build Prompt
        # Note: We bypass prompt_builder.build_prompt's character truncation internally
        # by doing this token-level prep before it, or we simply rely on this text being small enough.
        prompt = self.rag.prompt_builder.build_prompt(question, truncated_chunks)

        # 4. Generate Answer
        answer = self.rag.llm_engine.generate_response(prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_texts,
            "retrieval_scores": retrieval_scores,
            "retrieval_time_ms": retrieval_time_ms,
            "answer_length": len(answer),
            "context_utilization_percent": context_utilization_percent
        }

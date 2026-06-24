from typing import Dict, Any

class RAGEvaluator:
    """
    Lightweight evaluator for the RAG pipeline output.
    Used to detect answer generation status, source citation counts, and retrieval counts.
    """

    def answer_found(self, response: Dict[str, Any]) -> bool:
        """
        Detects whether a valid grounded answer was generated.
        Returns False if the LLM output exactly matches the fallback string.
        """
        answer = response.get("answer", "").strip()
        fallback_str = "Information not found in provided documents."
        return answer != fallback_str and bool(answer)

    def source_count(self, response: Dict[str, Any]) -> int:
        """
        Returns the number of cited sources.
        """
        return len(response.get("sources", []))

    def retrieval_count(self, response: Dict[str, Any]) -> int:
        """
        Returns the number of retrieved chunks.
        """
        return len(response.get("retrieved_chunks", []))

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

    def average_distance(self, response: Dict[str, Any]) -> float:
        """
        Calculates the average distance from the retrieved sources.
        """
        sources = response.get("sources", [])
        if not sources:
            return 0.0

        distances = [s.get("distance", 0.0) for s in sources]
        return sum(distances) / len(distances)

    def average_retrieval_score(self, response: Dict[str, Any]) -> float:
        """
        Calculates the average retrieval score from the retrieved chunks.
        """
        scores = response.get("retrieval_scores", [])
        if not scores:
            return 0.0

        return sum(scores) / len(scores)

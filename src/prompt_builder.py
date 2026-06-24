from typing import List, Dict, Any
from src.utils import get_logger

logger = get_logger(__name__)

class PromptBuilder:
    def __init__(self):
        """Initializes the PromptBuilder."""
        self.system_prompt = (
            "You are an automotive technical assistant.\n\n"
            "Answer ONLY using the provided context.\n\n"
            "If the answer cannot be found in the provided context, respond exactly:\n\n"
            "\"Information not found in provided documents.\"\n"
        )

    def build_prompt(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Builds the final prompt combining system instructions, retrieved context, and user question.
        Ensures strict grounding to prevent hallucinations.
        """
        from src.config import MAX_CONTEXT_CHARS

        # Combine all chunk texts
        context_text = "\n\n".join([item["chunk"] for item in retrieved_chunks if "chunk" in item])

        # Truncate context to prevent Model Context Window Overflow
        if len(context_text) > MAX_CONTEXT_CHARS:
            logger.warning(f"Context exceeds {MAX_CONTEXT_CHARS} characters. Truncating to prevent overflow.")
            context_text = context_text[:MAX_CONTEXT_CHARS]

        prompt = (
            f"{self.system_prompt}\n"
            f"Context:\n{context_text}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer:"
        )

        logger.info("Successfully built prompt from question and chunks.")
        return prompt

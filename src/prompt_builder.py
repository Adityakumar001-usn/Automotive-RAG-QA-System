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

        Why it exists:
        AI models need clear instructions on how to act. We take the facts we found (chunks) and the user's
        question, and package them into a strict template that tells the AI to NOT invent answers.

        Inputs:
        question (str): The user's query.
        retrieved_chunks (List): The blocks of text containing the facts.

        Outputs:
        str: The final, formatted string that is sent to the LLM.
        """
        import time
        start = time.time()
        logger.info("[PHASE 6] Prompt Construction - Start")

        from src.config import MAX_CONTEXT_CHARS

        # Combine all chunk texts with blank lines between them
        context_text = "\n\n".join([item["chunk"] for item in retrieved_chunks if "chunk" in item])

        # Truncate context to prevent Model Context Window Overflow. If we feed too much text to an AI, it crashes.
        if len(context_text) > MAX_CONTEXT_CHARS:
            logger.warning(f"Context exceeds {MAX_CONTEXT_CHARS} characters. Truncating to prevent overflow.")
            context_text = context_text[:MAX_CONTEXT_CHARS]

        prompt = (
            f"{self.system_prompt}\n"
            f"Context:\n{context_text}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer:"
        )

        logger.info(f"[PHASE 6] Prompt Construction - Completed in {time.time()-start:.4f}s.")
        return prompt

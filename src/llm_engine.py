import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from src.config import LLM_MODEL, MAX_NEW_TOKENS, TEMPERATURE, USE_4BIT
from src.utils import get_logger

logger = get_logger(__name__)

# Global singletons for lazy loading
_MODEL = None
_TOKENIZER = None

class LLMEngine:
    def __init__(self):
        """
        Initializes the LLMEngine.
        Model and tokenizer are lazy-loaded upon first use to save memory.
        """
        pass

    def _load_model(self):
        """Lazy loads the model and tokenizer into global variables."""
        global _MODEL, _TOKENIZER
        if _MODEL is not None and _TOKENIZER is not None:
            return

        logger.info(f"Loading tokenizer for {LLM_MODEL}")
        _TOKENIZER = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)

        # Determine device/quantization based on constraints
        if torch.cuda.is_available() and USE_4BIT:
            logger.info("CUDA detected. Loading model in 4-bit mode using BitsAndBytes.")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            _MODEL = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            logger.info("CUDA not available or 4-bit not requested. Loading model on CPU.")
            # For CPU, we simply map it to cpu
            _MODEL = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                device_map="cpu",
                trust_remote_code=True
            )

        _MODEL.eval()
        logger.info("Model loaded successfully.")

    def generate_response(self, prompt: str) -> str:
        """
        Generates a string answer given the constructed prompt.

        Why it exists:
        This is where the magic happens! We feed the final instructional prompt into the massive AI model.
        The AI predicts the best words to follow our prompt, generating the human-readable answer.

        Inputs:
        prompt (str): The structured instructions containing the context and question.

        Outputs:
        str: The AI's generated answer text.
        """
        import time
        start = time.time()
        logger.info("[PHASE 7] LLM Generation - Start")

        self._load_model()

        global _MODEL, _TOKENIZER

        # Let Transformers manage device placement automatically
        # to ensure compatibility with device_map="auto" and BitsAndBytes.
        inputs = _TOKENIZER(prompt, return_tensors="pt")

        # Generation settings to ensure deterministic behavior per constraints
        with torch.no_grad():
            outputs = _MODEL.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=False if TEMPERATURE <= 0 else True,
                pad_token_id=_TOKENIZER.eos_token_id
            )

        # Decode only the newly generated tokens
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        answer = _TOKENIZER.decode(generated_tokens, skip_special_tokens=True).strip()

        logger.info(f"[PHASE 7] LLM Generation - Completed in {time.time()-start:.2f}s.")
        return answer

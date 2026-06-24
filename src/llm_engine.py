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
        """
        self._load_model()

        global _MODEL, _TOKENIZER
        device = "cuda" if torch.cuda.is_available() else "cpu"

        inputs = _TOKENIZER(prompt, return_tensors="pt").to(device)

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

        logger.info("Generated answer successfully.")
        return answer

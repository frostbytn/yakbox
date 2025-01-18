import logging
from typing import Dict
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer)

class ModelManager:
    """Manages the model and tokenizer lifecycle and provides simple tokenize/decode methods."""
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # For 4-bit quantized models, we do not call `.to(self.device)` explicitly,
        # as it's not supported. The model should already be on the correct device.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # todo add configuration support here for these vaules.
            torch_dtype=torch.float32,
            trust_remote_code=True,
            load_in_4bit=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        logging.info(f"Model loaded: {model_name}")

    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize a given text input into model-compatible tensors."""
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def decode(self, tokens: torch.Tensor) -> str:
        """Decode a given tensor of token IDs into a string."""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

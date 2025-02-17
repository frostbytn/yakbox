import logging
from typing import Dict
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

# Quantization configuration from config.ini
load_in_4bit = config.getboolean("quantization", "load_in_4bit", fallback=True)
bnb_4bit_compute_dtype_str = config.get("quantization", "bnb_4bit_compute_dtype", fallback="bfloat16")
dtype_mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
bnb_4bit_compute_dtype = dtype_mapping.get(bnb_4bit_compute_dtype_str.lower(), torch.bfloat16)
bnb_4bit_use_double_quant = config.getboolean("quantization", "bnb_4bit_use_double_quant", fallback=False)
bnb_4bit_quant_type = config.get("quantization", "bnb_4bit_quant_type", fallback="nf4")

class ModelManager:
    """Manages the model and tokenizer lifecycle and provides simple tokenize/decode methods."""
    def __init__(self, model_name: str, device: str):
        self.device = device
                
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
            )
        
        if self.tokenizer.pad_token is None or self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            # Add a custom pad token if missing
            # This ensures pad token != EOS token
            self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            self.tokenizer.pad_token = "<PAD>"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            # load_in_4bit=load_in_4bit,
            quantization_config=quantization_config,
            device_map="auto" 
        )

        # self.tokenizer.pad_token = self.tokenizer.eos_token
        logging.info(f"Model loaded: {model_name}")

    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize a given text input into model-compatible tensors."""
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def decode(self, tokens: torch.Tensor) -> str:
        """Decode a given tensor of token IDs into a string."""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

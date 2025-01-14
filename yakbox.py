import asyncio
import configparser
import json
import logging
import uuid
from threading import Thread
from time import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse, JSONResponse
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList,
                          TextIteratorStreamer)

logging.basicConfig(level=logging.INFO)

config = configparser.ConfigParser()
config.read("config.ini")

MODEL_NAME: str = config["model"]["name"]
USE_CUDA: bool = config.getboolean("device", "use_cuda")
DEVICE: str = "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu"
STOP_TOKENS: List[str] = config["tokens"]["stop_tokens"].split(",")

app = FastAPI()
app.add_middleware(
    # Allow CORS for all origins, methods, and headers for local development
    # For production, consider restricting the origins to a specific domain.
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

yakky = r"""
  ___    ___  ________   ___  __     ________   ________      ___    ___ 
 |\  \  /  /||\   __  \ |\  \|\  \  |\   __  \ |\   __  \    |\  \  /  /|
 \ \  \/  / /\ \  \|\  \\ \  \/  /|_\ \  \|\ /_\ \  \|\  \   \ \  \/  / /
  \ \    / /  \ \   __  \\ \   ___  \\ \   __  \\ \  \\\  \   \ \    / / 
   \/  /  /    \ \  \ \  \\ \  \\ \  \\ \  \|\  \\ \  \\\  \   /     \/  
 __/  / /       \ \__\ \__\\ \__\\ \__\\ \_______\\ \_______\ /  /\   \  
|\___/ /         \|__|\|__| \|__| \|__| \|_______| \|_______|/__/ /\ __\ 
\|___|/                                                      |__|/ \|__| 
"""

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

class PromptGenerator:
    """Generates a prompt from user messages with a predefined instruction block."""
    # Instructions to kick off a conversational prompt for the model - remove this
    # if system prompts are coming in from the user.
    TOOL_INSTRUCTIONS = """
        <GENERAL>
        - You are a helpful and knowledgeable assistant.
        - Use </s> to indicate the end of your thought.
        </GENERAL>
        """
    @staticmethod
    def generate(messages: List["Message"]) -> str:
        """Generate a prompt by concatenating a system prompt and the conversation messages."""
        conversation = "\n".join(f"{msg.role}: {msg.content}" for msg in messages)
        return f"{PromptGenerator.TOOL_INSTRUCTIONS}\n\n{conversation}"

class StopOnTokens(StoppingCriteria):
    """Stopping criteria to end generation on encountering certain stop tokens."""
    def __init__(self, stop_token_ids: List[int]):
        super().__init__()
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        return input_ids[0, -1].item() in self.stop_token_ids

class Message(BaseModel):
    role: str = Field(..., description="Role of the speaker, e.g., 'user' or 'assistant'.")
    content: str = Field(..., description="Message content.")

class ChatCompletionRequest(BaseModel):
    messages: List[Message] = Field(..., description="List of messages in the conversation.")
    model: str = Field(..., description="Model name to use.")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional generation parameters.")
    stream: bool = Field(default=False, description="If True, stream the response.")

class ResponseStreamer:
    """Streams the response from the model as tokens are generated."""
    def __init__(self, model_manager: ModelManager, stop_token_ids: List[int]):
        self.model_manager = model_manager
        self.stop_token_ids = stop_token_ids

    async def stream(
        self, messages: List[Message], params: Dict[str, Any]
        ) -> AsyncGenerator[str, None]:
        """Asynchronously streams the model output token by token."""

        prompt = f'<s><INST>{prompt_generator.generate(messages)}</INST>'
        
        logging.info(f"Prompt: {prompt}")
        inputs = self.model_manager.tokenize(prompt)
        streamer = TextIteratorStreamer(
            self.model_manager.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        stopping_criteria = StoppingCriteriaList([StopOnTokens(self.stop_token_ids)])
        generation_kwargs = {
            "inputs": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "streamer": streamer,
            "stopping_criteria": stopping_criteria,
            "max_new_tokens": params.get("max_new_tokens", 512),
            "temperature": params.get("temperature", 1.0),
            "top_p": params.get("top_p", 0.95),
            "do_sample": True,
        }

        thread = Thread(target=self.model_manager.model.generate, kwargs=generation_kwargs)
        thread.start()

        async for token in self.async_token_generator(streamer):
            logging.info(f"Token: {token}")
            yield self.format_response_chunk(token)

        thread.join()
        yield self.format_response_chunk("", finish_reason="stop")

    @staticmethod
    async def async_token_generator(streamer: TextIteratorStreamer) -> AsyncGenerator[str, None]:
        """Yields generated tokens asynchronously from the streamer."""
        loop = asyncio.get_event_loop()
        while True:
            token = await loop.run_in_executor(None, lambda: next(streamer, None))
            if token is None:
                break
            yield token

    @staticmethod
    def format_response_chunk(content: str, finish_reason: Optional[str] = None) -> str:
        """Format a token into a streaming completion chunk."""
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time()),
            "model": MODEL_NAME,
            "choices": [{"delta": {"content": content}, "index": 0, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(chunk)}\n\n"

# todo a lot of these parameters should be ripped from the incoming request instead of
# hardcoded or from the config file - quick implementation to get going.
def extract_parameters(params: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """Extract generation parameters, using defaults from the config if not provided."""
    return {
        "max_new_tokens": params.get("max_tokens", config.getint("parameters", "max_tokens")),
        "temperature": params.get("temperature", config.getfloat("parameters", "temperature")),
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": config.getfloat("parameters", "repetition_penalty"),
        "top_k": config.getint("parameters", "top_k"),
        "top_p": config.getfloat("parameters", "top_p"),
    }

def build_response(content: str) -> Dict[str, Any]:
    """Build a non-streaming chat completion response."""
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time()),
        "model": MODEL_NAME,
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "index": 0,
            "finish_reason": "stop"
        }],
    }

def generate_model_output(prompt: str, params: Dict[str, Any], model_manager: ModelManager) -> str:
    """Generate model output without streaming."""
    inputs = model_manager.tokenize(prompt)
    with torch.no_grad():
        output = model_manager.model.generate(inputs["input_ids"], **params)
    return model_manager.decode(output[0])

model_manager = ModelManager(MODEL_NAME, DEVICE)
prompt_generator = PromptGenerator()
stop_token_ids = model_manager.tokenizer.convert_tokens_to_ids(STOP_TOKENS)
response_streamer = ResponseStreamer(model_manager, stop_token_ids)

logging.info("\n%s", yakky)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log incoming requests."""
    body = await request.body()
    logging.info(f"Request: {request.method} {request.url} Body: {body.decode('utf-8')}")
    response = await call_next(request)
    return response


@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Endpoint to handle chat completions (similar to OpenAI's /chat/completions).
    Supports streaming responses.
    """
    params = extract_parameters(request.parameters, model_manager.tokenizer)

    if request.stream:
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
        return StreamingResponse(
            response_streamer.stream(request.messages, params),
            headers=headers
        )

    # Non-streaming response
    prompt = f'<s><INST>{prompt_generator.generate(request.messages)}</INST>'
    generated_text = generate_model_output(prompt, params, model_manager)
    return JSONResponse(build_response(generated_text))

@app.get("/v1/models")
async def get_models():
    """Endpoint to list models (following OpenAI style endpoints)."""
    return {
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "owned_by": "your-organization",
            "permission": [{"allow_create_engine": True}]
        }]
    }

@app.get("/v1/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}

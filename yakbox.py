import configparser
import json
import logging
import uuid
from time import time
from typing import Any, Dict, List, Optional
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, JSONResponse
from transformers import (AutoTokenizer)
from services.model_manager import ModelManager
from services.prompt_generator import PromptGenerator
from services.response_streamer import ResponseStreamer
from models.chat_completion_request import ChatCompletionRequest

import re

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

# todo a lot of these parameters should be ripped from the incoming request instead of
# hardcoded or from the config file - quick implementation to get going.
def extract_parameters(params: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """Extract generation parameters, using defaults from the config if not provided."""
    return {
        "max_new_tokens": params.get("max_tokens", config.getint("parameters", "max_tokens")),
        "temperature": params.get("temperature", config.getfloat("parameters", "temperature")),
        "pad_token_id": tokenizer.pad_token_id,
        "repetition_penalty": config.getfloat("parameters", "repetition_penalty"),
        "top_k": config.getint("parameters", "top_k"),
        "top_p": config.getfloat("parameters", "top_p"),
    }

def build_response(content: str, tool_call: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a non-streaming chat completion response."""
    response = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time()),
        "model": MODEL_NAME,
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "index": 0,
            "finish_reason": "stop"
        }]
    }

    if tool_call:
        response["choices"][0]["tool_call"] = tool_call

    return response

def generate_model_output(prompt: str, params: Dict[str, Any], model_manager: ModelManager) -> str:
    """Generate model output without streaming."""
    inputs = model_manager.tokenize(prompt)
    with torch.no_grad():
        output = model_manager.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **params
        )
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
            response_streamer.stream(request.messages, params, request.tools),
            headers=headers
        )

    # Non-streaming response
    prompt = f'<s><INST>{prompt_generator.generate(request.messages, request.tools)}</INST>'
    generated_text = generate_model_output(prompt, params, model_manager)
    
    # Extract potential tool call from the generated content
    # Improved tool call extraction
    tool_call = None
    try:
        # Use a refined regex to match JSON blocks, handling possible variations in content formatting
        match = re.search(r"```json\s*({.*?})\s*```", generated_text, re.DOTALL)
        if match:
            # Extract the JSON string
            tool_call_json = match.group(1).strip()
            tool_call = json.loads(tool_call_json)
            logging.info("Tool call detected and parsed successfully.")
    except (ValueError, json.JSONDecodeError) as e:
        logging.warning(f"Failed to parse tool call from content: {e}")
    
    response = build_response(generated_text, tool_call)
    return JSONResponse(response)

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

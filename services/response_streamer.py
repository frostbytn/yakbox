import asyncio
import configparser
import json
import logging
import uuid
from threading import Thread
from time import time
from typing import Any, AsyncGenerator, Dict, Optional
from transformers import (StoppingCriteriaList,TextIteratorStreamer)
from services.prompt_generator import PromptGenerator
from models.stop_on_tokens import StopOnTokens

import re

config = configparser.ConfigParser()
config.read("config.ini")

MODEL_NAME: str = config["model"]["name"]

class ResponseStreamer:
    """
    Streams tokens in a form phidata's `OpenAIChat` can recognize:
    - For normal text: { "choices": [{"delta": {"role": "assistant", "content": "..."} }] }
    - For a function call: { "choices": [{"delta": {"role": "assistant", "tool_calls": [...]}}] }
    """

    def __init__(self, model_manager, stop_token_ids):
        self.model_manager = model_manager
        self.stop_token_ids = stop_token_ids

    async def stream(
        self,
        messages,
        params: Dict[str, Any],
        tools
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronously streams partial tokens as SSE lines.
        Once a function call is detected, emits a chunk with "tool_calls".
        """
        prompt = f'<s><INST>{PromptGenerator.generate(messages, tools)}</INST>'
        logging.info(f"Prompt: {prompt}")

        inputs = self.model_manager.tokenize(prompt)
        streamer = TextIteratorStreamer(
            self.model_manager.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
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

        token_buffer = ""
        function_call_detected = None

        async for token in self.async_token_generator(streamer):
            logging.info(f"Token: {token}")
            token_buffer += token

            # Check for a function call block: ```json { "function":"...", "arguments":{...}} ```
            matches = re.findall(r"```json\s*({.*?})\s*```", token_buffer, re.DOTALL)
            if matches:
                try:
                    parsed_json = json.loads(matches[-1].strip())
                    if "function" in parsed_json and "arguments" in parsed_json:
                        function_call_detected = parsed_json
                    else:
                        logging.warning("JSON block didn't have the expected keys.")
                except (ValueError, json.JSONDecodeError) as e:
                    logging.warning(f"Failed to parse function call JSON: {e}")

            if not function_call_detected:
                yield self.format_chunk(
                    role="assistant",
                    content=token
                )
            else:
                # Once a function call is detected, we stop sending partial text
                # so as not to stream out partial JSON tokens any further.
                pass

        thread.join()
        if function_call_detected:
            logging.info(f"Function call detected: {function_call_detected}")
            yield self.format_tool_calls_chunk(function_call_detected)
        yield self.format_chunk(
            finish_reason="stop"
        )

    @staticmethod
    async def async_token_generator(streamer: TextIteratorStreamer) -> AsyncGenerator[str, None]:
        """Yields tokens from the HF streamer in an async manner."""
        loop = asyncio.get_event_loop()
        while True:
            token = await loop.run_in_executor(None, lambda: next(streamer, None))
            if token is None:
                break
            yield token

    def format_tool_calls_chunk(self, function_call: Dict[str, Any]) -> str:
        """
        Create a chunk with "tool_calls" so phidata sees it as a function call.
        Example structure:
          "tool_calls": [
            {
              "id": "...",
              "type": "function_call",
              "function": { "name": "...", "arguments": "..." }
            }
          ]
        """
        name = function_call.get("function", "unknown_function")
        args_str = json.dumps(function_call.get("arguments", {}))
        tool_call_id = str(uuid.uuid4())

        chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time()),
            "model": MODEL_NAME,
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tool_call_id,
                                # todo index probably has to be incremented for multiple tool calls???
                                "index": 1,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": args_str
                                }
                            }
                        ]
                    },
                    "index": 0,
                    "finish_reason": None
                }
            ]
        }

        json_str = json.dumps(chunk)
        logging.info(f"data: {json_str}")
        return f"data: {json_str}\n\n"

    def format_chunk(
        self,
        role: Optional[str] = None,
        content: Optional[str] = None,
        finish_reason: Optional[str] = None
    ) -> str:
        """
        Creates a chunk for partial tokens or final "stop". If `role` and
        `content` are set, it streams normal text. If `finish_reason`=stop, 
        it signals the stream end.
        """
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time()),
            "model": MODEL_NAME,
            "choices": [
                {
                    "delta": {},
                    "index": 0,
                    "finish_reason": finish_reason
                }
            ]
        }

        if role:
            chunk["choices"][0]["delta"]["role"] = role
        if content:
            chunk["choices"][0]["delta"]["content"] = content

        json_str = json.dumps(chunk)
        logging.info(f"data: {json_str}")
        return f"data: {json_str}\n\n"
import json
from models.message import Message
from models.function import Function
from typing import List
import logging

class PromptGenerator:
    """Generates a prompt from user messages with a predefined instruction block."""
    # Instructions to kick off a conversational prompt for the model - remove this
    # if system prompts are coming in from the user.
    
    TOOL_INSTRUCTIONS = """
    <GENERAL>
    - Use the "Available Tools" structure to generate a tool call only when it is the most appropriate way to answer the user's question or complete the task.
    - When generating a tool call, always return it in this exact JSON format:

    ```json{
        "function": "function_name",
        "arguments": {
            "param1": "value1",
            "param2": "value2"
        }
    }```

    - Base your decision on the context of the user's request and the purpose of the available tools.
    - Use </s> to indicate the end of your thought or response.
    </GENERAL>

    <DECISION CRITERIA>
    - Generate a tool call if:
    - The user's request explicitly references or requires using a tool.
    - The task cannot be completed directly without using a tool (e.g., calculations, file operations, external lookups).
    </DECISION CRITERIA>

    <EXAMPLES>
    User: What is 2 plus 3?
    Assistant: Tool call:

    ```json{
        "function": "add",
        "arguments": {
            "a": 2,
            "b": 3
        }
    }```
    </s>

    User: Write "Hello, World!" to a file.
    Assistant: tool call:

    ```json{
        "function": "write_to_file",
        "arguments": {
            "filename": "example.txt",
            "content": "Hello, World!"
        }
    }```
    </s>
    </EXAMPLES>
    """
    
    @staticmethod
    def generate(messages: List[Message], tools: List[Function]) -> List[dict]:
        # if tools are not provided then return the conversation messages
        if tools is None or len(tools) == 0:
            conversation_messages = [{"role": msg.role, "content": f"{msg.content}"} for msg in messages]
            logging.info("No tools provided. Returning conversation messages.")
            return conversation_messages
        
        
        system_message = {
            "role": "system",
            "content": PromptGenerator.TOOL_INSTRUCTIONS
        }
        
        # if tools are provided then return the conversation messages along with the tools
        tool_descriptions = "\n".join([
            f"Function: {tool.function['name']}\nDescription: {tool.function['description']}\nParameters: {json.dumps(tool.function['parameters'])}"
            for tool in tools
        ])
        tool_message = {
            "role": "system",
            "content": f"Available Tools:\n{tool_descriptions}"
        }

        conversation_messages = [{"role": msg.role, "content": f"{msg.content}"} for msg in messages]
        logging.info("Returning conversation messages with tool instructions and tools.")
        return [system_message, tool_message] + conversation_messages

import json
from models.message import Message
from models.function import Function
from typing import List

class PromptGenerator:
    """Generates a prompt from user messages with a predefined instruction block."""
    # Instructions to kick off a conversational prompt for the model - remove this
    # if system prompts are coming in from the user.
    TOOL_INSTRUCTIONS = """
    <GENERAL>
    - You are a helpful and knowledgeable assistant.
    - Use the "Available Tools" structure to generate a tool call only when it is the most appropriate way to answer the user's question or complete the task.
    - When generating a tool call, always return it in this exact JSON format:

    ```json{
        "function": "function_name",
        "arguments": {
            "param1": "value1",
            "param2": "value2"
        }
    }```

    - If a question or task can be answered or resolved directly without a tool, provide a natural response instead of generating a function call.
    - Base your decision on the context of the user's request and the purpose of the available tools.
    - Use </s> to indicate the end of your thought or response.
    - IMPORTANT: Provide concise, straightforward, short, to the point answers.
    </GENERAL>

    <DECISION CRITERIA>
    - Generate a tool call if:
    - The user's request explicitly references or requires using a tool.
    - The task cannot be completed directly without using a tool (e.g., calculations, file operations, external lookups).
    - Provide a direct response if:
    - The answer is straightforward and does not require a tool (e.g., general knowledge, conversational responses).
    - The user has not explicitly requested the use of a tool.
    </DECISION CRITERIA>

    <EXAMPLES>
    User: What is 2 plus 3?
    Assistant: To answer this question, I will use the 'add' function. Here is the tool call:

    ```json{
        "function": "add",
        "arguments": {
            "a": 2,
            "b": 3
        }
    }```
    </s>

    User: How many days are in a week?
    Assistant: There are 7 days in a week.
    </s>

    User: Write "Hello, World!" to a file.
    Assistant: To accomplish this, I will use the 'write_to_file' function. Here is the tool call:

    ```json{
        "function": "write_to_file",
        "arguments": {
            "filename": "example.txt",
            "content": "Hello, World!"
        }
    }```
    </s>

    User: What is the capital of France?
    Assistant: The capital of France is Paris.
    </s>
    </EXAMPLES>
    """

    @staticmethod
    def generate(messages: List[Message], tools: List[Function]) -> List[dict]:
        system_message = {
            "role": "system",
            "content": PromptGenerator.TOOL_INSTRUCTIONS
        }

        tool_descriptions = "\n".join([
            f"Function: {tool.function['name']}\nDescription: {tool.function['description']}\nParameters: {json.dumps(tool.function['parameters'])}"
            for tool in tools
        ])
        tool_message = {
            "role": "system",
            "content": f"Available Tools:\n{tool_descriptions}"
        }

        conversation_messages = [{"role": msg.role, "content": f"{msg.content}</s>"} for msg in messages]
        return [system_message, tool_message] + conversation_messages

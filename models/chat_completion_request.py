from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from models.message import Message
from models.function import Function

class ChatCompletionRequest(BaseModel):
    messages: List[Message] = Field(..., description="List of messages in the conversation.")
    model: str = Field(..., description="Model name to use.")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional generation parameters.")
    stream: bool = Field(default=False, description="If True, stream the response.")
    stream_options: Optional[Dict[str, Any]] = Field(default=None, description="Options for streaming responses.")
    tools: List[Function] = Field(default_factory=list, description="List of available functions/tools with metadata.")
    
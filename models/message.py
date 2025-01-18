from pydantic import BaseModel, Field

class Message(BaseModel):
    role: str = Field(..., description="Role of the speaker, e.g., 'user' or 'assistant'.")
    content: str = Field(..., description="Message content.")

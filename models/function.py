from typing import Any, Dict
from pydantic import BaseModel, Field

class Function(BaseModel):
    type: str = Field(..., description="Type of the function, e.g., 'function'.")
    function: Dict[str, Any] = Field(..., description="Details of the function including name, description, and parameters.")

import json
import re
import logging
from typing import Optional, Dict, Any

def parse_function_call(text: str) -> Optional[Dict[str, Any]]:
    """
    Detects and parses a function call from the provided text.
    Looks for a JSON block delimited by ```json and returns it as a dictionary if valid.
    """
    matches = re.findall(r"```json\s*({.*?})\s*```", text, re.DOTALL)
    if matches:
        try:
            parsed = json.loads(matches[-1].strip())
            if "function" in parsed and "arguments" in parsed:
                logging.info(f"Function call detected: {parsed}")
                return parsed
            else:
                logging.warning("Parsed JSON does not contain required keys 'function' and 'arguments'.")
        except (ValueError, json.JSONDecodeError) as e:
            logging.warning(f"Failed to parse function call JSON: {e}")
    return None

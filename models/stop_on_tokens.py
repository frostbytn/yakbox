import torch
from transformers import (StoppingCriteria)

class StopOnTokens(StoppingCriteria):
    """Stops generation once a token in stop_token_ids is encountered."""
    def __init__(self, stop_token_ids):
        super().__init__()
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        return input_ids[0, -1].item() in self.stop_token_ids
from typing import Any, List, Union

import numpy as np
import torch
from torch.autograd import Variable


def to_python_float(input: Union[torch.Tensor, List, float]) -> float:
    """Convert a torch tensor or list to a Python float.

    Args:
        input (Union[torch.Tensor, List, float]): Input tensor, list, or float.

    Returns:
        float: Converted Python float.
    """
    if isinstance(input, torch.Tensor) and hasattr(input, "item"):
        return input.cpu().detach().numpy() if input.numel() > 1 else input.item()
    if isinstance(input, list):
        return input[0]
    return input


def as_numpy(obj: Any) -> Any:
    """Convert a nested structure of torch tensors or lists to numpy arrays.

    Args:
        obj (Any): Object to be converted.

    Returns:
        Any: Converted object with torch tensors replaced by numpy arrays.
    """
    if isinstance(obj, list):
        return [as_numpy(v) for v in obj]
    if isinstance(obj, dict):
        return {k: as_numpy(v) for k, v in obj.items()}
    if isinstance(obj, Variable):
        return obj.data.cpu().numpy()
    if torch.is_tensor(obj):
        return obj.cpu().numpy()
    return np.array(obj)

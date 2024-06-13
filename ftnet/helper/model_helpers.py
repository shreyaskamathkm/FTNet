import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def check_mismatch(
    model_dict: Dict[str, torch.Tensor], pretrained_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Checks for mismatches between the model state dict and the pretrained
    state dict.

    Args:
        model_dict (Dict[str, torch.Tensor]): The model's state dict.
        pretrained_dict (Dict[str, torch.Tensor]): The pretrained state dict.

    Returns:
        Dict[str, torch.Tensor]: The updated model state dict.
    """
    pretrained_dict = {
        key[6:]: item for key, item in pretrained_dict.items()
    }  # Remove the prefix from pretrained_dict keys
    temp_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if model_dict[k].shape != pretrained_dict[k].shape:
                logger.info(
                    f"Skip loading parameter: {k}, "
                    f"required shape: {model_dict[k].shape}, "
                    f"loaded shape: {pretrained_dict[k].shape}"
                )
                continue
            temp_dict[k] = v
    return temp_dict


def save_model_summary(model: nn.Module, dir_: Path) -> None:
    """Print and save the network summary to a file.

    Args:
        model (nn.Module): The model to summarize.
        dir_ (Path): The directory to save the model summary.
    """
    path = dir_ / "model.txt"
    num_params = sum(param.numel() for param in model.parameters())
    config = repr(model)
    config += f"\nTotal number of parameters: {num_params}"
    config += f"\nTotal number of parameters in M: {num_params / (1000**2)}M"
    with path.open("w") as text_file:
        text_file.write(config)


def total_gradient(parameters: Any) -> float:
    """Computes the total gradient norm of the parameters.

    Args:
        parameters (Any): The parameters to compute the gradient norm.

    Returns:
        float: The total gradient norm.
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    totalnorm = 0.0
    for p in parameters:
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm**2
    return totalnorm**0.5


def print_network(net: nn.Module) -> None:
    """Print the network architecture and the total number of parameters.

    Args:
        net (nn.Module): The network to print.
    """
    num_params = sum(param.numel() for param in net.parameters())
    print(net)
    print(f"Total number of parameters: {num_params / (1000**2)} M")

import logging
import os
from collections import OrderedDict

import torch

logger = logging.getLogger(__name__)


def check_mismatch(model_dict: dict, pretrained_dict: dict) -> dict:
    """Checks for mismatches between the model state dict and the pretrained
    state dict.

    Args:
        model_dict (dict): The model's state dict.
        pretrained_dict (dict): The pretrained state dict.

    Returns:
        dict: The updated model state dict.
    """

    pretrained_dict = {key[6:]: item for key, item in pretrained_dict.items()}
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


def save_model_summary(model, dir_):
    """Print and save the network."""
    path = os.path.join(dir_, "model.txt")
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    config = repr(model)
    config += f"\nTotal number of parameters: {num_params}"
    config += f"\nTotal number of parameters in M: {num_params / (1000**2)}M"
    with open(path, "w") as text_file:
        text_file.write(config)


def total_gradient(parameters):
    # =============================================================================
    #     Computes a gradient clipping coefficient based on gradient norm
    # =============================================================================
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    totalnorm = 0
    for p in parameters:
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm**2
    return totalnorm ** (1.0 / 2)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print(f"Total number of parameters: {int(num_params) / 1000**2} M")


def save_checkpoint(states, is_best, output_dir, filename="checkpoint.pth.tar"):
    torch.save(states, os.path.join(output_dir, filename))

    if is_best and "state_dict" in states:
        torch.save(states, os.path.join(output_dir, "model_best.pth.tar"))

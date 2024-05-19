import collections
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.autograd import Variable

__all__ = [
    "get_rank",
    "save_model_summary",
    "plot_tensors",
    "as_numpy",
    "plot_tensors",
    "save_checkpoint",
    "print_network",
    "total_gradient",
]


def th_delete(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


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


def plot_tensors(img, x=None):
    im = img.detach().cpu().numpy()
    dim = img.ndim
    # (ch, _, _) = im.shape
    plt.figure()
    if dim == 3:
        im = np.moveaxis(im, 0, 2)
        plt.imshow(im)
    else:
        plt.imshow(im, cmap=plt.get_cmap("gray"))
    if x is not None:
        plt.title(x)
    plt.show()


def to_python_float(t):
    if hasattr(t, "item"):
        if t.numel() > 1:
            return t.cpu().detach().numpy()
        else:
            return t.item()
    elif isinstance(t, list):
        return t[0]
    else:
        return t


def as_numpy(obj):
    if isinstance(obj, collections.Sequence):
        return [as_numpy(v) for v in obj]
    elif isinstance(obj, collections.Mapping):
        return {k: as_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, Variable):
        return obj.data.cpu().numpy()
    elif torch.is_tensor(obj):
        return obj.cpu().numpy()
    else:
        return np.array(obj)


def save_checkpoint(states, is_best, output_dir, filename="checkpoint.pth.tar"):
    torch.save(states, os.path.join(output_dir, filename))

    if is_best and "state_dict" in states:
        torch.save(states, os.path.join(output_dir, "model_best.pth.tar"))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print(f"Total number of parameters: {int(num_params) / 1000**2} M")


def total_gradient(parameters):
    # =============================================================================
    #     Computes a gradient clipping coefficient based on gradient norm
    # =============================================================================
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    totalnorm = 0
    for p in parameters:
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm**2
    totalnorm = totalnorm ** (1.0 / 2)
    return totalnorm

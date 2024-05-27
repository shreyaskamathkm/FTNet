import collections

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.autograd import Variable

__all__ = [
    "get_rank",
    "plot_tensors",
    "as_numpy",
    "plot_tensors",
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
        return t.cpu().detach().numpy() if t.numel() > 1 else t.item()
    return t[0] if isinstance(t, list) else t


def as_numpy(obj):
    if isinstance(obj, collections.abc.Sequence):
        return [as_numpy(v) for v in obj]
    if isinstance(obj, collections.abc.Mapping):
        return {k: as_numpy(v) for k, v in obj.items()}
    if isinstance(obj, Variable):
        return obj.data.cpu().numpy()
    return obj.cpu().numpy() if torch.is_tensor(obj) else np.array(obj)

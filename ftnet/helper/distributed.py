import torch.distributed as dist


def get_rank() -> int:
    """Get the rank of the current process in distributed training.

    Returns:
        int: Rank of the current process. Returns 0 if distributed environment is not available or not initialized.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

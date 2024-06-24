from typing import Callable, Iterator

import torch
import torch.utils.data as data

from .batch_sampler import IterationBasedBatchSampler
from .multiscale_batch_samplers import (
    IterationBasedMultiscaleBatchSampler,
    MultiscaleBatchSampler,
)


def make_data_sampler(dataset: Callable, shuffle: bool, distributed: bool) -> Iterator:
    """Create a data sampler for a dataset.

    Args:
        dataset (Callable): The dataset to sample from.
        shuffle (bool): Whether to shuffle the data.
        distributed (bool): Whether the sampling is for distributed training.

    Returns:
        Iterator: The data sampler.
    """
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = data.sampler.RandomSampler(dataset)
    else:
        sampler = data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(
    sampler: Iterator, batch_size: int, num_iters: int = None, start_iter: int = 0
) -> Iterator:
    """Create a batch data sampler.

    Args:
        sampler (Iterator): The data sampler.
        batch_size (int): The size of each batch.
        num_iters (int, optional): The number of iterations. Defaults to None.
        start_iter (int, optional): The starting iteration. Defaults to 0.

    Returns:
        Iterator: The batch data sampler.
    """
    batch_sampler = data.sampler.BatchSampler(sampler, batch_size, drop_last=True)
    if num_iters:
        batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iters, start_iter)
    return batch_sampler


def make_multiscale_batch_data_sampler(
    sampler: Iterator,
    batch_size: int = 2,
    multiscale_step: int = 1,
    scales: int = 1,
    num_iters: int = None,
    start_iter: int = 0,
) -> Iterator:
    """Creates a multiscale batch data sampler based on the provided
    parameters.

    Args:
        sampler: An iterator to sample data from.
        batch_size: An integer representing the batch size (default is 2).
        multiscale_step: An integer specifying the multiscale step i.e across how many steps, the sizes need to be same (default is 1).
        scales: An integer indicating the number of scales i.e how many times of sizes are available (default is 1).
        num_iters: An integer representing the number of iterations (default is None).
        start_iter: An integer specifying the starting iteration (default is 0).

    Returns:
        Iterator: A multiscale batch data sampler.

    Examples:
        batch_sampler = make_multiscale_batch_data_sampler(sampler, batch_size=4, multiscale_step=2, scales=3)
    """

    batch_sampler = MultiscaleBatchSampler(
        sampler=sampler,
        batch_size=batch_size,
        drop_last=True,
        multiscale_step=multiscale_step,
        scales=scales,
    )
    if num_iters:
        batch_sampler = IterationBasedMultiscaleBatchSampler(batch_sampler, num_iters, start_iter)
    return batch_sampler

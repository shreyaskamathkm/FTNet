#!/usr/bin/env python3
"""
Code Adapted from:
https://github.com/CaoWGG/multi-scale-training

"""

from typing import Callable

import numpy as np
from torch.utils.data.sampler import Sampler

__all__ = ["MultiscaleBatchSampler", "IterationBasedMultiscaleBatchSampler"]


class MultiscaleBatchSampler:
    r"""Samples elements with different scales per batch i.e. when scale == 2
    then the dataloader __getitem__ will be a tuple with index and scale.

    Usage:
        def __getitem__(self, idx):
            if type(idx) == list or type(idx) == tuple:
                idx, scale = idx

    Arguments:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
            with ``__len__`` implemented.
        batch_size (int): Size of mini-batch
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
        multiscale_step(int): Number of times a scale needs to be repeated in a sequential batch
        scales(int): Number of scale variations required in the dataloader, for example, is two different crop sizes are required in during training,
        then scales =2.
    """

    def __init__(
        self,
        sampler: Callable,
        batch_size: int,
        drop_last: bool,
        multiscale_step: int = 1,
        scales: int = 1,
    ) -> None:
        if not isinstance(sampler, Sampler):
            raise ValueError(
                f"sampler should be an instance of torch.utils.data.Sampler, but got sampler={sampler}"
            )
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        if multiscale_step is not None and multiscale_step < 1:
            raise ValueError(
                f"multiscale_step should be > 0, but got multiscale_step={multiscale_step}"
            )

        if not isinstance(scales, int):
            raise ValueError(f"scales must a int, got {scales}")

        self.scales = scales
        self.multiscale_step = multiscale_step

    def __iter__(self):
        num_batch = 0
        batch = []
        scale = np.random.randint(self.scales)
        print(scale)
        for idx in self.sampler:
            batch.append([idx, scale])
            if len(batch) == self.batch_size:
                yield batch
                num_batch += 1
                batch = []
                if self.multiscale_step and num_batch % self.multiscale_step == 0:
                    scale = np.random.randint(self.scales)

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size

        return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class IterationBasedMultiscaleBatchSampler(MultiscaleBatchSampler):
    def __init__(self, batch_sampler: Callable, num_iterations: int, start_iter: int = 0) -> None:
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter
        super().__init__(
            sampler=batch_sampler.sampler,
            batch_size=batch_sampler.batch_size,
            drop_last=False,
            multiscale_step=batch_sampler.multiscale_step,
            scales=batch_sampler.scales,
        )

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations

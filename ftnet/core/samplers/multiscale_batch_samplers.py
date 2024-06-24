#!/usr/bin/env python3
"""
Code Adapted from:
https://github.com/CaoWGG/multi-scale-training

"""

from typing import Iterator, List, Union

import numpy as np
from torch.utils.data.sampler import Sampler


class MultiscaleBatchSampler:
    r"""Samples elements with different scales per batch i.e. when scale == 2
    then the dataloader __getitem__ will be a tuple with index and scale.

    Usage:
        def __getitem__(self, idx):
            if type(idx) == list or type(idx) == tuple:
                idx, scale = idx

    Arguments:
        sampler (Sampler): Base sampler. Must be an instance of `torch.utils.data.Sampler`.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If `True`, the sampler will drop the last batch if its size would be less than `batch_size`.
        multiscale_step (int): Number of times a scale needs to be repeated in a sequential batch.
        scales (int): Number of scale variations required in the dataloader.
    """

    def __init__(
        self,
        sampler: Sampler,
        batch_size: int,
        drop_last: bool,
        multiscale_step: int = 1,
        scales: int = 1,
    ) -> None:
        """Initialize the MultiscaleBatchSampler.

        Args:
            sampler (Sampler): The sampler to use.
            batch_size (int): Number of samples per batch.
            drop_last (bool): If True, drop the last incomplete batch.
            multiscale_step (int, optional): Step size for multiscale sampling. Defaults to 1.
            scales (int, optional): Number of scale variations. Defaults to 1.

        Raises:
            ValueError: If invalid arguments are provided.
        """
        if not isinstance(sampler, Sampler):
            raise ValueError(
                f"sampler should be an instance of torch.utils.data.Sampler, but got sampler={sampler}"
            )
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")
        if multiscale_step < 1:
            raise ValueError(
                f"multiscale_step should be > 0, but got multiscale_step={multiscale_step}"
            )
        if not isinstance(scales, int):
            raise ValueError(f"scales must be an int, but got scales={scales}")

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.multiscale_step = multiscale_step
        self.scales = scales

    def __iter__(self) -> Iterator[List[Union[int, int]]]:
        """Yield batches of samples with their corresponding scale.

        Yields:
            Iterator[List[Union[int, int]]]: Batches of samples with scale.
        """
        num_batch = 0
        batch = []
        scale = np.random.randint(self.scales)
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

    def __len__(self) -> int:
        """Return the number of batches.

        Returns:
            int: Number of batches.
        """
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class IterationBasedMultiscaleBatchSampler(MultiscaleBatchSampler):
    """Wraps a MultiscaleBatchSampler, resampling from it until a specified
    number of iterations have been sampled."""

    def __init__(
        self, batch_sampler: MultiscaleBatchSampler, num_iterations: int, start_iter: int = 0
    ) -> None:
        """Initialize the IterationBasedMultiscaleBatchSampler.

        Args:
            batch_sampler (MultiscaleBatchSampler): The batch sampler to wrap.
            num_iterations (int): Number of iterations to sample.
            start_iter (int, optional): Starting iteration. Defaults to 0.
        """
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter
        super().__init__(
            sampler=batch_sampler.sampler,
            batch_size=batch_sampler.batch_size,
            drop_last=batch_sampler.drop_last,
            multiscale_step=batch_sampler.multiscale_step,
            scales=batch_sampler.scales,
        )

    def __iter__(self) -> Iterator[List[Union[int, int]]]:
        """Yield batches of samples until the specified number of iterations is
        reached.

        Yields:
            Iterator[List[Union[int, int]]]: Batches of samples with scale.
        """
        iteration = self.start_iter
        while iteration < self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration >= self.num_iterations:
                    break
                yield batch

    def __len__(self) -> int:
        """Return the number of iterations.

        Returns:
            int: Number of iterations.
        """
        return self.num_iterations

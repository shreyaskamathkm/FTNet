#!/usr/bin/env python3
"""
Code Adapted from:
https://github.com/CaoWGG/multi-scale-training

"""

from typing import Iterator, List, Optional, Union

import numpy as np
from torch.utils.data import Sampler


class BatchSampler:
    def __init__(
        self,
        sampler: Sampler,
        batch_size: int,
        drop_last: bool,
        multiscale_step: Optional[int] = None,
        img_sizes: Optional[List[int]] = None,
    ) -> None:
        """Initialize the BatchSampler.

        Args:
            sampler (Sampler): The sampler to use.
            batch_size (int): Number of samples per batch.
            drop_last (bool): If True, drop the last incomplete batch.
            multiscale_step (Optional[int], optional): Step size for multiscale sampling. Defaults to None.
            img_sizes (Optional[List[int]], optional): List of image sizes for multiscale sampling. Defaults to None.

        Raises:
            ValueError: If invalid arguments are provided.
        """
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                f"torch.utils.data.Sampler, but got sampler={sampler}"
            )
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got drop_last={drop_last}")
        if multiscale_step is not None and multiscale_step < 1:
            raise ValueError(
                "multiscale_step should be > 0, but got multiscale_step={multiscale_step}"
            )
        if multiscale_step is not None and img_sizes is None:
            raise ValueError(f"img_sizes must be a list, but got img_sizes={img_sizes}")

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.multiscale_step = multiscale_step
        self.img_sizes = img_sizes

    def __iter__(self) -> Iterator[List[Union[int, int]]]:
        """Yield batches of samples.

        Yields:
            Iterator[List[Union[int, int]]]: Batches of samples with size.
        """
        num_batch = 0
        batch = []
        size = np.random.choice(self.img_sizes) if self.img_sizes else None
        for idx in self.sampler:
            batch.append([idx, size])
            if len(batch) == self.batch_size:
                yield batch
                num_batch += 1
                batch = []
                if self.multiscale_step and num_batch % self.multiscale_step == 0:
                    size = np.random.choice(self.img_sizes)

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


class IterationBasedBatchSampler(BatchSampler):
    """Wraps a BatchSampler, resampling from it until a specified number of
    iterations have been sampled."""

    def __init__(
        self, batch_sampler: BatchSampler, num_iterations: int, start_iter: int = 0
    ) -> None:
        """Initialize the IterationBasedBatchSampler.

        Args:
            batch_sampler (BatchSampler): The batch sampler to wrap.
            num_iterations (int): Number of iterations to sample.
            start_iter (int, optional): Starting iteration. Defaults to 0.
        """
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self) -> Iterator[List[Union[int, int]]]:
        """Yield batches of samples until the specified number of iterations is
        reached.

        Yields:
            Iterator[List[Union[int, int]]]: Batches of samples with size.
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
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self) -> int:
        """Return the number of iterations.

        Returns:
            int: Number of iterations.
        """
        return self.num_iterations

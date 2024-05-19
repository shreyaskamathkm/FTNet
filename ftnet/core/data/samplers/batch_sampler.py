#!/usr/bin/env python3
"""
Code Adapted from:
https://github.com/CaoWGG/multi-scale-training

"""

import numpy as np
from torch.utils.data import Sampler

__all__ = ["BatchSampler", "IterationBasedBatchSampler"]


class BatchSampler:
    def __init__(self, sampler, batch_size, drop_last, multiscale_step=None, img_sizes=None):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                f"torch.utils.data.Sampler, but got sampler={sampler}"
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value, but got " f"drop_last={drop_last}"
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        if multiscale_step is not None and multiscale_step < 1:
            raise ValueError(
                "multiscale_step should be > 0, but got " f"multiscale_step={multiscale_step}"
            )
        if multiscale_step is not None and img_sizes is None:
            raise ValueError(f"img_sizes must a list, but got img_sizes={img_sizes} ")

        self.multiscale_step = multiscale_step
        self.img_sizes = img_sizes

    def __iter__(self):
        num_batch = 0
        batch = []
        size = np.random.choice(self.img_sizes)
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

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class IterationBasedBatchSampler(BatchSampler):
    """Wraps a BatchSampler, resampling from it until a specified number of
    iterations have been sampled."""

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

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

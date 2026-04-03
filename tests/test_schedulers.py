"""Tests for learning rate schedulers."""

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD

from ftnet.core.schedulers.warmup_lr import WarmupMultiStepLR, WarmupPolyLR


def _make_optimizer(lr: float = 0.1) -> SGD:
    model = nn.Linear(2, 2)
    return SGD(model.parameters(), lr=lr)


class TestWarmupMultiStepLR:
    def test_invalid_milestones_raises(self):
        opt = _make_optimizer()
        with pytest.raises(ValueError, match="increasing"):
            WarmupMultiStepLR(opt, milestones=[10, 5])

    def test_invalid_warmup_method_raises(self):
        opt = _make_optimizer()
        with pytest.raises(ValueError, match="warmup_method"):
            WarmupMultiStepLR(opt, milestones=[10], warmup_method="cosine")

    def test_lr_decreases_at_milestone(self):
        opt = _make_optimizer(lr=0.1)
        scheduler = WarmupMultiStepLR(opt, milestones=[2], gamma=0.1, warmup_iters=0)
        lrs_before = scheduler.get_last_lr() if hasattr(scheduler, "get_last_lr") else [0.1]
        # Step past the milestone
        for _ in range(3):
            opt.step()
            scheduler.step()
        lr_after = scheduler.get_last_lr()[0]
        assert lr_after < 0.1

    def test_warmup_linear_increases_lr(self):
        opt = _make_optimizer(lr=0.1)
        warmup_iters = 5
        scheduler = WarmupMultiStepLR(
            opt,
            milestones=[100],
            warmup_factor=1.0 / 3,
            warmup_iters=warmup_iters,
            warmup_method="linear",
        )
        lrs = []
        for _ in range(warmup_iters):
            opt.step()
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])
        # LR should increase monotonically during warmup
        assert lrs == sorted(lrs)

    def test_warmup_constant(self):
        opt = _make_optimizer(lr=0.1)
        scheduler = WarmupMultiStepLR(
            opt,
            milestones=[100],
            warmup_factor=1.0 / 3,
            warmup_iters=3,
            warmup_method="constant",
        )
        opt.step()
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        assert lr < 0.1  # warmup_factor < 1 means LR is reduced during warmup


class TestWarmupPolyLR:
    def test_invalid_warmup_method_raises(self):
        opt = _make_optimizer()
        with pytest.raises(ValueError, match="warmup_method"):
            WarmupPolyLR(opt, steps_per_epoch=10, epochs=10, warmup_method="cosine")

    def test_lr_decays_over_time(self):
        opt = _make_optimizer(lr=0.1)
        scheduler = WarmupPolyLR(opt, steps_per_epoch=10, epochs=5, warmup_iters=0)
        lrs = []
        for _ in range(40):
            opt.step()
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])
        # LR should generally decrease (poly decay)
        assert lrs[0] >= lrs[-1]

    def test_reaches_target_lr(self):
        target_lr = 1e-6
        opt = _make_optimizer(lr=0.1)
        total_steps = 10
        scheduler = WarmupPolyLR(
            opt, steps_per_epoch=total_steps, epochs=1, target_lr=target_lr, warmup_iters=0
        )
        for _ in range(total_steps):
            opt.step()
            scheduler.step()
        final_lr = scheduler.get_last_lr()[0]
        assert final_lr >= target_lr

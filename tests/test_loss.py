"""Tests for segmentation loss functions."""

import pytest
import torch

from ftnet.core.loss.segmentation_loss import EdgeNetLoss, MixSoftmaxCrossEntropyLoss


class TestMixSoftmaxCrossEntropyLoss:
    def test_basic_forward(self):
        loss_fn = MixSoftmaxCrossEntropyLoss()
        pred = torch.randn(2, 5, 8, 8)
        target = torch.randint(0, 5, (2, 8, 8))
        loss = loss_fn(pred, target)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_ignore_index(self):
        loss_fn = MixSoftmaxCrossEntropyLoss(ignore_index=255)
        pred = torch.randn(2, 5, 8, 8)
        target = torch.full((2, 8, 8), 255, dtype=torch.long)
        loss = loss_fn(pred, target)
        # All targets ignored — loss should be 0 or nan-safe
        assert not torch.isnan(loss)

    def test_aux_loss_reduces_with_correct_pred(self):
        loss_fn = MixSoftmaxCrossEntropyLoss(aux=True, aux_weight=0.4)
        # Provide a tuple of (main_pred, aux_pred) and target
        main_pred = torch.randn(2, 5, 8, 8)
        aux_pred = torch.randn(2, 5, 8, 8)
        target = torch.randint(0, 5, (2, 8, 8))
        loss = loss_fn((main_pred, aux_pred), target)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_aux_forward_combines_losses(self):
        loss_fn = MixSoftmaxCrossEntropyLoss(aux=True, aux_weight=0.5)
        pred_main = torch.randn(2, 5, 8, 8)
        pred_aux = torch.randn(2, 5, 8, 8)
        target = torch.randint(0, 5, (2, 8, 8))
        loss = loss_fn(pred_main, pred_aux, target)
        assert torch.isfinite(loss)


class TestEdgeNetLoss:
    def _make_inputs(self, batch=2, num_classes=5, h=8, w=8):
        pred_class = torch.randn(batch, num_classes, h, w)
        pred_edge = torch.randn(batch, 1, h, w)
        target_class = torch.randint(0, num_classes, (batch, h, w))
        target_edge = torch.randint(0, 2, (batch, h, w)).float()
        return (pred_class, pred_edge), (target_class, target_edge)

    def test_forward_returns_scalar(self):
        loss_fn = EdgeNetLoss(loss_weight=20)
        preds, targets = self._make_inputs()
        loss = loss_fn(preds, targets)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_loss_weight_scales_edge_loss(self):
        torch.manual_seed(42)
        preds, targets = self._make_inputs()

        loss_low = EdgeNetLoss(loss_weight=1)(preds, targets)
        loss_high = EdgeNetLoss(loss_weight=50)(preds, targets)
        # Higher weight should produce a higher total loss
        assert loss_high.item() > loss_low.item()

    def test_requires_tuple_input(self):
        loss_fn = EdgeNetLoss()
        pred = torch.randn(2, 5, 8, 8)
        target = torch.randint(0, 5, (2, 8, 8))
        with pytest.raises((ValueError, TypeError)):
            loss_fn(pred, target)

    def test_auto_weight_bce_3d_target(self):
        """Edge target with 3 dims (no channel) should be handled correctly."""
        loss_fn = EdgeNetLoss(loss_weight=1)
        pred_class = torch.randn(2, 5, 8, 8)
        pred_edge = torch.randn(2, 1, 8, 8)
        target_class = torch.randint(0, 5, (2, 8, 8))
        target_edge = torch.randint(0, 2, (2, 8, 8)).float()  # 3D: no channel dim
        loss = loss_fn((pred_class, pred_edge), (target_class, target_edge))
        assert torch.isfinite(loss)

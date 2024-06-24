"""
MixSoftmaxCrossEntropyLoss adapted from
https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/core/utils/loss.py
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    """Softmax Cross Entropy Loss with optional auxiliary loss.

    Args:
        aux (bool, optional): Whether to include auxiliary loss. Defaults to False.
        aux_weight (float, optional): Weight of the auxiliary loss. Defaults to 0.2.
        ignore_index (int, optional): Index to ignore in the target. Defaults to -1.
        **kwargs: Additional arguments for nn.CrossEntropyLoss.
    """

    def __init__(
        self,
        aux: bool = False,
        aux_weight: float = 0.2,
        ignore_index: int = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__(ignore_index=ignore_index, **kwargs)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass with auxiliary loss.

        Args:
            *inputs (torch.Tensor): Predictions and target.

        Returns:
            torch.Tensor: Calculated loss.
        """
        *preds, target = inputs
        loss = super().forward(preds[0], target)
        for pred in preds[1:]:
            aux_loss = super().forward(pred, target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            *inputs (torch.Tensor): Predictions and target.

        Returns:
            torch.Tensor: Calculated loss.
        """
        if isinstance(inputs[0], tuple):
            preds, target = inputs
            inputs = tuple(list(preds) + [target])
        if self.aux:
            return self._aux_forward(*inputs)
        return super().forward(*inputs)


class EdgeNetLoss(nn.Module):
    """Custom loss function for edge detection networks.

    Args:
        loss_weight (int, optional): Weight for the auto-weighted BCE loss. Defaults to 1.
        **kwargs: Additional arguments for MixSoftmaxCrossEntropyLoss.
    """

    def __init__(self, loss_weight: int = 1, **kwargs: Any) -> None:
        super().__init__()
        self.cross_entropy = MixSoftmaxCrossEntropyLoss(**kwargs)
        self.loss_weight = loss_weight

    def auto_weight_bce(self, y_hat_log: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute auto-weighted binary cross entropy loss.

        Args:
            y_hat_log (torch.Tensor): Logits.
            y (torch.Tensor): Targets.

        Returns:
            torch.Tensor: Calculated loss.
        """
        if y.ndim == 3:
            y = y.unsqueeze(1)
        with torch.no_grad():
            beta = y.mean(dim=[2, 3], keepdim=True)
        logit_1 = F.logsigmoid(y_hat_log)
        logit_0 = F.logsigmoid(-y_hat_log)
        loss = -(1 - beta) * logit_1 * y - beta * logit_0 * (1 - y)
        return loss.mean()

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for the loss computation.

        Args:
            *inputs (torch.Tensor): Tuple containing predictions and targets.

        Returns:
            torch.Tensor: Total loss.
        """
        if isinstance(inputs[0], tuple):
            pred_class_map, pred_edge_map = inputs[0]
            target_maps, target_edges = inputs[1]
        else:
            raise ValueError("Expected inputs to be a tuple of predictions and targets")

        loss1 = self.loss_weight * self.auto_weight_bce(pred_edge_map, target_edges.float())
        loss2 = self.cross_entropy(pred_class_map, target_maps)
        logger.debug(
            f"loss_weight: {self.loss_weight} BCE Loss: {loss1}  Cross Entropy Loss: {loss2}"
        )
        return loss1 + loss2

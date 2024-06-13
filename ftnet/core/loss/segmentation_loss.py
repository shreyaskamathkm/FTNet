"""
MixSoftmaxCrossEntropyLoss adapted from
https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/core/utils/loss.py
"""

import logging
from typing import Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = ["get_segmentation_loss"]


class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
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
        *preds, target = inputs
        loss = super().forward(preds[0], target)
        for pred in preds[1:]:
            aux_loss = super().forward(pred, target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if isinstance(inputs[0], tuple):
            preds, target = inputs
            inputs = tuple(list(preds) + [target])
        if self.aux:
            return self._aux_forward(*inputs)
        return super().forward(*inputs)


class EdgeNetLoss(nn.Module):
    def __init__(self, loss_weight: int = 1, **kwargs: Any) -> None:
        super().__init__()
        self.cross_entropy = MixSoftmaxCrossEntropyLoss(**kwargs)
        self.loss_weight = loss_weight

    def auto_weight_bce(self, y_hat_log: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 3:
            y = y.unsqueeze(1)
        with torch.no_grad():
            beta = y.mean(dim=[2, 3], keepdim=True)
        logit_1 = F.logsigmoid(y_hat_log)
        logit_0 = F.logsigmoid(-y_hat_log)
        loss = -(1 - beta) * logit_1 * y - beta * logit_0 * (1 - y)
        return loss.mean()

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
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


def get_segmentation_loss(
    model: str, **kwargs: Any
) -> Union[MixSoftmaxCrossEntropyLoss, EdgeNetLoss]:
    model = model.lower()
    if "ftnet" in model:
        return EdgeNetLoss(**kwargs)
    return MixSoftmaxCrossEntropyLoss(**kwargs)

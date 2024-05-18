"""
MixSoftmaxCrossEntropyLoss adapted from
https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/core/utils/loss.py
"""

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

__all__ = ["MixSoftmaxCrossEntropyLoss", "EdgeNetLoss", "get_segmentation_loss"]


class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(
        self,
        aux: bool = False,
        aux_weight: float = 0.2,
        ignore_index: int = -1,
        **kwargs: Any,
    ) -> None:
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs: torch.Tensor, **kwargs: Any):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if isinstance(inputs[0], tuple):
            preds, target = tuple(inputs)
            inputs = tuple(list(preds) + [target])

        if self.aux:
            return self._aux_forward(*inputs)
        else:
            return super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs)


class EdgeNetLoss(nn.Module):
    # https://github.com/gasparian/PicsArtHack-binary-segmentation/blob/ecab001f334949d5082a79b8fbd1dc2fdb8b093e/utils.py#L217
    def __init__(
        self, bce_weight: float = None, loss_weight: int = 1, **kwargs: Any
    ) -> None:
        super(EdgeNetLoss, self).__init__()
        self.cross_entropy = MixSoftmaxCrossEntropyLoss(**kwargs)
        self.loss_weight = loss_weight

    def auto_weight_bce(self, y_hat_log, y):
        if y.ndim == 3:
            y = y.unsqueeze(1)
        with torch.no_grad():
            beta = y.mean(dim=[2, 3], keepdims=True)
        logit_1 = F.logsigmoid(y_hat_log)
        logit_0 = F.logsigmoid(-y_hat_log)
        loss = -(1 - beta) * logit_1 * y - beta * logit_0 * (1 - y)
        return loss.mean()

    def forward(self, *inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if isinstance(inputs[0], tuple):
            pred_class_map, pred_edge_map = inputs[0]
            target_maps, target_edges = inputs[1]

        loss1 = self.loss_weight * (
            self.auto_weight_bce(pred_edge_map, target_edges.float())
        )
        loss2 = self.cross_entropy(pred_class_map, target_maps)
        logger.debug(f"loss_weight: {self.loss_weight} loss1: {loss1}  loss2: {loss2} ")
        return loss1 + loss2


def get_segmentation_loss(model, **kwargs):
    model = model.lower()
    if "ftnet" in model:
        return EdgeNetLoss(**kwargs)
    else:
        return MixSoftmaxCrossEntropyLoss(**kwargs)

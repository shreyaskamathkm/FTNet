"""
Code  Adapted from
https://github.com/mseg-dataset/mseg-semantic

Evaluation Metrics for Semantic Segmentation
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Metric
from typing import Any, Optional


__all__ = ["pl_IOU", "intersectionAndUnionGPU", "intersectionAndUnion"]


class pl_IOU(Metric):
    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.add_state(
            "area_intersection", default=torch.zeros(num_classes), dist_reduce_fx="sum"
        )
        self.add_state(
            "area_union", default=torch.zeros(num_classes), dist_reduce_fx="sum"
        )
        self.add_state(
            "area_target", default=torch.zeros(num_classes), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.argmax(preds.long(), 1)
        assert preds.dim() in [1, 2, 3]
        preds = preds.view(-1)
        target = target.view(-1)
        assert preds.shape == target.shape
        preds[target == self.ignore_index] = self.ignore_index
        intersection = preds[preds == target]

        area_intersection = torch.histc(
            intersection, bins=self.num_classes, min=0, max=self.num_classes - 1
        )
        area_output = torch.histc(
            preds, bins=self.num_classes, min=0, max=self.num_classes - 1
        )
        area_target = torch.histc(
            target, bins=self.num_classes, min=0, max=self.num_classes - 1
        )

        self.area_union += area_output + area_target - area_intersection
        self.area_intersection += area_intersection
        self.area_target += area_target

    def compute_mean(self):
        mean_iou = torch.mean(self.area_intersection / (self.area_union + 1e-10))
        mean_accuracy = torch.mean(self.area_intersection / (self.area_target + 1e-10))
        all_accuracy = torch.sum(self.area_intersection) / torch.sum(
            self.area_target + 1e-10
        )
        return mean_iou, mean_accuracy, all_accuracy

    def compute(self):
        iou = self.area_intersection / (self.area_union + 1e-10)
        accuracy = self.area_intersection / (self.area_target + 1e-10)
        all_accuracy = torch.sum(self.area_intersection) / torch.sum(
            self.area_target + 1e-10
        )
        return iou, accuracy, all_accuracy


def intersectionAndUnionGPU(
    output: torch.Tensor,
    target: torch.Tensor,
    K: int,
    ignore_index: int = -1,
    test: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Note output and target sizes are N or N * L or N * H * W.

    Args:
    -   output: Pytorch tensor representing predicted label map,
            each value in range 0 to K - 1.
    -   target: Pytorch tensor representing ground truth label map,
            each value in range 0 to K - 1.
    -   K: integer number of possible classes
    -   ignore_index: integer representing class index to ignore
    Returns:
    -   area_intersection: 1d Pytorch tensor of length (K,) with counts
            for each of K classes, where pred & target matched
    -   area_union: 1d Pytorch tensor of length (K,) with counts
    -   area_target: 1d Pytorch tensor of length (K,) with bin counts
            for each of K classes, present in this GT label map.
    """

    if test:
        if output.shape != target.shape:
            output = F.interpolate(
                output,
                size=(target.shape[1], target.shape[2]),
                mode="bilinear",
                align_corners=True,
            )

    output = torch.argmax(output.long(), 1)
    assert output.dim() in [1, 2, 3]
    output = output.view(-1)
    target = target.view(-1)
    assert output.shape == target.shape
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnion(
    output: np.ndarray, target: np.ndarray, K: int, ignore_index: int = 255
) -> Tuple[np.array, np.array, np.array]:
    """Compute IoU on Numpy arrays on CPU. We will be reasoning about each
    matrix cell individually, so we can reshape (flatten) these arrays into
    column vectors and the evaluation result won’t change. Compare
    horizontally-corresponding cells. Wherever ground truth (target) pixels
    should be ignored, set prediction also to the ignore label. `intersection`
    represents values (class indices) in cells where.

    output and target are identical. We bin such correct class indices.
    Note output and target sizes are N or N * L or N * H * W
        Args:
        -   output: Numpy array representing predicted label map,
                each value in range 0 to K - 1.
        -   target: Numpy array representing ground truth label map,
                each value in range 0 to K - 1.
        -   K: integer number of possible classes
        -   ignore_index: integer representing class index to ignore
        Returns:
        -   area_intersection: 1d Numpy array of length (K,) with counts
                for each of K classes, where pred & target matched
        -   area_union: 1d Numpy array of length (K,) with counts
        -   area_target: 1d Numpy array of length (K,) with bin counts
                for each of K classes, present in this GT label map.
    """
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    # flatten the tensors to 1d arrays
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = 255
    intersection = output[np.where(output == target)[0]]
    # contains the number of samples in each bin.
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


if __name__ == "__main__":
    target = torch.randint(-1, 1, (10, 25, 25)).unsqueeze(0)
    pred = torch.tensor(target.clone())
    pred[2:5, 7:13, 9:15] = 1 - pred[2:5, 7:13, 9:15]

    # metric = SegmentationMetric(nclass=2)
    # pixAcc, miou = metric(pred, target)

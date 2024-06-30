# import seaborn as sns
from pathlib import Path
from typing import Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from ..core.dataloaders.base_dataloader import SegmentationDataset
from .visualize import get_color_palette


def plot_tensors(img: torch.Tensor, x: Union[str, None] = None) -> None:
    """Plot a tensor as an image.

    Args:
        img (torch.Tensor): Tensor to be plotted as an image.
        x (str, optional): Title of the plot. Defaults to None.
    """
    im = img.detach().cpu().numpy()
    dim = img.ndim
    plt.figure()
    if dim == 3:
        im = np.moveaxis(im, 0, 2)
        plt.imshow(im)
    else:
        plt.imshow(im, cmap=plt.get_cmap("gray"))
    if x is not None:
        plt.title(x)
    plt.show()


def plot_confusion_matrix(
    save_dir: Path, confusion_matrix: np.ndarray, class_names: list[str]
) -> None:
    """Plot and save the confusion matrix as a heatmap.

    Args:
        save_dir (Path): Directory to save the confusion matrix plot.
        confusion_matrix (np.ndarray): The confusion matrix to plot.
        class_names (list[str]): List of class names for the confusion matrix labels.
    """
    cmn = confusion_matrix / confusion_matrix.sum(1)[:, None]
    cmn[np.isnan(cmn)] = 0

    fig, _ = plt.subplots(figsize=(15, 15))
    sns.heatmap(
        cmn,
        annot=True,
        fmt=".1f",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    fig.savefig(
        save_dir / "confusion_matrix.png",
        bbox_inches="tight",
    )
    plt.close()


def save_all_images(
    original: np.ndarray,
    groundtruth: np.ndarray,
    prediction: np.ndarray,
    edge_map: np.ndarray,
    filenames: list[str],
    save_dir: Path,
    current_epoch: Union[int, None],
    dataset: SegmentationDataset,
    save_images_as_subplots: bool = False,
) -> None:
    """Save segmentation images with ground truth and predictions.

    Args:
        original (np.ndarray): The original input images.
        groundtruth (np.ndarray): The ground truth segmentation masks.
        prediction (np.ndarray): The predicted segmentation masks.
        edge_map (np.ndarray): The edge maps.
        filenames (List[str]): List of filenames for the images.
        save_dir (Path): Directory to save the images.
        current_epoch (Union[int, None]): The current epoch number.
        dataset (SegmentationDataset): The dataset instance providing mean and std for normalization.
        save_images_as_subplots (bool, optional): Whether to save images as subplots. Defaults to False.
    """
    base_path = save_dir / str(current_epoch) if current_epoch is not None else save_dir
    base_path.mkdir(parents=True, exist_ok=True)

    pred_path = base_path / "Predictions"
    pred_path.mkdir(parents=True, exist_ok=True)

    edge_path = base_path / "Edges"
    edge_path.mkdir(parents=True, exist_ok=True)

    for i, filename in enumerate(filenames):
        pred_image_path = pred_path / Path(filename).with_suffix(".png")
        edge_image_path = edge_path / Path(filename).with_suffix(".png")

        if save_images_as_subplots:
            fig, axes = plt.subplots(1, 4, figsize=(14, 7))

            # Original image
            axes[0].imshow(np.clip(original[i] * dataset.std + dataset.mean, 0, 1))
            axes[0].set_title("Original")

            # Ground truth
            axes[1].imshow(np.array(get_color_palette(groundtruth[i], dataset.NAME)))
            axes[1].set_title("Ground Truth")

            # Prediction
            axes[2].imshow(np.array(get_color_palette(prediction[i], dataset.NAME)))
            axes[2].set_title("Prediction")

            # Edge map
            axes[3].imshow(np.array(edge_map[i][0]))
            axes[3].set_title("Edge Map")

            for ax in axes:
                ax.axis("off")

            fig.savefig(save_dir / Path(filename).with_suffix(".png"), bbox_inches="tight")
            plt.close(fig)
        else:
            # Save prediction
            plt.imsave(pred_image_path, np.array(get_color_palette(prediction[i], dataset.NAME)))

            # Save edge map
            plt.imsave(edge_image_path, np.array(edge_map[i][0]))


def save_pred(
    save_dir: Path, filename: str, prediction: np.ndarray, edges: np.ndarray, dataset: str
) -> None:
    """Save predicted segmentation and edge map images.

    Args:
        save_dir (Path): Directory to save the images.
        filename (str): Filename for the images.
        prediction (np.ndarray): Predicted segmentation mask.
        edges (np.ndarray): Edge map.
        dataset (str): Dataset name.
    """
    prediction_path = save_dir / "Predictions"
    prediction_path.mkdir(parents=True, exist_ok=True)

    edge_path = save_dir / "Edges"
    edge_path.mkdir(parents=True, exist_ok=True)

    # Save prediction
    plt.imsave(
        prediction_path / Path(filename).with_suffix(".png"),
        get_color_palette(prediction, dataset),
    )

    # Save edge map
    plt.imsave(edge_path / Path(filename).with_suffix(".png"), edges)

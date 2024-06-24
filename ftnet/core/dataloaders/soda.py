import logging
from pathlib import Path
from typing import List

from .base_dataloader import SegmentationDataset
from .file_path_handler import FilePathHandler
from .transforms import NormalizationTransform

logger = logging.getLogger(__name__)


class SODADataset(SegmentationDataset):
    """Dataset class for SODA dataset.

    Attributes:
        NUM_CLASS (int): Number of classes in the dataset.
        IGNORE_INDEX (int): Index to ignore in loss calculations.
        NAME (str): Name of the dataset.
        mean (List[float]): Mean values for normalization.
        std (List[float]): Standard deviation values for normalization.
    """

    NUM_CLASS = 21
    IGNORE_INDEX = 255
    NAME = "soda"
    mean = [0.41079543, 0.41079543, 0.41079543]
    std = [0.18772296, 0.18772296, 0.18772296]

    def __init__(
        self,
        root: Path,
        split: str = "train",
        base_size: List[List[int]] = [[520, 520]],
        crop_size: List[List[int]] = [[480, 480]],
        mode: str = None,
        sobel_edges: bool = False,
    ) -> None:
        """Initialize SODA dataset.

        Args:
            root (Path): Root directory path of the dataset.
            split (str, optional): Split of the dataset ('train', 'val', 'test'). Defaults to "train".
            base_size (List[List[int]], optional): List of base sizes for resizing images. Defaults to [[520, 520]].
            crop_size (List[List[int]], optional): List of crop sizes for cropping images. Defaults to [[480, 480]].
            mode (str, optional): Mode of the dataset. Defaults to None.
            sobel_edges (bool, optional): Whether to apply Sobel edge detection. Defaults to False.

        Raises:
            ValueError: If the number of images does not match masks or edges, or if no images are found.
            RuntimeError: If no images are found in the specified root folder.
        """
        super().__init__(root, split, mode, base_size, crop_size, sobel_edges)

        self.images, self.mask_paths, self.edge_paths = FilePathHandler._get_pairs(root, split)
        if len(self.images) != len(self.mask_paths):
            raise ValueError("Mismatch between images and masks")
        if len(self.images) != len(self.edge_paths):
            raise ValueError("Mismatch between images and edges")
        if len(self.images) == 0:
            raise RuntimeError(f"Found 0 images in subfolder of: {root}")

        self.update_normalization(NormalizationTransform(self.mean, self.std))

    @property
    def class_names(self) -> List[str]:
        """Return the list of class names in the dataset.

        Returns:
            List[str]: List of class names.
        """
        return [
            "background",
            "person",
            "building",
            "tree",
            "road",
            "pole",
            "grass",
            "door",
            "table",
            "chair",
            "car",
            "bicycle",
            "lamp",
            "monitor",
            "trafficCone",
            "trash can",
            "animal",
            "fence",
            "sky",
            "river",
            "sidewalk",
        ]

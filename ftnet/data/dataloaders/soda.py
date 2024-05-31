import logging
from pathlib import Path
from typing import List, Tuple, Union

from PIL import Image

from .segbase import SegmentationDataset

logger = logging.getLogger(__name__)


class SODADataset(SegmentationDataset):
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
    ):
        super().__init__(root, split, mode, base_size, crop_size, sobel_edges)

        self.images, self.mask_paths, self.edge_paths = SegmentationDataset._get_pairs(root, split)
        if len(self.images) != len(self.mask_paths):
            raise ValueError("Mismatch between images and masks")
        if len(self.images) != len(self.edge_paths):
            raise ValueError("Mismatch between images and edges")
        if len(self.images) == 0:
            raise RuntimeError(f"Found 0 images in subfolder of: {root}")

    def __getitem__(self, index: Union[int, List, Tuple]):  # type: ignore
        scale = None
        if isinstance(index, (list, tuple)):
            index, scale = index

        # reading the images
        img = Image.open(self.images[index]).convert("RGB")

        if self.mode == "infer":
            img = self.normalize(img)
            return img, self.images[index].name

        # reading the mask
        mask = Image.open(self.mask_paths[index])

        # testing if sobel edge filter is required
        edge = None
        if not self.sobel_edges:
            edge = Image.open(self.edge_paths[index])

        # post process as required
        img, mask, edge = self.post_process(img, mask, edge, scale)

        return img, mask, edge, self.images[index].name

    @property
    def class_names(self) -> List[str]:
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


if __name__ == "__main__":
    from pathlib import Path

    from core.data.samplers import make_data_sampler, make_multiscale_batch_data_sampler
    from torch.utils.data import DataLoader

    from datasets import *

    data_kwargs = {"base_size": [[520, 520]], "crop_size": [[480, 480]]}

    train_dataset = SODADataset(
        root="/mnt/f/lab-work/FTNet/data/processed_dataset/",
        split="test",
        mode="train",
        **data_kwargs,
    )

    train_sampler = make_data_sampler(dataset=train_dataset, shuffle=True, distributed=False)

    train_batch_sampler = make_multiscale_batch_data_sampler(
        sampler=train_sampler, batch_size=1, multiscale_step=1, scales=1
    )

    loader_random_sample = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=0,
        pin_memory=True,
    )

    x = train_dataset.__getitem__(10)

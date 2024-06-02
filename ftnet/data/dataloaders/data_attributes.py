from dataclasses import dataclass


@dataclass
class ImageSize:
    """Class for storing image dimensions."""

    width: int
    height: int


@dataclass
class ImageSizes:
    """Class for storing base and crop sizes of images."""

    base_size: ImageSize
    crop_size: ImageSize

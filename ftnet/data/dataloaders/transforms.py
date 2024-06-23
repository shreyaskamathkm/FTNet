import random
from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

from .data_attributes import ImageSizes


class ImageTransform:
    """Base class for image transformations.

    Methods:
        img_transform(img: Image.Image) -> np.ndarray:
            Transform image to numpy array.
        mask_transform(mask: Image.Image) -> torch.Tensor:
            Transform mask to torch tensor.
        edge_transform(edge: Image.Image) -> np.ndarray:
            Transform edge to numpy array.
        im2double(im_: np.ndarray) -> np.ndarray:
            Convert image to double precision numpy array.
        np2tensor(array: np.ndarray) -> torch.Tensor:
            Convert numpy array to torch tensor.
    """

    def img_transform(self, img: Image.Image) -> np.ndarray:
        """Transform image to numpy array."""
        return np.array(img)

    def mask_transform(self, mask: Image.Image) -> torch.Tensor:
        """Transform mask to torch tensor."""
        return torch.LongTensor(np.array(mask).astype("int32"))

    def edge_transform(self, edge: Image.Image) -> np.ndarray:
        """Transform edge to numpy array."""
        return np.array(edge).astype("int32")

    def im2double(self, im_: np.ndarray) -> np.ndarray:
        """Convert image to double precision."""
        info = np.iinfo(im_.dtype)
        return im_.astype(float) / info.max

    def np2tensor(self, array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor."""
        if len(array.shape) == 3:
            np_transpose = np.ascontiguousarray(array.transpose((2, 0, 1)))
        else:
            np_transpose = np.ascontiguousarray(array[np.newaxis, :])
        return torch.from_numpy(np_transpose).float()


class NormalizationTransform:
    """Class to handle image normalization for segmentation dataset.

    Attributes:
        mean (List[float]): List of mean values for normalization.
        std (List[float]): List of standard deviation values for normalization.
    """

    def __init__(self, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std
        self.img_transform = ImageTransform()

    def normalize(self, img: Image.Image) -> torch.Tensor:
        """Normalize the input image.

        Args:
            img (Image.Image): Input image.

        Returns:
            torch.Tensor: Normalized image as torch tensor.
        """
        img = self.img_transform.im2double(np.array(img))  # type: ignore
        img = (img - self.mean) * np.reciprocal(self.std)  # type: ignore
        return self.img_transform.np2tensor(img).float()


class ResizingTransformations:
    """Class to handle image transformations for segmentation dataset.

    Attributes:
        sizes (List[ImageSizes]): List of image sizes for resizing.
    """

    def __init__(self, sizes: List[ImageSizes]):
        self.sizes = sizes

    def sync_transform(
        self,
        img: Image.Image,
        mask: Image.Image,
        edge: Image.Image,
        scale: Union[int, None] = None,
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """Synchronously transform training images with random augmentations.

        Args:
            img (Image.Image): Input image.
            mask (Image.Image): Input mask image.
            edge (Image.Image): Input edge image.
            scale (Union[int, None], optional): Scale index. Defaults to None.

        Returns:
            Tuple[Image.Image, Image.Image, Image.Image]: Transformed images.
        """
        # Random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            edge = edge.transpose(Image.FLIP_LEFT_RIGHT)

        if scale is not None:
            crop_size = self.sizes[scale].crop_size
            base_size = self.sizes[scale].base_size
        else:
            crop_size = self.sizes[0].crop_size
            base_size = self.sizes[0].base_size

        # Random scale (short edge)
        short_size = random.randint(
            int(min(base_size.width, base_size.height) * 0.5),
            int(min(base_size.width, base_size.height) * 2.0),
        )
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(h * ow / w)
        else:
            oh = short_size
            ow = int(w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        edge = edge.resize((ow, oh), Image.NEAREST)

        # Pad crop
        if short_size < max(crop_size.width, crop_size.height):
            padh = crop_size.height - oh if oh < crop_size.height else 0
            padw = crop_size.width - ow if ow < crop_size.width else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
            edge = ImageOps.expand(edge, border=(0, 0, padw, padh), fill=0)

        # Random crop
        w, h = img.size
        x1 = random.randint(0, w - crop_size.width)
        y1 = random.randint(0, h - crop_size.height)
        img = img.crop((x1, y1, x1 + crop_size.width, y1 + crop_size.height))
        mask = mask.crop((x1, y1, x1 + crop_size.width, y1 + crop_size.height))
        edge = edge.crop((x1, y1, x1 + crop_size.width, y1 + crop_size.height))

        # Gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        return img, mask, edge

    def val_sync_transform(
        self, img: Image.Image, mask: Image.Image, edge: Image.Image
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """Synchronously transform validation images.

        Args:
            img (Image.Image): Input image.
            mask (Image.Image): Input mask image.
            edge (Image.Image): Input edge image.

        Returns:
            Tuple[Image.Image, Image.Image, Image.Image]: Transformed images.
        """
        outsize = self.sizes[0].crop_size
        short_size = min(outsize.width, outsize.height)
        w, h = img.size

        if w > h:
            oh = short_size
            ow = int(w * oh / h)
        else:
            ow = short_size
            oh = int(h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        edge = edge.resize((ow, oh), Image.NEAREST)

        # Center crop
        w, h = img.size
        x1 = int(round((w - outsize.width) / 2.0))
        y1 = int(round((h - outsize.height) / 2.0))
        img = img.crop((x1, y1, x1 + outsize.width, y1 + outsize.height))
        mask = mask.crop((x1, y1, x1 + outsize.width, y1 + outsize.height))
        edge = edge.crop((x1, y1, x1 + outsize.width, y1 + outsize.height))
        return img, mask, edge

    def test_sync_transform(self, img: Image.Image) -> Tuple[Image.Image, int, int]:
        """Synchronously transform test images.

        Args:
            img (Image.Image): Input image.

        Returns:
            Tuple[Image.Image, int, int]: Transformed image and its width and height.
        """
        outsize = self.sizes[0]
        short_size = min(outsize.width, outsize.height)
        w, h = img.size

        if w > h:
            oh = short_size
            ow = int(w * oh / h)
        else:
            ow = short_size
            oh = int(h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        return img, w, h


class ScutsegTransform(ImageTransform):
    """Derived class with a custom mask transformation for ScutSEG dataset.

    Attributes:
        valid_classes (list): List of valid classes for the dataset.
        class_map (dict): Mapping of class IDs to encoded IDs.
    """

    def __init__(self, valid_classes: list, class_map: dict):
        self.valid_classes = valid_classes
        self.class_map = class_map

    def encode_segmap(self, mask: np.ndarray) -> np.ndarray:
        """Encode segmentation mask using class mapping.

        Args:
            mask (np.ndarray): Input mask as numpy array.

        Returns:
            np.ndarray: Encoded mask as numpy array.
        """
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def mask_transform(self, mask: Image.Image) -> torch.Tensor:
        """Transform mask using a custom class mapping.

        Args:
            mask (Image.Image): Input mask image.

        Returns:
            torch.Tensor: Transformed mask as torch tensor.
        """
        mask = self.encode_segmap(np.array(mask))
        return torch.LongTensor(mask.astype("int32"))


class CityscapeTransform(ImageTransform):
    """Derived class with a custom mask transformation for Cityscape dataset.

    Attributes:
        valid_classes (list): List of valid classes for the dataset.
        key (dict): Mapping of class indices.
        ignore_index (int): Index to ignore in loss calculations.
    """

    def __init__(self, valid_classes: list, key: dict, ignore_index: int):
        self.valid_classes = valid_classes
        self.key = key
        self._mapping = np.array(range(-1, len(key) - 1)).astype("int32")
        self.ignore_index = ignore_index

    def _class_to_index(self, mask: np.ndarray):
        """Map class labels to indices using key mapping.

        Args:
            mask (np.ndarray): Input mask as numpy array.

        Returns:
            np.ndarray: Mapped mask indices as numpy array.
        """
        # assert the value
        values = np.unique(mask)
        for value in values:
            assert value in self._mapping
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self.key[index].reshape(mask.shape)

    def mask_transform(self, mask: Image.Image) -> torch.Tensor:
        """Transform mask using a custom class mapping and handle ignore index.

        Args:
            mask (Image.Image): Input mask image.

        Returns:
            torch.Tensor: Transformed mask as torch tensor.
        """
        target = self._class_to_index(np.array(mask).astype("int32"))
        target[target == -1] = self.ignore_index
        return torch.LongTensor(np.array(target).astype("int32"))

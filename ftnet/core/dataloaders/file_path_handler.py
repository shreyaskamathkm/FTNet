import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


class FilePathHandler:
    """Class for handling file paths for datasets."""

    @staticmethod
    def _get_pairs(
        folder: Path, split: str = "train"
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        """Get pairs of image, mask, and edge paths for a given dataset split.

        Args:
            folder (Path): Root folder containing the dataset.
            split (str, optional): Split type ('train', 'val', 'test'). Defaults to 'train'.

        Returns:
            Tuple[List[Path], List[Path], List[Path]]: Lists of image paths, mask paths, and edge paths.

        Raises:
            ValueError: If the split type is unknown.
        """

        def get_path_pairs(
            img_folder: Path, mask_folder: Path, edge_folder: Path
        ) -> Tuple[List[Path], List[Path], List[Path]]:
            """Helper function to get paths of images, masks, and edges.

            Args:
                img_folder (Path): Folder containing images.
                mask_folder (Path): Folder containing masks.
                edge_folder (Path): Folder containing edges.

            Returns:
                Tuple[List[Path], List[Path], List[Path]]: Lists of image paths, mask paths, and edge paths.
            """
            img_folder = img_folder / split
            mask_folder = mask_folder / split
            edge_folder = edge_folder / split
            img_paths = []
            mask_paths = []
            edge_paths = []

            for imgpath in img_folder.rglob("*[.jpg, png]"):
                maskname = f"{imgpath.stem}.png"
                maskpath = mask_folder / maskname
                edgepath = edge_folder / maskname

                if maskpath.is_file() and edgepath.is_file():
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                    edge_paths.append(edgepath)
                else:
                    logger.warning(f"Cannot find the {imgpath}, {maskpath}, or {edgepath}")

            logger.info(f"Found {len(img_paths)} images in the folder {img_folder}")
            return img_paths, mask_paths, edge_paths

        if split in {"train", "val", "test"}:
            img_folder = folder / "image"
            mask_folder = folder / "mask"
            edge_folder = folder / "edges"
            return get_path_pairs(img_folder, mask_folder, edge_folder)

        raise ValueError("Split type unknown")

    @staticmethod
    def _get_city_pairs(
        folder: Path, split: str = "train"
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        """Get pairs of image, mask, and edge paths for the Cityscapes dataset.

        Args:
            folder (Path): Root folder containing the dataset.
            split (str, optional): Split type ('train', 'val', 'test'). Defaults to 'train'.

        Returns:
            Tuple[List[Path], List[Path], List[Path]]: Lists of image paths, mask paths, and edge paths.

        Raises:
            ValueError: If the split type is unknown.
        """

        def get_path_pairs(
            img_folder: Path, mask_folder: Path, edge_folder: Path
        ) -> Tuple[List[Path], List[Path], List[Path]]:
            """Helper function to get paths of images, masks, and edges for the
            Cityscapes dataset.

            Args:
                img_folder (Path): Folder containing images.
                mask_folder (Path): Folder containing masks.
                edge_folder (Path): Folder containing edges.

            Returns:
                Tuple[List[Path], List[Path], List[Path]]: Lists of image paths, mask paths, and edge paths.
            """
            img_paths = []
            mask_paths = []
            edge_paths = []

            for imgpath in img_folder.rglob("*[.jpg,png]"):
                maskname = imgpath.name.replace(
                    "leftImg8bit_synthesized_image.jpg", "gtFine_labelIds.png"
                )
                maskpath = mask_folder / maskname
                edgepath = edge_folder / maskname

                if maskpath.is_file() and edgepath.is_file():
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                    edge_paths.append(edgepath)
                else:
                    logger.warning(f"Cannot find the {imgpath}, {maskpath}, or {edgepath}")

            logger.info(f"Found {len(img_paths)} images in the folder {img_folder}")
            return img_paths, mask_paths, edge_paths

        if split in {"train", "val", "test"}:
            img_folder = folder / "image" / split
            mask_folder = folder / "mask" / split
            edge_folder = folder / "edges" / split
            return get_path_pairs(img_folder, mask_folder, edge_folder)

        raise ValueError("Split type unknown")

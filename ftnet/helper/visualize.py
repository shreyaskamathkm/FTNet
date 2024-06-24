from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

cityspalette = [
    128,
    64,
    128,
    244,
    35,
    232,
    70,
    70,
    70,
    102,
    102,
    156,
    190,
    153,
    153,
    153,
    153,
    153,
    250,
    170,
    30,
    220,
    220,
    0,
    107,
    142,
    35,
    152,
    251,
    152,
    0,
    130,
    180,
    220,
    20,
    60,
    255,
    0,
    0,
    0,
    0,
    142,
    0,
    0,
    70,
    0,
    60,
    100,
    0,
    80,
    100,
    0,
    0,
    230,
    119,
    11,
    32,
]

color_palettes = {
    "scutseg": {
        0: np.array([0, 0, 0]),
        1: np.array([128, 64, 128]),
        2: np.array([60, 20, 220]),
        3: np.array([0, 0, 255]),
        4: np.array([142, 0, 0]),
        5: np.array([70, 0, 0]),
        6: np.array([153, 153, 190]),
        7: np.array([35, 142, 107]),
        8: np.array([100, 60, 0]),
        9: np.array([153, 153, 153]),
    },
    "soda": {
        0: np.array([0, 0, 0]),
        1: np.array([128, 64, 128]),
        2: np.array([244, 35, 232]),
        3: np.array([70, 70, 70]),
        4: np.array([102, 102, 156]),
        5: np.array([190, 153, 153]),
        6: np.array([153, 153, 153]),
        7: np.array([250, 170, 30]),
        8: np.array([220, 220, 0]),
        9: np.array([107, 142, 35]),
        10: np.array([0, 130, 180]),
        11: np.array([220, 20, 60]),
        12: np.array([60, 20, 220]),
        13: np.array([0, 0, 255]),
        14: np.array([142, 0, 0]),
        15: np.array([70, 0, 0]),
        16: np.array([153, 153, 190]),
        17: np.array([35, 142, 107]),
        18: np.array([100, 60, 0]),
        19: np.array([255, 0, 0]),
        20: np.array([0, 64, 128]),
    },
    "mfn": {
        0: np.array([0, 0, 0]),
        1: np.array([128, 64, 128]),
        2: np.array([244, 35, 232]),
        3: np.array([70, 70, 70]),
        4: np.array([102, 102, 156]),
        5: np.array([190, 153, 153]),
        6: np.array([153, 153, 153]),
        7: np.array([250, 170, 30]),
        8: np.array([220, 220, 0]),
    },
}


def print_iou(
    iou: np.ndarray,
    mean_pixel_acc: float,
    class_names: Optional[List[str]] = None,
    show_no_back: bool = False,
) -> None:
    """Print the Intersection over Union (IoU) and mean pixel accuracy.

    Args:
        iu (np.ndarray): Array containing IoU values for each class.
        mean_pixel_acc (float): Mean pixel accuracy value.
        class_names (Optional[List[str]], optional): List of class names. Defaults to None.
        show_no_back (bool, optional): Whether to show mean IoU without background class. Defaults to False.
    """
    n = iou.size
    lines = []
    for i in range(n):
        cls = f"Class {i + 1}:" if class_names is None else f"{i + 1} {class_names[i]}"
        lines.append(f"{cls:8s}: {iou[i] * 100:.3f}%")
    mean_IU = np.nanmean(iou)
    mean_IU_no_back = np.nanmean(iou[1:])
    if show_no_back:
        lines.append(
            f"mean_IU: {mean_IU * 100:.3f}% || mean_IU_no_back: {mean_IU_no_back * 100:.3f}% || mean_pixel_acc: {mean_pixel_acc * 100:.3f}%"
        )
    else:
        lines.append(
            f"mean_IU: {mean_IU * 100:.3f}% || mean_pixel_acc: {mean_pixel_acc * 100:.3f}%"
        )
    lines.append("=================================================")
    print("\n".join(lines))


def set_img_color(
    img: np.ndarray,
    label: np.ndarray,
    colors: Dict[int, np.ndarray],
    background: int = 0,
    show255: bool = False,
) -> np.ndarray:
    """Set the colors for the image based on the labels.

    Args:
        img (np.ndarray): The image to be colored.
        label (np.ndarray): The label mask for the image.
        colors (Dict[int, np.ndarray]): Dictionary of colors for each label.
        background (int, optional): The background label. Defaults to 0.
        show255 (bool, optional): Whether to show 255 label. Defaults to False.

    Returns:
        np.ndarray: The colored image.
    """
    for i, color in colors.items():
        if i != background:
            img[label == i] = color
    if show255:
        img[label == 255] = 255
    return img


def show_prediction(
    img: np.ndarray, pred: np.ndarray, colors: Dict[int, np.ndarray], background: int = 0
) -> np.ndarray:
    """Show the prediction on the image.

    Args:
        img (np.ndarray): The original image.
        pred (np.ndarray): The prediction mask.
        colors (Dict[int, np.ndarray]): Dictionary of colors for each label.
        background (int, optional): The background label. Defaults to 0.

    Returns:
        np.ndarray: The image with the prediction overlay.
    """
    im = np.array(img, np.uint8)
    set_img_color(im, pred, colors, background)
    return im


def show_colorful_images(prediction: np.ndarray, palettes: np.ndarray) -> None:
    """Show colorful images from the prediction.

    Args:
        prediction (np.ndarray): The prediction mask.
        palettes (np.ndarray): The color palettes for the labels.
    """
    im = Image.fromarray(palettes[prediction.astype("uint8").squeeze()])
    im.show()


def save_colorful_images(
    prediction: np.ndarray, filename: str, output_dir: Path, palettes: np.ndarray
) -> None:
    """Save colorful images from the prediction.

    Args:
        prediction (np.ndarray): The prediction mask.
        filename (str): The filename to save the image.
        output_dir (Path): The directory to save the image.
        palettes (np.ndarray): The color palettes for the labels.
    """
    im = Image.fromarray(palettes[prediction.astype("uint8").squeeze()])
    output_dir.mkdir(parents=True, exist_ok=True)
    im.save(output_dir / filename)


def decode_segmap(
    label_mask: np.ndarray, label_colours: Dict[int, np.ndarray], n_classes: int
) -> np.ndarray:
    """Decode segmentation map into RGB image.

    Args:
        label_mask (np.ndarray): The label mask.
        label_colours (Dict[int, np.ndarray]): Dictionary of colors for each label.
        n_classes (int): Number of classes.

    Returns:
        np.ndarray: The RGB image.
    """
    r, g, b = (label_mask.copy() for _ in range(3))
    for ll in range(n_classes):
        r[label_mask == ll] = label_colours[ll][0]
        g[label_mask == ll] = label_colours[ll][1]
        b[label_mask == ll] = label_colours[ll][2]
    rgb = np.stack([r, g, b], axis=-1)
    return rgb.astype(np.uint8)


def get_color_palette(npimg: np.ndarray, dataset: str = "soda") -> Image:
    """Get the color palette for the image based on the dataset.

    Args:
        npimg (np.ndarray): The input image.
        dataset (str, optional): The dataset name. Defaults to "soda".

    Returns:
        Image: The image with the color palette applied.
    """
    dataset = dataset.lower()
    if dataset == "cityscapes":
        out_img = Image.fromarray(npimg.astype("uint8"))
        out_img.putpalette(cityspalette)
    elif dataset in color_palettes:
        out_img = decode_segmap(
            npimg, label_colours=color_palettes[dataset], n_classes=len(color_palettes[dataset])
        )
    else:
        out_img = Image.fromarray(npimg.astype("uint8"))
        out_img.putpalette(generic)
    return out_img


def _generatepalette(num_cls: int) -> List[int]:
    """Generate a color palette.

    Args:
        num_cls (int): Number of classes.

    Returns:
        List[int]: The generated color palette.
    """
    palette = [0] * (num_cls * 3)
    for j in range(num_cls):
        lab = j
        for i in range(8):
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            lab >>= 3
    return palette


generic = _generatepalette(1000)

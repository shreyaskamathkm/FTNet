import os

import numpy as np
from PIL import Image

__all__ = [
    "get_color_palette",
    "print_iou",
    "set_img_color",
    "show_prediction",
    "show_colorful_images",
    "save_colorful_images",
]

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


scutseg = {
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
}


soda = {
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
}


mfn = {
    0: np.array([0, 0, 0]),
    1: np.array([128, 64, 128]),
    2: np.array([244, 35, 232]),
    3: np.array([70, 70, 70]),
    4: np.array([102, 102, 156]),
    5: np.array([190, 153, 153]),
    6: np.array([153, 153, 153]),
    7: np.array([250, 170, 30]),
    8: np.array([220, 220, 0]),
}


def print_iou(iu, mean_pixel_acc, class_names=None, show_no_back=False):
    n = iu.size
    lines = []
    for i in range(n):
        if class_names is None:
            cls = (
                "Class %d:" % (i + 1) if class_names is None else "%d %s" % (i + 1, class_names[i])
            )
        lines.append("%-8s: %.3f%%" % (cls, iu[i] * 100))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    if show_no_back:
        lines.append(
            f"mean_IU: {mean_IU * 100:.3f}% || mean_IU_no_back: {mean_IU_no_back * 100:3f}% || mean_pixel_acc: {mean_pixel_acc * 100:3f}%"
        )
    else:
        lines.append(
            f"mean_IU: {mean_IU * 100:.3f}% || mean_pixel_acc: {mean_pixel_acc * 100:3f}%"
        )
    lines.append("=================================================")
    line = "\n".join(lines)

    print(line)


def set_img_color(img, label, colors, background=0, show255=False):
    for i in range(len(colors)):
        if i != background:
            img[np.where(label == i)] = colors[i]
    if show255:
        img[np.where(label == 255)] = 255

    return img


def show_prediction(img, pred, colors, background=0):
    im = np.array(img, np.uint8)
    set_img_color(im, pred, colors, background)
    out = np.array(im)

    return out


def show_colorful_images(prediction, palettes):
    im = Image.fromarray(palettes[prediction.astype("uint8").squeeze()])
    im.show()


def save_colorful_images(prediction, filename, output_dir, palettes):
    """
    :param prediction: [B, H, W, C]
    """
    im = Image.fromarray(palettes[prediction.astype("uint8").squeeze()])
    fn = os.path.join(output_dir, filename)
    out_dir = os.path.split(fn)[0]
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    im.save(fn)


def decode_segmap(label_mask, label_colours, n_classes):
    # label_colours = self.dataset.get_class_colors()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll][0]
        g[label_mask == ll] = label_colours[ll][1]
        b[label_mask == ll] = label_colours[ll][2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb * 255


def get_color_palette(npimg, dataset="idine"):
    """Visualize image.

    Parameters
    ----------
    npimg : numpy.ndarray
        Single channel image with shape `H, W, 1`.
    dataset : str, default: 'idine'
    Returns
    -------
    out_img : PIL.Image
        Image with color palette
    """
    if "cityscapes" in dataset.lower():
        out_img = Image.fromarray(npimg.astype("uint8"))
        out_img.putpalette(cityspalette)
        return out_img

    if dataset.lower() == "soda":
        out_img = decode_segmap(npimg, label_colours=soda, n_classes=len(soda))
        return out_img.astype(np.uint8)

    if dataset.lower() == "mfn":
        out_img = decode_segmap(npimg, label_colours=mfn, n_classes=len(mfn))
        return out_img.astype(np.uint8)

    if dataset.lower() == "scutseg":
        out_img = decode_segmap(npimg, label_colours=scutseg, n_classes=len(scutseg))
        return out_img.astype(np.uint8)

    out_img = Image.fromarray(npimg.astype("uint8"))
    out_img.putpalette(generic)
    return out_img


def _generatepalette(num_cls):
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab > 0:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
    return palette


# mfn1 = _generatepalette(9)
# soda1 = _generatepalette(21)
generic = _generatepalette(1000)

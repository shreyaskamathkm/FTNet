from typing import Tuple

import torch


def resize_by_scaled_short_side(
    image: torch.tensor,
    base_size: int,
    scale: float,
) -> torch.tensor:
    """Equivalent to ResizeShort(), but functional, instead of OOP paradigm,
    and w/ scale param.

    Args:
        image: Numpy array of shape ()
        scale: scaling factor for image

    Returns:
        image_scaled:
    """
    n, c, h, w = image.shape
    short_size = round(scale * base_size)
    new_h = short_size
    new_w = short_size
    # Preserve the aspect ratio
    if h > w:
        new_h = round(short_size / float(w) * h)
    else:
        new_w = round(short_size / float(h) * w)
    return torch.nn.functional.interpolate(image, (new_w, new_h), mode="bilinear")


def pad_to_crop_sz(
    image: torch.tensor, crop_h: int, crop_w: int, mean: Tuple[float, float, float]
) -> Tuple[torch.tensor, int, int]:
    """Network input should be at least crop size, so we pad using mean values
    if provided image is too small. No rescaling is performed here. We use
    cv2.copyMakeBorder to copy the source image into the middle of a
    destination image. The areas to the left, to the right, above and below the
    copied source image will be filled with extrapolated pixels, in this case
    the provided mean pixel intensity.

    Args:
        image:
        crop_h: integer representing crop height
        crop_w: integer representing crop width

    Returns:
        image: Numpy array of shape (crop_h x crop_w) representing a
               square image, with short side of square is at least crop size.
         pad_h_half: half the number of pixels used as padding along height dim
         pad_w_half" half the number of pixels used as padding along width dim
    """
    n, ch, orig_h, orig_w = image.shape

    pad_h = max(crop_h - orig_h, 0)
    pad_w = max(crop_w - orig_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        temp = []
        for c in range(ch):
            temp.append(
                torch.nn.functional.pad(
                    input=image[:, c, :, :],
                    pad=(
                        pad_w_half,
                        pad_w - pad_w_half,
                        pad_h_half,
                        pad_h - pad_h_half,
                    ),
                    mode="constant",
                    value=mean[c],
                )
            )  # (padding_left, padding_right, \text{padding\_top}, \text{padding\_bottom})padding_top, padding_bottom)
        image = torch.stack(temp, dim=1)

    return image, pad_h_half, pad_w_half

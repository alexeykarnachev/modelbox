from typing import Literal

import numpy as np
from PIL import Image

from modelbox.utils import infer_triton_image

ImageToBackgroundMaskModelName = Literal["rmbg"]


def _transform_image_to_background_mask_rmbg(image: Image.Image) -> Image.Image:
    """https://huggingface.co/briaai/RMBG-1.4"""

    out_image = infer_triton_image(
        image,
        model_name="bg_removal",
        input_name="input",
        output_name="mask",
    )

    # --------------------------------------------------------------------
    # Prepare output
    out_image = out_image.resize(
        (image.width, image.height), resample=Image.Resampling.BILINEAR
    )
    out_image_np = np.array(out_image)
    out_image_np = (
        (out_image_np - out_image_np.min())
        / (out_image_np.max() - out_image_np.min() + 1e-6)
        * 255
    )
    out_image_np = out_image_np.astype(np.uint8)

    out_image = Image.fromarray(out_image_np, mode="L")
    return out_image


def transform_image_to_background_mask(
    image: Image.Image, model_name: ImageToBackgroundMaskModelName
) -> Image.Image:
    if model_name == "rmbg":
        out_image = _transform_image_to_background_mask_rmbg(image)

    return out_image

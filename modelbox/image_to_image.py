from typing import Literal

import numpy as np
from PIL import Image

from modelbox.utils import infer_triton_image

ImageToImageModelName = Literal["depth_pro", "rmbg"]


def _infer_rmbg(image: Image.Image) -> Image.Image:
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


def _infer_depthpro(image: Image.Image) -> Image.Image:
    """https://huggingface.co/apple/DepthPro"""

    out_image = infer_triton_image(
        image,
        model_name="depth_pro",
        input_name="images",
        output_name="depth",
    )

    # --------------------------------------------------------------------
    # Prepare output
    out_image = out_image.resize(
        (image.width, image.height), resample=Image.Resampling.BILINEAR
    )
    out_image_np = np.array(out_image)
    out_image_np = 1.0 / np.clip(out_image_np, a_min=1e-4, a_max=1e4)
    out_image_np = np.clip(out_image_np, 0, max(out_image_np.max(), 1e-6))
    out_image_np = (
        255.0
        * (out_image_np - out_image_np.min())
        / (out_image_np.max() - out_image_np.min() + 1e-6)
    )
    out_image_np = out_image_np.astype(np.uint8)

    out_image = Image.fromarray(out_image_np, mode="L")
    return out_image


def infer_image_to_image(
    image: Image.Image, model_name: ImageToImageModelName
) -> Image.Image:
    if model_name == "depth_pro":
        out_image = _infer_depthpro(image)
    elif model_name == "rmbg":
        out_image = _infer_rmbg(image)

    return out_image

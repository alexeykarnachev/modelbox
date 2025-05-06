import numpy as np
from PIL import Image


def prepare_input_image(pil_image):
    input_img = pil_image.convert("RGB")
    input_img_np = np.array(
        input_img.resize((1024, 1024), resample=Image.Resampling.BILINEAR)
    )
    input_img_np = input_img_np / 255.0
    input_img_np = (input_img_np - 0.5) / 0.5
    input_img_np = input_img_np.transpose(2, 0, 1)
    input_img_np = input_img_np[None, ...].astype(np.float32)

    return input_img, input_img_np


def post_process_mask(mask_np, input_img):
    mask_img = Image.fromarray(mask_np)
    mask_img = mask_img.resize(
        (input_img.width, input_img.height), resample=Image.Resampling.BILINEAR
    )
    mask_np = np.array(mask_img)
    if mask_np.max() > 0:  # Avoid division by zero
        mask_np = (
            (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-6) * 255
        )
    mask_np = mask_np.astype(np.uint8)
    result_img = input_img.copy()
    result_img.putalpha(Image.fromarray(mask_np))

    return result_img

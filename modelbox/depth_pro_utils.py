import numpy as np
from PIL import Image


def prepare_input_image(image_path):
    input_img = Image.open(image_path).convert("RGB")
    input_img_np = np.array(
        input_img.resize((1536, 1536), resample=Image.Resampling.BILINEAR)
    )
    input_img_np = (input_img_np / 255.0 - 0.5) / 0.5
    input_img_np = input_img_np.transpose(2, 0, 1)
    input_img_np = input_img_np[None, ...].astype(np.float16)
    return input_img, input_img_np


def post_process_depthmap(output_depth_np, input_img):
    output_img = Image.fromarray(output_depth_np)
    output_img = output_img.resize(
        (input_img.width, input_img.height), resample=Image.Resampling.BILINEAR
    )
    output_img_np = np.array(output_img)
    output_img_np = 1.0 / np.clip(output_img_np, a_min=1e-4, a_max=1e4)
    output_img_np = np.clip(output_img_np, 0, max(output_img_np.max(), 1e-6))
    output_img_np = (
        255.0
        * (output_img_np - output_img_np.min())
        / (output_img_np.max() - output_img_np.min() + 1e-6)
    )
    output_img_np = output_img_np.astype(np.uint8)
    return Image.fromarray(output_img_np).convert("RGB")

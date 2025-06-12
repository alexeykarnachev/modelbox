import base64
import io

import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image

from modelbox.settings import settings


def base64_to_image(image_base64: str) -> Image.Image:
    image_data = base64.b64decode(image_base64)
    buffer = io.BytesIO(image_data)
    image = Image.open(buffer)

    image = image.convert("RGB")
    return image


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return image_base64


def _prepare_inp_image_np(
    image: Image.Image, size: tuple[int, int] | None, dtype: np.dtype
) -> np.ndarray:
    inp_image = image.convert("RGB")

    if size is not None:
        inp_image_np = np.array(
            inp_image.resize(size, resample=Image.Resampling.BILINEAR)
        )
    else:
        inp_image_np = np.array(inp_image)

    inp_image_np = inp_image_np / 255.0
    inp_image_np = (inp_image_np - 0.5) / 0.5
    inp_image_np = inp_image_np.transpose(2, 0, 1)
    inp_image_np = inp_image_np[None, ...].astype(dtype)

    return inp_image_np


def infer_triton_image(
    image: Image.Image,
    model_name: str,
    input_name: str,
    output_name: str,
) -> Image.Image:
    str_to_dtype = {"FP32": np.float32, "FP16": np.float16}
    triton = grpcclient.InferenceServerClient(url=settings.triton_url, verbose=False)

    # --------------------------------------------------------------------
    # Extract input parameters from the model's config
    model_config = triton.get_model_config(model_name, as_json=True)
    if model_config is None:
        raise ValueError(f"Failed to obtain model config for model '{model_name}'")

    input_dtype = ""
    input_size = ()
    for inp in model_config["config"]["input"]:
        if inp["name"] == input_name:
            input_dtype = inp["data_type"].replace("TYPE_", "")
            input_size = tuple(int(x) for x in inp["dims"][-2:][::-1])

    if len(input_size) != 2 or input_dtype not in str_to_dtype:
        raise ValueError(f"Failed to extract input parameters for model '{model_name}'")

    # --------------------------------------------------------------------
    # Prepare triton input
    input_dtype_np = str_to_dtype[input_dtype]
    inp_image_np = _prepare_inp_image_np(image, size=input_size, dtype=input_dtype_np)
    inputs = [grpcclient.InferInput(input_name, inp_image_np.shape, input_dtype)]
    inputs[0].set_data_from_numpy(inp_image_np)
    outputs = [grpcclient.InferRequestedOutput(output_name)]

    # --------------------------------------------------------------------
    # Infer
    results = triton.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    out_image_np = np.squeeze(results.as_numpy(output_name))  # type: ignore

    out_image = Image.fromarray(out_image_np)
    return out_image

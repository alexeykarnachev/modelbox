import base64
import io

import numpy as np
import tritonclient.grpc as grpcclient
from litestar import Litestar, post
from litestar.datastructures import UploadFile
from litestar.params import Body
from loguru import logger
from PIL import Image
from pydantic import BaseModel

from modelbox import bg_removal_utils, depth_pro_utils
from modelbox.bg_removal_utils import post_process_mask
from modelbox.settings import settings


class DepthProResult(BaseModel):
    image: str
    f_px: float


class BackgroundRemovalResult(BaseModel):
    image: str


class ImageRequest(BaseModel):
    image: UploadFile

    class Config:
        arbitrary_types_allowed = True


_body_multipart = Body(media_type="multipart/form-data")


@post("/infer_depth_pro", media_type="application/json")
async def infer_depth_pro(
    data: ImageRequest = _body_multipart,
) -> DepthProResult:
    logger.debug("Received image file")
    try:
        image_data = await data.image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        logger.debug(f"Image size: {pil_image.size}")

        client = grpcclient.InferenceServerClient(
            url=settings.triton_url, verbose=False
        )
        logger.debug(f"Connected to Triton at {settings.triton_url}")

        input_img, input_img_np = depth_pro_utils.prepare_input_image(pil_image)
        logger.debug(f"Input image prepared: {input_img_np.shape}")

        inputs = [grpcclient.InferInput("images", input_img_np.shape, "FP16")]
        inputs[0].set_data_from_numpy(input_img_np)
        outputs = [
            grpcclient.InferRequestedOutput("depth"),
            grpcclient.InferRequestedOutput("f_px"),
        ]

        logger.debug("Running inference")
        results = client.infer(model_name="depth_pro", inputs=inputs, outputs=outputs)

        output_depth_np = np.squeeze(results.as_numpy("depth"))  # type: ignore
        output_f_px_np = np.squeeze(results.as_numpy("f_px"))  # type: ignore
        logger.debug(f"Depth shape: {output_depth_np.shape}, Focal: {output_f_px_np}")

        output_img = depth_pro_utils.post_process_depthmap(output_depth_np, input_img)
        logger.debug("Depth map processed")

        buffer = io.BytesIO()
        output_img.save(buffer, format="PNG")
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return DepthProResult(image=base64_image, f_px=float(output_f_px_np))
    except Exception as e:
        logger.exception(f"Inference failed: {e}")
        raise


@post("/infer_bg_removal", media_type="application/json")
async def infer_bg_removal(
    data: ImageRequest = _body_multipart,
) -> BackgroundRemovalResult:
    logger.debug("Received image file for background removal")
    try:
        image_data = await data.image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        logger.debug(f"Image size: {pil_image.size}")

        client = grpcclient.InferenceServerClient(
            url=settings.triton_url, verbose=False
        )
        logger.debug(f"Connected to Triton at {settings.triton_url}")

        input_img, input_img_np = bg_removal_utils.prepare_input_image(pil_image)
        logger.debug(f"Input image prepared: {input_img_np.shape}")

        inputs = [grpcclient.InferInput("input", input_img_np.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_img_np)
        outputs = [grpcclient.InferRequestedOutput("mask")]

        logger.debug("Running inference")
        results = client.infer(model_name="bg_removal", inputs=inputs, outputs=outputs)

        mask_np = np.squeeze(results.as_numpy("mask"))  # type: ignore
        logger.debug(f"Mask shape: {mask_np.shape}")

        output_img = post_process_mask(mask_np, input_img)
        logger.debug("Background removed")

        buffer = io.BytesIO()
        output_img.save(buffer, format="PNG")
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return BackgroundRemovalResult(image=base64_image)
    except Exception as e:
        logger.exception(f"Background removal failed: {e}")
        raise


app = Litestar(
    route_handlers=[infer_depth_pro, infer_bg_removal],
    request_max_body_size=100 * 1024 * 1024,
)

import base64
import io
from dataclasses import dataclass

import numpy as np
import tritonclient.grpc as grpcclient
from litestar import Litestar, post
from litestar.params import Body
from loguru import logger
from PIL import Image
from pydantic import BaseModel

from modelbox.depth_pro_utils import post_process_depthmap, prepare_input_image
from modelbox.settings import settings


@dataclass
class DepthProResult:
    image: str  # Base64 encoded image
    f_px: float


class Base64ImageRequest(BaseModel):
    image: str


@post("/infer_depth_pro", media_type="application/json")
async def infer_depth_pro(
    data: Base64ImageRequest = Body(),
) -> DepthProResult:
    logger.debug("Received base64 image")
    try:
        # Decode base64 image
        image_data = base64.b64decode(data.image)
        pil_image = Image.open(io.BytesIO(image_data))
        logger.debug(f"Decoded image size: {pil_image.size}")

        client = grpcclient.InferenceServerClient(
            url=settings.triton_url, verbose=False
        )
        logger.debug(f"Connected to Triton at {settings.triton_url}")

        input_img, input_img_np = prepare_input_image(pil_image)
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

        output_img = post_process_depthmap(output_depth_np, input_img)
        logger.debug("Depth map processed")

        # Convert output image to base64
        buffer = io.BytesIO()
        output_img.save(buffer, format="PNG")
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return DepthProResult(image=base64_image, f_px=float(output_f_px_np))
    except Exception as e:
        logger.exception(f"Inference failed: {str(e)}")
        raise


app = Litestar(
    route_handlers=[infer_depth_pro],
    request_body_max_size=100 * 1024 * 1024  # 100MB max request size
)

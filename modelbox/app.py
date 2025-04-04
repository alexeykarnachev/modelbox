from dataclasses import dataclass

import numpy as np
import tritonclient.grpc as grpcclient
from litestar import Litestar, post
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body
from loguru import logger

from modelbox.depth_pro_utils import post_process_depthmap, prepare_input_image
from modelbox.settings import settings


@dataclass
class DepthProResult:
    image: bytes
    f_px: float


@post("/infer_depth_pro", media_type="application/json")
async def infer_depth_pro(
    data: UploadFile = Body(media_type=RequestEncodingType.MULTI_PART),
) -> DepthProResult:
    logger.debug(f"Received file: {data.filename}")
    try:
        client = grpcclient.InferenceServerClient(
            url=settings.triton_url, verbose=False
        )
        logger.debug(f"Connected to Triton at {settings.triton_url}")

        input_img, input_img_np = prepare_input_image(data.file)
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

        return DepthProResult(image=output_img.tobytes(), f_px=float(output_f_px_np))
    except Exception as e:
        logger.exception(f"Inference failed: {str(e)}")
        raise


app = Litestar([infer_depth_pro])

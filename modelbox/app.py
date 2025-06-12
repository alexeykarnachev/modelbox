from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

from modelbox.to_background_mask import (
    ImageToBackgroundMaskModelName,
    transform_image_to_background_mask,
)
from modelbox.to_depth_mask import (
    ImageToDepthMaskModelName,
    transform_image_to_depth_mask,
)
from modelbox.utils import base64_to_image, image_to_base64

app = FastAPI(max_request_body_size=100 * 1024 * 1024)


class ImageToDepthRequest(BaseModel):
    image_base64: str
    model_name: ImageToDepthMaskModelName


class ImageToBackgroundMaskRequest(BaseModel):
    image_base64: str
    model_name: ImageToBackgroundMaskModelName


class ImageToDepthMaskResult(BaseModel):
    image_base64: str


class ImageToBackgroundMaskResult(BaseModel):
    image_base64: str


@app.post("/image_to_depth_mask")
async def image_to_depth_mask(request: ImageToDepthRequest) -> ImageToDepthMaskResult:
    try:
        inp_image = base64_to_image(request.image_base64)
        out_image = transform_image_to_depth_mask(
            inp_image,
            model_name=request.model_name,
        )
        out_image_base64 = image_to_base64(out_image)
        return ImageToDepthMaskResult(image_base64=out_image_base64)
    except Exception as e:
        logger.info(e)
        raise e


@app.post("/image_to_background_mask")
async def image_to_background_mask(
    request: ImageToBackgroundMaskRequest,
) -> ImageToBackgroundMaskResult:
    inp_image = base64_to_image(request.image_base64)
    out_image = transform_image_to_background_mask(
        inp_image,
        model_name=request.model_name,
    )
    out_image_base64 = image_to_base64(out_image)
    return ImageToBackgroundMaskResult(image_base64=out_image_base64)

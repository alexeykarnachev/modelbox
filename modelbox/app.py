from typing import get_args

from fastapi import FastAPI
from pydantic import BaseModel

from modelbox.image_to_image import ImageToImageModelName, infer_image_to_image
from modelbox.utils import base64_to_image, image_to_base64

app = FastAPI(max_request_body_size=100 * 1024 * 1024)


class MediaModelRequest(BaseModel):
    src: str
    result: str

    model_name: MediaModelName


@app.get("/image_to_image")
async def image_to_image_get() -> list[str]:
    model_names = get_args(ImageToImageModelName)
    return list(model_names)


@app.post("/image_to_image")
async def image_to_image_post(request: ImageToImageRequest) -> ImageToImageResult:
    inp_image = base64_to_image(request.image_base64)
    out_image = infer_image_to_image(inp_image, model_name=request.model_name)
    out_image_base64 = image_to_base64(out_image)
    return ImageToImageResult(image_base64=out_image_base64)

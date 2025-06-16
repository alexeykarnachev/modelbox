import shutil
import uuid
from pathlib import Path
from typing import Literal, get_args

from fastapi import FastAPI, File, HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel
from starlette.responses import FileResponse

from modelbox.settings import settings
from modelbox.triton_inferer import MediaModelName, TritonInferer

app = FastAPI(max_request_body_size=100 * 1024 * 1024)
inferer = TritonInferer(url=settings.triton_url)

MediaDestination = Literal["local", "base64"]


class MediaModelRequest(BaseModel):
    file_name: str
    model_name: MediaModelName


class MediaModelResult(BaseModel):
    file_name: str


class ModelboxInfo(BaseModel):
    media_model_names: list[str]


@app.get("/info")
async def info_get() -> ModelboxInfo:
    return ModelboxInfo(
        media_model_names=get_args(MediaModelName),  # type: ignore
    )


_file = File()


@app.post("/file")
async def file_post(file: UploadFile = _file) -> str:
    extension = Path(file.filename or "").suffix
    file_name = str(uuid.uuid4()) + extension
    file_path = Path(settings.media_dir / file_name)
    file_path.parent.mkdir(exist_ok=True, parents=True)

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info(f"File from client received: {file_path}")
    return file_name


@app.get("/file/{file_name}")
async def get_file(file_name: str) -> FileResponse:
    file_path = Path(settings.media_dir / file_name)

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    logger.info(f"Returning file to client: {file_path}")
    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=file_name,
    )


@app.post("/media_model")
async def media_model_post(request: MediaModelRequest) -> MediaModelResult:
    extension = Path(request.file_name).suffix
    input_file_path = settings.media_dir / request.file_name

    output_file_name = str(uuid.uuid4()) + extension
    output_file_path = Path(settings.media_dir / output_file_name)
    logger.info(
        f"Inferencing '{request.model_name}' on file '{input_file_path}' and saving it in '{output_file_path}'"
    )

    inferer.infer_media_model(
        input_file_path=input_file_path,
        output_file_path=output_file_path,
        model_name=request.model_name,
    )

    return MediaModelResult(file_name=output_file_name)

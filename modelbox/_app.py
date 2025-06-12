import base64
import io
import tempfile
import time
from pathlib import Path

import cv2
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


class DepthProVideoResult(BaseModel):
    video: str
    f_px: list[float]


class BackgroundRemovalResult(BaseModel):
    image: str


class BackgroundRemovalVideoResult(BaseModel):
    video: str


class ImageRequest(BaseModel):
    image: UploadFile

    class Config:
        arbitrary_types_allowed = True


class VideoRequest(BaseModel):
    video: UploadFile

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
        logger.debug("Background mask processed")

        buffer = io.BytesIO()
        output_img.save(buffer, format="PNG")
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return BackgroundRemovalResult(image=base64_image)
    except Exception as e:
        logger.exception(f"Background removal failed: {e}")
        raise


@post("/infer_depth_pro_video", media_type="application/json")
async def infer_depth_pro_video(
    data: VideoRequest = _body_multipart,
) -> DepthProVideoResult:
    logger.debug("Received video file")
    temp_input_file = None
    temp_output_file = None
    cap = None
    out = None
    try:
        video_data = await data.video.read()

        temp_dir = Path(tempfile.gettempdir())
        temp_input_file = temp_dir / f"video_input_{int(time.time() * 1000)}.mp4"
        with temp_input_file.open("wb") as f:
            f.write(video_data)

        cap = cv2.VideoCapture(str(temp_input_file))
        if not cap.isOpened():
            logger.error("Failed to open video")
            raise ValueError("Invalid video file")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.debug(f"Video properties: {frame_width}x{frame_height}, {fps} fps")

        temp_output_file = temp_dir / f"video_output_{int(time.time() * 1000)}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        out = cv2.VideoWriter(
            str(temp_output_file),
            fourcc,
            fps,
            (frame_width, frame_height),
            True,
        )
        if not out.isOpened():
            logger.error("Failed to initialize VideoWriter")
            raise ValueError("Failed to initialize video writer")

        client = grpcclient.InferenceServerClient(
            url=settings.triton_url, verbose=False
        )
        logger.debug(f"Connected to Triton at {settings.triton_url}")

        f_px = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            logger.debug(f"Processing frame {frame_count}")

            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            input_img, input_img_np = depth_pro_utils.prepare_input_image(pil_image)
            logger.debug(f"Input image prepared: {input_img_np.shape}")

            inputs = [grpcclient.InferInput("images", input_img_np.shape, "FP16")]
            inputs[0].set_data_from_numpy(input_img_np)
            outputs = [
                grpcclient.InferRequestedOutput("depth"),
                grpcclient.InferRequestedOutput("f_px"),
            ]

            results = client.infer(
                model_name="depth_pro", inputs=inputs, outputs=outputs
            )

            output_depth_np = np.squeeze(results.as_numpy("depth"))  # type: ignore
            output_f_px_np = np.squeeze(results.as_numpy("f_px"))  # type: ignore
            logger.debug(
                f"Depth shape: {output_depth_np.shape}, Focal: {output_f_px_np}"
            )

            output_img = depth_pro_utils.post_process_depthmap(
                output_depth_np, input_img
            )

            output_frame = cv2.cvtColor(np.array(output_img), cv2.COLOR_RGB2BGR)
            out.write(output_frame)
            f_px.append(float(output_f_px_np))

        cap.release()
        out.release()

        with temp_output_file.open("rb") as f:
            base64_video = base64.b64encode(f.read()).decode("utf-8")
        logger.debug(f"Processed {frame_count} frames")

        return DepthProVideoResult(video=base64_video, f_px=f_px)

    except Exception as e:
        logger.exception(f"Video depth processing failed: {e}")
        raise
    finally:
        if cap is not None:
            cap.release()

        if out is not None:
            out.release()

        for temp_file in [temp_input_file, temp_output_file]:
            if temp_file and temp_file.exists():
                temp_file.unlink()


@post("/infer_bg_removal_video", media_type="application/json")
async def infer_bg_removal_video(
    data: VideoRequest = _body_multipart,
) -> BackgroundRemovalVideoResult:
    logger.debug("Received video file for background removal")
    temp_input_file = None
    temp_output_file = None
    cap = None
    out = None
    try:
        video_data = await data.video.read()

        temp_dir = Path(tempfile.gettempdir())
        temp_input_file = temp_dir / f"video_input_bg_{int(time.time() * 1000)}.mp4"
        with temp_input_file.open("wb") as f:
            f.write(video_data)

        cap = cv2.VideoCapture(str(temp_input_file))
        if not cap.isOpened():
            logger.error("Failed to open video")
            raise ValueError("Invalid video file")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.debug(f"Video properties: {frame_width}x{frame_height}, {fps} fps")

        temp_output_file = temp_dir / f"video_output_bg_{int(time.time() * 1000)}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        out = cv2.VideoWriter(
            str(temp_output_file),
            fourcc,
            fps,
            (frame_width, frame_height),
            False,
        )
        if not out.isOpened():
            logger.error("Failed to initialize VideoWriter")
            raise ValueError("Failed to initialize video writer")

        client = grpcclient.InferenceServerClient(
            url=settings.triton_url, verbose=False
        )
        logger.debug(f"Connected to Triton at {settings.triton_url}")

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            logger.debug(f"Processing frame {frame_count}")

            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            input_img, input_img_np = bg_removal_utils.prepare_input_image(pil_image)
            logger.debug(f"Input image prepared: {input_img_np.shape}")

            inputs = [grpcclient.InferInput("input", input_img_np.shape, "FP32")]
            inputs[0].set_data_from_numpy(input_img_np)
            outputs = [grpcclient.InferRequestedOutput("mask")]

            results = client.infer(
                model_name="bg_removal", inputs=inputs, outputs=outputs
            )

            mask_np = np.squeeze(results.as_numpy("mask"))  # type: ignore
            logger.debug(f"Mask shape: {mask_np.shape}")

            output_img = post_process_mask(mask_np, input_img)
            output_frame = np.array(output_img)
            out.write(output_frame)

        cap.release()
        out.release()

        with temp_output_file.open("rb") as f:
            base64_video = base64.b64encode(f.read()).decode("utf-8")
        logger.debug(f"Processed {frame_count} frames")

        return BackgroundRemovalVideoResult(video=base64_video)

    except Exception as e:
        logger.exception(f"Video background removal failed: {e}")
        raise
    finally:
        if cap is not None:
            cap.release()

        if out is not None:
            out.release()

        for temp_file in [temp_input_file, temp_output_file]:
            if temp_file and temp_file.exists():
                temp_file.unlink()


app = Litestar(
    route_handlers=[
        infer_depth_pro,
        infer_bg_removal,
        infer_depth_pro_video,
        infer_bg_removal_video,
    ],
    request_max_body_size=100 * 1024 * 1024,
)

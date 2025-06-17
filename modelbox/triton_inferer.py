import mimetypes
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from loguru import logger
from numpy._typing import NDArray
from PIL import Image
from pydantic import BaseModel

mimetypes.init()

MediaModelName = Literal["depth_pro", "bg_removal"]


def _iterate_video_frames(cap: cv2.VideoCapture) -> Iterator[np.ndarray]:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        yield frame


class TritonInferer:
    class _ModelIO(BaseModel):
        inputs: list[grpcclient.InferInput] = []
        outputs: list[grpcclient.InferRequestedOutput] = []

        model_config = {"arbitrary_types_allowed": True}

    def __init__(self, url: str) -> None:
        self._client = grpcclient.InferenceServerClient(url=url, verbose=False)
        self._model_io: dict[str, TritonInferer._ModelIO] = {}

    def _ensure_model_io(self, model_name: str) -> None:
        if model_name not in self._model_io:
            model_io = self._ModelIO()
            model_config = cast(
                dict, self._client.get_model_config(model_name, as_json=True)
            )["config"]

            for input_config in model_config["input"]:
                shape = tuple(map(int, input_config["dims"]))
                infer_input = grpcclient.InferInput(
                    name=input_config["name"],
                    shape=shape,
                    datatype=input_config["data_type"].replace("TYPE_", ""),
                )
                model_io.inputs.append(infer_input)

            for output_config in model_config["output"]:
                infer_output = grpcclient.InferRequestedOutput(
                    name=output_config["name"]
                )
                model_io.outputs.append(infer_output)

            self._model_io[model_name] = model_io

    def _infer_on_input_data(
        self,
        model_name: str,
        input_data: list[np.ndarray],
    ) -> list[NDArray]:
        self._ensure_model_io(model_name)

        inputs = self._model_io[model_name].inputs
        outputs = self._model_io[model_name].outputs

        dtype_map = {
            "FP16": np.float16,
            "FP32": np.float32,
            "FP64": np.float64,
            "INT8": np.int8,
            "INT16": np.int16,
            "INT32": np.int32,
            "INT64": np.int64,
            "UINT8": np.uint8,
            "UINT16": np.uint16,
            "UINT32": np.uint32,
            "UINT64": np.uint64,
            "BOOL": np.bool_,
            "STRING": np.string_,
        }

        for i, data in enumerate(input_data):
            expected_dtype = inputs[i].datatype()
            data = data.astype(dtype_map[expected_dtype])
            inputs[i].set_data_from_numpy(data)

        result = self._client.infer(
            model_name=model_name, inputs=inputs, outputs=outputs
        )
        assert result is not None

        result_np = [result.as_numpy(output.name()) for output in outputs]
        return result_np  # type: ignore

    def infer_media_model(
        self,
        input_file_path: Path,
        output_file_path: Path,
        model_name: MediaModelName,
    ):
        self._ensure_model_io(model_name)

        # ----------------------------------------------------------------
        # Define preprocessing and postprocessing functions
        input_shape = self._model_io[model_name].inputs[0].shape()  # e.g., [1, 3, H, W]
        model_h, model_w = input_shape[2], input_shape[3]  # type: ignore

        if model_name == "bg_removal":
            assert model_h == 1024 and model_w == 1024, (
                "bg_removal expects 1024x1024 input"
            )

            def preproc_fn(frame: np.ndarray) -> np.ndarray:
                frame_resized = cv2.resize(
                    frame, (model_w, model_h), interpolation=cv2.INTER_LINEAR
                )
                frame_float = frame_resized.astype(np.float32) / 255.0
                frame_normalized = frame_float - 0.5
                return frame_normalized.transpose(2, 0, 1)

            def postproc_fn(output: np.ndarray) -> np.ndarray:
                output_resized = cv2.resize(
                    output.squeeze(),
                    (original_size[1], original_size[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                output_norm = (output_resized - output_resized.min()) / (
                    output_resized.max() - output_resized.min() + 1e-8
                )
                return (output_norm * 255).astype(np.uint8)

        elif model_name == "depth_pro":
            assert model_h == 1536 and model_w == 1536, (
                "DepthPro expects 1536x1536 input"
            )

            def preproc_fn(frame: np.ndarray) -> np.ndarray:
                frame_resized = cv2.resize(
                    frame, (model_w, model_h), interpolation=cv2.INTER_LINEAR
                )
                frame_float = frame_resized.astype(np.float32) / 255.0
                return frame_float.transpose(2, 0, 1)

            def postproc_fn(output: np.ndarray) -> np.ndarray:
                output_resized = cv2.resize(
                    output.squeeze(),
                    (original_size[1], original_size[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                output_norm = (output_resized - output_resized.min()) / (
                    output_resized.max() - output_resized.min() + 1e-8
                )
                return 1 - (output_norm * 255).astype(np.uint8)

        # ----------------------------------------------------------------
        # Process based on file type
        file_type = mimetypes.guess_type(input_file_path)[0] or ""

        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_type.startswith("image"):
            input_image_np = cv2.imread(str(input_file_path))
            if input_image_np is None:
                raise ValueError(f"Failed to read image: {input_file_path}")
            original_size = input_image_np.shape[:2]  # (height, width)

            input_frame = preproc_fn(input_image_np)[np.newaxis, ...]

            output_data = self._infer_on_input_data(model_name, [input_frame])
            output_frame = postproc_fn(output_data[0][0])
            output_image = Image.fromarray(output_frame)

            output_image.save(output_file_path, format="PNG")

        elif file_type.startswith("video"):
            cap = cv2.VideoCapture(str(input_file_path))
            if not cap.isOpened():
                raise ValueError(f"Failed to read video: {input_file_path}")
            original_size = (
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            )
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
            out = cv2.VideoWriter(
                str(output_file_path), fourcc, fps, (original_size[1], original_size[0])
            )

            for i, input_frame in enumerate(_iterate_video_frames(cap)):
                input_frame = preproc_fn(input_frame)[np.newaxis, ...]

                output_data = self._infer_on_input_data(model_name, [input_frame])
                output_frame = postproc_fn(output_data[0][0])
                output_frame = cv2.cvtColor(output_frame, cv2.COLOR_GRAY2BGR)

                out.write(output_frame)

                n_frames_processed = i + 1
                if n_frames_processed % 10 == 0:
                    logger.debug(f"Frames processed: {i + 1}/{n_frames}")

            cap.release()
            out.release()

        else:
            raise ValueError(f"Unsupported file type: {file_type}")

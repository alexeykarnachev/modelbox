import ctypes
import os

import numpy as np
import onnxruntime as ort
import typer
from PIL import Image

cudnn_path = ".venv/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9"
if os.path.exists(cudnn_path):
    ctypes.CDLL(cudnn_path)


app = typer.Typer()


@app.command()
def infer(
    model_path: str = typer.Option(
        "depth_pro.onnx", help="Path to the depth_pro.onnx model file"
    ),
    image_path: str = typer.Option(..., help="Path to the input image"),
    output_path: str = typer.Option("depth_map.png", help="Path to save the depth map"),
    cuda: bool = typer.Option(False, help="Use CUDA device for inference"),
):
    # --------------------------------------------------------------------
    # Load model
    typer.echo(f"Loading model on {'gpu' if cuda else 'cpu'}")
    provider = "CUDAExecutionProvider" if cuda else "CPUExecutionProvider"
    session = ort.InferenceSession(model_path, providers=[provider])

    # --------------------------------------------------------------------
    # Prepare input image
    typer.echo("Preparing input image")
    input_img = Image.open(image_path).convert("RGB")
    input_img_np = np.array(
        input_img.resize((1536, 1536), resample=Image.Resampling.BILINEAR)
    )
    input_img_np = (input_img_np / 255.0 - 0.5) / 0.5
    input_img_np = input_img_np.transpose(2, 0, 1)
    input_img_np = input_img_np[None, ...].astype(np.float16)

    # --------------------------------------------------------------------
    # Infer the model
    typer.echo("Estimating depthmap")
    model_outputs = session.run(None, {"images": input_img_np})
    output_img_np = np.squeeze(model_outputs[0])
    output_f_px_np = np.squeeze(model_outputs[1])
    typer.echo(f"Estimated focal length: {output_f_px_np} px")

    # --------------------------------------------------------------------
    # Post-process depthmap
    typer.echo("Post-processing depthmap (resize, rescale)")
    output_img = Image.fromarray(output_img_np)
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

    # --------------------------------------------------------------------
    # Save depthmap
    output_img = Image.fromarray(output_img_np).convert("RGB")
    output_img.save(output_path)

    typer.echo(f"Success! Depthmap saved: {output_path}")


if __name__ == "__main__":
    app()

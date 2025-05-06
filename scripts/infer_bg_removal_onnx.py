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
    onnx_file_path: str = typer.Option(
        "model_repository/bg_removal/1/model.onnx",
        help="Path to the background removal model onnx file",
    ),
    image_path: str = typer.Option(..., help="Path to the input image"),
    output_path: str = typer.Option("no_bg.png", help="Path to save the output image"),
    cuda: bool = typer.Option(False, help="Use CUDA device for inference"),
):
    # --------------------------------------------------------------------
    # Load model
    typer.echo(f"Loading model on {'gpu' if cuda else 'cpu'}")
    provider = "CUDAExecutionProvider" if cuda else "CPUExecutionProvider"
    session = ort.InferenceSession(onnx_file_path, providers=[provider])

    # --------------------------------------------------------------------
    # Prepare input image
    typer.echo("Preparing input image")
    input_img = Image.open(image_path).convert("RGB")
    input_img_np = np.array(
        input_img.resize((1024, 1024), resample=Image.Resampling.BILINEAR)
    )
    input_img_np = input_img_np / 255.0
    input_img_np = (input_img_np - 0.5) / 0.5
    input_img_np = input_img_np.transpose(2, 0, 1)
    input_img_np = input_img_np[None, ...].astype(np.float32)

    # --------------------------------------------------------------------
    # Infer the model
    typer.echo("Removing background")
    model_outputs = session.run(None, {"input": input_img_np})
    mask_np = np.squeeze(model_outputs[0])

    # --------------------------------------------------------------------
    # Post-process mask
    typer.echo("Post-processing mask")
    mask_img = Image.fromarray(mask_np)
    mask_img = mask_img.resize(
        (input_img.width, input_img.height), resample=Image.Resampling.BILINEAR
    )

    # Convert mask to proper alpha channel format (0-255)
    mask_np = np.array(mask_img)
    # Normalize to 0-255 range
    if mask_np.max() > 0:  # Avoid division by zero
        mask_np = (
            (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-6) * 255
        )
    mask_np = mask_np.astype(np.uint8)

    # Apply mask as alpha channel
    result_img = input_img.copy()
    result_img.putalpha(Image.fromarray(mask_np))

    # --------------------------------------------------------------------
    # Save output image
    result_img.save(output_path)

    typer.echo(f"Success! Image with removed background saved: {output_path}")


if __name__ == "__main__":
    app()

import ctypes
import os

import numpy as np
import tritonclient.grpc as grpcclient
import typer
from PIL import Image

cudnn_path = ".venv/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9"
if os.path.exists(cudnn_path):
    ctypes.CDLL(cudnn_path)


app = typer.Typer()


@app.command()
def infer(
    triton_url: str = typer.Option(
        "localhost:8001", help="Triton server gRPC URL (host:port)"
    ),
    model_name: str = typer.Option("depth_pro", help="Name of the model in Triton"),
    image_path: str = typer.Option(..., help="Path to the input image"),
    output_path: str = typer.Option("depth_map.png", help="Path to save the depth map"),
):
    # --------------------------------------------------------------------
    # Connect to Triton server
    typer.echo(f"Connecting to Triton server at {triton_url}")
    client = grpcclient.InferenceServerClient(url=triton_url, verbose=False)

    # --------------------------------------------------------------------
    # Prepare input image
    typer.echo("Preparing input image")
    input_img = Image.open(image_path).convert("RGB")
    input_img_np = np.array(
        input_img.resize((1536, 1536), resample=Image.Resampling.BILINEAR)
    )
    input_img_np = (input_img_np / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1]
    input_img_np = input_img_np.transpose(2, 0, 1)  # HWC -> CHW
    input_img_np = input_img_np[None, ...].astype(np.float16)  # Add batch dim, FP16

    # --------------------------------------------------------------------
    # Set up Triton input and output
    inputs = [grpcclient.InferInput("images", input_img_np.shape, "FP16")]
    inputs[0].set_data_from_numpy(input_img_np)

    outputs = [
        grpcclient.InferRequestedOutput("depth"),
        grpcclient.InferRequestedOutput("f_px"),
    ]

    # --------------------------------------------------------------------
    # Infer the model
    typer.echo("Estimating depthmap via Triton")
    results = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    assert results is not None
    output_depth_np = results.as_numpy("depth")  # [1, 1536, 1536]
    output_f_px_np = results.as_numpy("f_px")  # [1, 1]
    output_depth_np = np.squeeze(output_depth_np)  # type: ignore [1536, 1536]
    output_f_px_np = np.squeeze(output_f_px_np)  # type: ignore [1,]
    typer.echo(f"Estimated focal length: {output_f_px_np:.2f} px")

    # --------------------------------------------------------------------
    # Post-process depthmap
    typer.echo("Post-processing depthmap (resize, rescale)")
    output_img = Image.fromarray(output_depth_np)
    output_img = output_img.resize(
        (input_img.width, input_img.height), resample=Image.Resampling.BILINEAR
    )
    output_img_np = np.array(output_img)
    output_img_np = 1.0 / np.clip(output_img_np, a_min=1e-4, a_max=1e4)  # Inverse depth
    output_img_np = np.clip(output_img_np, 0, max(output_img_np.max(), 1e-6))  # Clip
    output_img_np = (
        255.0
        * (output_img_np - output_img_np.min())
        / (output_img_np.max() - output_img_np.min() + 1e-6)
    )  # Normalize to [0, 255]
    output_img_np = output_img_np.astype(np.uint8)

    # --------------------------------------------------------------------
    # Save depthmap
    output_img = Image.fromarray(output_img_np).convert("RGB")
    output_img.save(output_path)

    typer.echo(f"Success! Depthmap saved: {output_path}")


if __name__ == "__main__":
    app()

import base64
import io

import numpy as np
import tritonclient.grpc as grpcclient
import typer
from PIL import Image

from modelbox.bg_removal_utils import post_process_mask, prepare_input_image
from modelbox.settings import settings

app = typer.Typer()


@app.command()
def infer(
    image_path: str = typer.Option(..., help="Path to the input image"),
    output_path: str = typer.Option("no_bg.png", help="Path to save the output image"),
    triton_url: str = typer.Option(
        settings.triton_url, help="Triton server URL with port"
    ),
):
    # --------------------------------------------------------------------
    # Load and prepare input image
    typer.echo("Preparing input image")
    input_img = Image.open(image_path).convert("RGB")
    input_img, input_img_np = prepare_input_image(input_img)

    # --------------------------------------------------------------------
    # Set up Triton client
    typer.echo(f"Connecting to Triton at {triton_url}")
    client = grpcclient.InferenceServerClient(url=triton_url, verbose=False)

    # --------------------------------------------------------------------
    # Prepare request
    inputs = [grpcclient.InferInput("input", input_img_np.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_img_np)
    outputs = [grpcclient.InferRequestedOutput("mask")]

    # --------------------------------------------------------------------
    # Infer the model
    typer.echo("Removing background")
    results = client.infer(model_name="bg_removal", inputs=inputs, outputs=outputs)
    mask_np = np.squeeze(results.as_numpy("mask"))  # type: ignore

    # --------------------------------------------------------------------
    # Post-process mask
    typer.echo("Post-processing mask")
    result_img = post_process_mask(mask_np, input_img)

    # --------------------------------------------------------------------
    # Save output image
    result_img.save(output_path)
    typer.echo(f"Success! Image with removed background saved: {output_path}")


if __name__ == "__main__":
    app()
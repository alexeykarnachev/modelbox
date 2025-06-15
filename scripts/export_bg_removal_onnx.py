from pathlib import Path

import torch
import typer
from transformers import AutoModelForImageSegmentation

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def export_to_onnx(
    onnx_file_path: Path = typer.Option(
        "model_repository/bg_removal/1/model.onnx",
        help="Path to the output onnx model file",
    ),
    cuda: bool = typer.Option(False, help="Use cuda device for export"),
):
    device = torch.device("cuda" if cuda else "cpu")

    # --------------------------------------------------------------------
    # Load model from huggingface
    typer.echo("Loading BRIA Background Removal model from Hugging Face")
    model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-1.4", trust_remote_code=True
    )

    model = model.to(device)
    model = model.eval()

    # --------------------------------------------------------------------
    # Create directory for the model
    onnx_file_path = Path(onnx_file_path)
    onnx_file_path.parent.mkdir(exist_ok=True, parents=True)

    # --------------------------------------------------------------------
    # Define a wrapper class for ONNX export
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, images):
            result = self.model(images)
            mask = result[0][0]
            return mask.reshape(-1, 1, 1024, 1024)

    wrapper_model = ModelWrapper(model)

    # --------------------------------------------------------------------
    # Export model
    typer.echo(f"Exporting model to {onnx_file_path}")
    input_tensor = torch.randn(1, 3, 1024, 1024, device=device)

    torch.onnx.export(
        wrapper_model,
        input_tensor,
        onnx_file_path,
        input_names=["input"],
        output_names=["mask"],
        opset_version=17,
        do_constant_folding=True,
    )
    typer.echo(f"Model exported successfully to {onnx_file_path}")


if __name__ == "__main__":
    app()

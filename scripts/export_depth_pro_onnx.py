from pathlib import Path

import torch
import torch.nn as nn
import typer
from depth_pro.network.decoder import MultiresConvDecoder
from depth_pro.network.encoder import DepthProEncoder
from depth_pro.network.fov import FOVNetwork
from depth_pro.network.vit_factory import VIT_CONFIG_DICT, create_vit
from huggingface_hub import hf_hub_download

app = typer.Typer(pretty_exceptions_show_locals=False)


class DepthPro(nn.Module):
    """DepthPro network."""

    def __init__(
        self,
        encoder: DepthProEncoder,
        decoder: MultiresConvDecoder,
        last_dims: tuple[int, int],
        fov_encoder: nn.Module,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        dim_decoder = decoder.dim_decoder
        self.head = nn.Sequential(
            nn.Conv2d(
                dim_decoder, dim_decoder // 2, kernel_size=3, stride=1, padding=1
            ),
            nn.ConvTranspose2d(
                in_channels=dim_decoder // 2,
                out_channels=dim_decoder // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            ),
            nn.Conv2d(
                dim_decoder // 2,
                last_dims[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(last_dims[0], last_dims[1], kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

        self.fov = FOVNetwork(num_features=dim_decoder, fov_encoder=fov_encoder)

    def forward(self, images: torch.Tensor):
        _, _, _, W = images.shape

        encodings = self.encoder(images)
        features, features_0 = self.decoder(encodings)
        canonical_inverse_depth = self.head(features)

        fov_deg = self.fov.forward(images, features_0.detach())
        deg = fov_deg.to(torch.float)
        rad = 3.141592 * deg / 180.0
        f_px = 0.5 * W / torch.tan(0.5 * rad)
        inverse_depth = canonical_inverse_depth * (W / f_px)

        inverse_depth = inverse_depth.reshape(-1, 1536, 1536)
        f_px = f_px.reshape(-1, 1)

        return inverse_depth, f_px


@app.command()
def export_to_onnx(
    pt_file_path: Path = typer.Option(
        "depth_pro.pt", help="Path to the input pt model file"
    ),
    onnx_file_path: Path = typer.Option(
        "model_repository/depth_pro/1/model.onnx",
        help="Path to the output onnx model file",
    ),
    cuda: bool = typer.Option(False, help="Use cuda device for export"),
):
    onnx_file_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if cuda else "cpu")
    pt_file_path = Path(pt_file_path)

    # --------------------------------------------------------------------
    # Download model from hf
    local_dir = pt_file_path.parent
    local_dir.mkdir(exist_ok=True, parents=True)

    if not pt_file_path.exists():
        hf_hub_download(
            repo_id="apple/DepthPro",
            filename=str(pt_file_path),
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )

    # --------------------------------------------------------------------
    # Load model on device
    preset = "dinov2l16_384"
    patch_encoder_config = VIT_CONFIG_DICT[preset]
    patch_encoder = create_vit(preset)
    image_encoder = create_vit(preset)
    fov_encoder = create_vit(preset)

    dims_encoder = patch_encoder_config.encoder_feature_dims  # type: ignore
    hook_block_ids = patch_encoder_config.encoder_feature_layer_ids  # type: ignore
    encoder = DepthProEncoder(
        dims_encoder=dims_encoder,
        patch_encoder=patch_encoder,
        image_encoder=image_encoder,
        hook_block_ids=hook_block_ids,
        decoder_features=256,
    )
    decoder = MultiresConvDecoder(
        dims_encoder=[256] + list(encoder.dims_encoder),
        dim_decoder=256,
    )
    model = DepthPro(
        encoder=encoder,
        decoder=decoder,
        last_dims=(32, 1),
        fov_encoder=fov_encoder,
    )

    state_dict = torch.load(pt_file_path, map_location=device)
    model.load_state_dict(state_dict=state_dict, strict=True)

    model = model.to(device)
    model = model.eval()
    model = model.half()

    # --------------------------------------------------------------------
    # Export model
    images = torch.randn(1, 3, 1536, 1536, device=device, dtype=torch.float16)

    torch.onnx.export(
        model,
        (images,),
        onnx_file_path,
        input_names=["images"],
        output_names=["depth", "f_px"],
        opset_version=19,
        do_constant_folding=True,
    )


if __name__ == "__main__":
    app()

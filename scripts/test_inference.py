from pathlib import Path
from typing import get_args

from modelbox.core import MediaModelName, TritonInferer

_THID_DIR = Path(__file__).parent


def main():
    inferer = TritonInferer(url="127.0.0.1:8001")

    model_names = get_args(MediaModelName)
    file_names = ("spoty.png", "ring.mp4")

    for model_name in model_names:
        for file_name in file_names:
            input_file_path = _THID_DIR / "data" / file_name
            output_file_path = _THID_DIR / "data" / f"{model_name}_{file_name}"

            print(f"Inferencing model {model_name} on file {input_file_path}")
            inferer.infer_media_model(
                input_file_path=input_file_path,
                output_file_path=output_file_path,
                model_name=model_name,
            )
            print(f"File saved: {output_file_path}")


if __name__ == "__main__":
    main()

# ModelBox

A local image processing server that hosts machine learning models for background removal and depth estimation.

## Requirements

- NVIDIA GPU with CUDA support (up to 8GB VRAM required)
- Docker with NVIDIA Container Toolkit
- Windows: Docker Desktop with WSL2 backend
- macOS: CPU-only inference (limited performance)

## Models

- **Background Removal**: [BRIA RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4)
- **Depth Estimation**: [Apple DepthPro](https://huggingface.co/apple/DepthPro)

## Installation

1. Install uv package manager:
   
Linux/macOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

See [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for other installation methods.

2. Install dependencies:
```bash
uv sync
```

3. Setup and run services:
```bash
uv run python scripts/setup.py
```

This will download the required models and start all services using Docker Compose.

Run `uv run python scripts/test_inference.py` to test the inference pipeline.

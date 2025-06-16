FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    g++ \
    libmagic1 \
    ffmpeg \
    libgl1-mesa-dev \
    libegl1 \
    libopengl0 \
    libglx-mesa0 \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --no-dev

COPY modelbox ./modelbox/
COPY .env .

ENV PYTHONPATH=/app

CMD ["/app/.venv/bin/uvicorn", "modelbox.app:app", "--host", "0.0.0.0", "--port", "8228"]

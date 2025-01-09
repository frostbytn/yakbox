FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Pre-install libraries to reduce build time
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu117 \
    transformers fastapi[all] uvicorn \
    sentencepiece protobuf bitsandbytes scipy tokenizers accelerate datasets

# Copy code after installing dependencies
COPY . /app

EXPOSE 8000
CMD ["uvicorn", "yakbox:app", "--host", "0.0.0.0", "--port", "8000"]

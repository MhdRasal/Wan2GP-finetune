# ==============================================================================
# Wan2GP RunPod Serverless Container
# ==============================================================================
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /workspace

# Install system dependencies & multimedia tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone Wan2GP repository
RUN git clone https://github.com/MhdRasal/Wan2GP-finetune.git /workspace/Wan2GP

WORKDIR /workspace/Wan2GP

# Install dependencies (requirements + fast Hugging Face transfer + RunPod SDK)
RUN pip install --no-cache-dir \
    -r requirements.txt \
    runpod>=1.7.0 \
    hf-transfer

# Copy RunPod Serverless entrypoint
COPY rp_handler.py /workspace/Wan2GP/rp_handler.py

CMD ["python", "-u", "/workspace/Wan2GP/rp_handler.py"]

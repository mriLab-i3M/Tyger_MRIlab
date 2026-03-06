
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# Global environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

ENV APP_DIR=/app
ENV UV_PROJECT_ENVIRONMENT=${APP_DIR}/.venv
ENV PATH="${APP_DIR}/.venv/bin:${PATH}"
ENV UV_NO_SYNC=1

# 1. System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-venv \
    build-essential \
    git \
    git-lfs \
    curl \
    wget \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# 2. Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# 3. Copy SNRAware repository and install Python dependencies
WORKDIR ${APP_DIR}
COPY SNRAware/ ${APP_DIR}/SNRAware/

WORKDIR ${APP_DIR}/SNRAware
RUN uv sync --no-dev

# 4. Install additional requirements
COPY requirements.txt ${APP_DIR}/requirements.txt
RUN uv pip install --no-cache-dir -r ${APP_DIR}/requirements.txt

# 5. Copy reconstruction pipeline
COPY snraware_pipeline/ ${APP_DIR}/snraware_pipeline/

WORKDIR ${APP_DIR}

# 6. Verify PyTorch and CUDA availability at build time
RUN python -c "import torch; print('PyTorch:', torch.__version__); \
               print('CUDA available:', torch.cuda.is_available())"

CMD ["python", "/app/snraware_pipeline/recon_scripts/stream_recon_RARE.py", "-i", "-", "-o", "-"]
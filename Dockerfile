# Dockerfile
FROM python:3.11-slim-bookworm

# System dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./

# Install dependencies with exact versions
RUN uv sync --frozen

# Copy source code
COPY . .

CMD ["bash"]
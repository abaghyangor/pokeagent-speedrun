# Dockerfile for PokéAgent speedrunning agent environment
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"

# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    ca-certificates \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libgl1 \
    libglu1-mesa \
    libsdl2-2.0-0 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install mGBA system library required by the Python bindings
RUN wget https://github.com/mgba-emu/mgba/releases/download/0.10.5/mGBA-0.10.5-ubuntu64-focal.tar.xz \
    && tar -xf mGBA-0.10.5-ubuntu64-focal.tar.xz \
    && apt-get update \
    && apt-get install -y ./mGBA-0.10.5-ubuntu64-focal/libmgba.deb \
    && rm -rf /var/lib/apt/lists/* mGBA-0.10.5-ubuntu64-focal.tar.xz mGBA-0.10.5-ubuntu64-focal

WORKDIR /app

COPY pyproject.toml uv.lock requirements.txt* ./
COPY . .

# Create the virtual environment and install project dependencies
RUN uv sync --frozen

# Default command opens a shell; override with `docker run ... python run.py`
CMD ["bash"]

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libopencv-dev \
    python3-opencv \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import cv2; import torch; print('healthy')" || exit 1

# Run the CV service
CMD ["python", "main.py"]

# docker/Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    exiftool \
    libraw-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app
COPY requirements/ requirements/
RUN pip install -r requirements/base.txt

# Copy application
COPY . .
RUN pip install -e .

# Download models
RUN python scripts/download_models.py

EXPOSE 8000
CMD ["vintageoptics", "serve"]
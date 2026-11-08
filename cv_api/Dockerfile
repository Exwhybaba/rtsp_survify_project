# Single image, two entrypoints: the analytics loop (pipeline.py) and the
# dashboard API (dashboard/api.py). docker-compose.yml runs each service
# from this same image with a different command, which keeps the
# dependency set (and build) identical for both.
FROM python:3.11-slim

# libgl1/libglib2.0-0 are required by opencv-python-headless's video
# decoding backend even though we never open a display.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Default: run the analytics pipeline. docker-compose overrides the
# command for the dashboard service.
CMD ["python", "-m", "src.analytics.pipeline", "--config", "config/pipeline.yaml"]

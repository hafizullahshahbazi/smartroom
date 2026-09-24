FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

# System deps (no patchelf needed)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Use ctranslate2 >= 4.6.0 (fixed execstack issue)
# Added edge-tts for voice-to-voice translation
RUN pip install --no-cache-dir \
    "ctranslate2>=4.6.0" \
    "transformers==4.45.0" \
    sentencepiece \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    websockets \
    requests \
    huggingface_hub \
    numpy \
    edge-tts

WORKDIR /app
COPY main.py pipeline.py rooms.py ./
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
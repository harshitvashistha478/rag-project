FROM python:3.11-slim

# System deps — Tesseract for OCR, libGL for PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command — overridden per-service in docker-compose.yml
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
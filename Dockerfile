FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir kafka-python psycopg2-binary

# Copy source code
COPY . .

# Create data directories
RUN mkdir -p data model/weights

EXPOSE 8501

CMD ["python", "run_local.py", "--video", "data/input.mp4", "--fresh"]

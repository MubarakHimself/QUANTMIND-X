FROM node:20-slim

# Install System Dependencies
# python3/pip -> for Python packages
# ffmpeg -> for media processing
# git -> for npm dependencies
# curl -> for healthchecks
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for video processing and API
# Break-system-packages is needed on newer Debian/Ubuntu
RUN pip3 install --break-system-packages \
    pytubefix \
    opencv-python-headless \
    moviepy \
    pillow \
    uvicorn \
    fastapi \
    httpx \
    sse-starlette \
    duckdb \
    prometheus_client \
    pandas \
    numpy \
    backtrader \
    pytz \
    aiohttp \
    websockets \
    python-multipart \
    pydantic-settings \
    pyyaml \
    requests \
    aiofiles \
    schedule \
    python-dotenv \
    sqlalchemy \
    pydantic \
    redis \
    apscheduler \
    psutil \
    pyarrow \
    python-snappy \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-instrumentation-fastapi \
    opentelemetry-instrumentation-sqlalchemy \
    opentelemetry-instrumentation-psycopg2 \
    opentelemetry-exporter-otlp \
    opentelemetry-exporter-otlp-proto-grpc

# Install Gemini CLI and Qwen Code CLI
RUN npm install -g @google/gemini-cli @qwen-code/qwen-code

# Setup Application Directory
WORKDIR /app

# Copy Source Code
COPY server/package.json ./server/
COPY extensions/video-ingest-extension/package.json ./extensions/video-ingest-extension/

# Install Node Dependencies
WORKDIR /app/server
RUN npm install

# Copy main source
WORKDIR /app
COPY . .

# Set Working Directory
WORKDIR /app/server

# Set Python path to find src module
ENV PYTHONPATH=/app

# Expose API Port
EXPOSE 8000 9090

# Default Command
CMD ["python3", "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]

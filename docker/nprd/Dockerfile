# NPRD Docker - Gemini CLI with YOLO Mode
# Primary: Gemini CLI (headless + yolo for bypass permissions)
# Backup: Qwen API (cloud, not local models)

FROM node:20-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Gemini CLI
RUN npm install -g @anthropic-ai/gemini-cli

WORKDIR /app

# Create directories
RUN mkdir -p /root/.gemini /app/data/nprd /app/data/videos

# Copy NPRD code
COPY src/nprd /app/src/nprd
COPY requirements.txt /app/

# Install Python deps
RUN pip3 install --break-system-packages -r requirements.txt

# Volume for OAuth credentials (set up once)
VOLUME ["/root/.gemini", "/app/data"]

# Environment - YOLO mode enabled by default
ENV PYTHONPATH=/app
ENV NPRD_DATA_DIR=/app/data/nprd
ENV GEMINI_YOLO_MODE=true

EXPOSE 8001

COPY docker/nprd/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-m", "src.nprd.api", "--port", "8001"]

FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install httpx for PageIndex HTTP calls
RUN pip install --no-cache-dir httpx

# Copy scripts
COPY scripts/ /app/scripts/

# Copy data directories
COPY data/scraped_articles/ /app/data/scraped_articles/
COPY data/knowledge_base/ /app/data/knowledge_base/
COPY data/logs/ /app/data/logs/

# Default command - index all collections to PageIndex
CMD ["python", "scripts/index_to_pageindex.py", "--check-health", "--all"]
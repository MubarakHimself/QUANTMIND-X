FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt sentence-transformers

# Copy scripts
COPY scripts/ /app/scripts/
COPY data/scraped_articles/ /app/data/scraped_articles/
COPY data/knowledge_index/ /app/data/knowledge_index/

# Install Qdrant client and additional deps
RUN pip install --no-cache-dir qdrant-client sentence-transformers tqdm

# Default command
CMD ["python", "scripts/index_to_qdrant.py", "--input", "data/scraped_articles/"]

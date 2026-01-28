#!/usr/bin/env python3
"""
Knowledge Base Indexer using OpenAI Embeddings
No PyTorch/sentence-transformers required!
"""

import os
import sys
from pathlib import Path
import hashlib
import uuid

# Check dependencies
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    print("❌ qdrant-client not found. Run: pip install qdrant-client")
    sys.exit(1)

try:
    import openai
except ImportError:
    print("❌ openai not found. Run: pip install openai")
    sys.exit(1)

# Configuration
SCRAPED_DIR = Path("data/scraped_articles")
QDRANT_PATH = Path("data/qdrant_db")
COLLECTION_NAME = "mql5_knowledge"
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI's model
EMBEDDING_DIM = 1536

def get_embedding(text: str) -> list:
    """Get embedding from OpenAI API."""
    client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

def extract_frontmatter(content: str) -> tuple:
    """Extract YAML frontmatter from markdown."""
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            frontmatter = parts[1].strip()
            meta = {}
            for line in frontmatter.split('\n'):
                if ':' in line:
                    key, val = line.split(':', 1)
                    meta[key.strip()] = val.strip()
            return meta, parts[2].strip()
    return {}, content

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks

def generate_id(text: str) -> uuid.UUID:
    """Generate unique ID from text."""
    hash_hex = hashlib.md5(text.encode()).hexdigest()
    return uuid.UUID(hex=hash_hex[:32])

def main():
    print("🔍 Knowledge Base Indexer (OpenAI)")
    print("=" * 60)

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        print("   Add to .env file or export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    # Check if scraped articles exist
    if not SCRAPED_DIR.exists():
        print(f"❌ No scraped articles found at: {SCRAPED_DIR}")
        sys.exit(1)

    # Initialize Qdrant
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))

    # Create collection
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        print(f"\n📂 Creating collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )

    # Process articles
    md_files = list(SCRAPED_DIR.rglob("*.md"))
    print(f"\n📚 Processing {len(md_files)} articles...")

    all_points = []
    batch_embeddings = []

    for i, file_path in enumerate(md_files, 1):
        if i % 10 == 0:
            print(f"   Processing {i}/{len(md_files)}...")

        try:
            content = file_path.read_text(encoding='utf-8')
            meta, body = extract_frontmatter(content)
            chunks = chunk_text(body, chunk_size=1000, overlap=200)

            for j, chunk in enumerate(chunks):
                # Get embedding from OpenAI
                embedding = get_embedding(chunk)

                point_id = generate_id(f"{file_path.stem}_{j}")
                all_points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "title": meta.get('title', file_path.stem),
                        "url": meta.get('url', ''),
                        "categories": meta.get('categories', ''),
                        "file_path": str(file_path.relative_to(SCRAPED_DIR)),
                        "text": chunk[:500]
                    }
                ))

        except Exception as e:
            print(f"   ⚠️  Error: {e}")
            continue

    # Batch upsert
    print(f"\n💾 Upserting {len(all_points)} vectors...")
    batch_size = 50
    for i in range(0, len(all_points), batch_size):
        batch = all_points[i:i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)

    print(f"\n✅ Done! {len(all_points)} vectors indexed")
    print(f"   Cost: ~${len(all_points) * 0.00002 / 1000:.2f} (OpenAI API)")

if __name__ == "__main__":
    main()

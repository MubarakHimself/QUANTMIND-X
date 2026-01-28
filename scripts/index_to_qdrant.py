#!/usr/bin/env python3
"""
Qdrant Knowledge Base Indexer
Indexes all scraped articles into Qdrant vector database for RAG queries.
Uses sentence-transformers for embeddings.

Run setup.sh first to install dependencies.
"""

import json
import sys
from pathlib import Path
import hashlib
import uuid

# Check dependencies
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    print("❌ qdrant-client not found.")
    print("   Run: ./setup.sh")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ sentence-transformers not found.")
    print("   Run: ./setup.sh")
    sys.exit(1)

# Configuration
SCRAPED_DIR = Path("data/scraped_articles")
QDRANT_PATH = Path("data/qdrant_db")
COLLECTION_NAME = "mql5_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

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

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """Split text into overlapping chunks for better retrieval."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk) > 50:
            chunks.append(chunk)

    return chunks

def generate_id(text: str) -> uuid.UUID:
    """Generate unique ID from text (proper UUID format)."""
    hash_hex = hashlib.md5(text.encode()).hexdigest()
    return uuid.UUID(hex=hash_hex[:32])

def main():
    print("🔍 Qdrant Knowledge Base Indexer")
    print("=" * 60)

    # Check if scraped articles exist
    if not SCRAPED_DIR.exists():
        print(f"❌ No scraped articles found at: {SCRAPED_DIR}")
        print("   Run the scraper first to collect articles.")
        sys.exit(1)

    # Initialize Qdrant
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))

    # Initialize embedding model
    print("📦 Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"   Model: {EMBEDDING_MODEL} (dim={embedding_dim})")

    # Create collection if not exists
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        print(f"\n📂 Creating collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
        )
    else:
        print(f"\n📂 Collection exists: {COLLECTION_NAME}")

    # Process articles
    md_files = list(SCRAPED_DIR.rglob("*.md"))
    print(f"\n📚 Processing {len(md_files)} articles...")

    if len(md_files) == 0:
        print("❌ No .md files found in scraped articles directory.")
        sys.exit(1)

    all_points = []

    for i, file_path in enumerate(md_files, 1):
        if i % 100 == 0:
            print(f"   Embedding {i}/{len(md_files)}...")

        try:
            content = file_path.read_text(encoding='utf-8')
            meta, body = extract_frontmatter(content)

            # Create chunks
            chunks = chunk_text(body, chunk_size=300, overlap=50)

            for j, chunk in enumerate(chunks):
                # Generate embedding
                embedding = model.encode(chunk).tolist()

                # Create point
                point_id = generate_id(f"{file_path.stem}_{j}")

                all_points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "title": meta.get('title', file_path.stem),
                        "url": meta.get('url', ''),
                        "categories": meta.get('categories', ''),
                        "relevance_score": int(meta.get('relevance_score', 0)),
                        "file_path": str(file_path.relative_to(SCRAPED_DIR)),
                        "chunk_index": j,
                        "text": chunk[:1000]
                    }
                ))

        except Exception as e:
            print(f"   ⚠️  Error processing {file_path.name}: {e}")
            continue

    if not all_points:
        print("❌ No vectors were generated.")
        sys.exit(1)

    # Batch upsert
    print(f"\n💾 Upserting {len(all_points)} vectors to Qdrant...")
    batch_size = 100
    for i in range(0, len(all_points), batch_size):
        batch = all_points[i:i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        if (i // batch_size) % 10 == 0:
            print(f"   Uploaded {min(i + batch_size, len(all_points))}/{len(all_points)}")

    print(f"\n✅ Indexing complete!")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Total vectors: {len(all_points)}")
    print(f"   Storage: {QDRANT_PATH}")

    # Test query
    print("\n🔍 Testing search...")
    test_query = "RSI divergence trading strategy"
    query_vector = model.encode(test_query).tolist()

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=3
    )

    print(f"   Query: '{test_query}'")
    print(f"   Top results:")
    for r in results:
        print(f"   - {r.payload['title'][:50]}... (score: {r.score:.3f})")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Knowledge Base Indexer using ChromaDB
Lightweight vector database, no heavy ML libraries!
"""

import os
import sys
from pathlib import Path

# Check dependencies
try:
    import chromadb
except ImportError:
    print("❌ chromadb not found. Run: pip install chromadb")
    sys.exit(1)

# Configuration
SCRAPED_DIR = Path("data/scraped_articles")
COLLECTION_NAME = "mql5_knowledge"
CHROMA_PATH = Path("data/chromadb")

def main():
    print("🔍 Knowledge Base Indexer (ChromaDB)")
    print("=" * 60)

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Process articles
    md_files = list(SCRAPED_DIR.rglob("*.md"))
    print(f"\n📚 Processing {len(md_files)} articles...")

    for i, file_path in enumerate(md_files, 1):
        if i % 50 == 0:
            print(f"   Processing {i}/{len(md_files)}...")

        try:
            content = file_path.read_text(encoding='utf-8')
            # Extract title from first line or filename
            lines = content.split('\n')
            title = lines[0].replace('#', '').strip() if lines else file_path.stem

            # Add to collection
            collection.add(
                documents=[content],
                metadatas=[{
                    "title": title,
                    "file_path": str(file_path.relative_to(SCRAPED_DIR)),
                    "categories": "Trading"
                }],
                ids=[file_path.stem]
            )

        except Exception as e:
            print(f"   ⚠️  Error: {e}")
            continue

    print(f"\n✅ Done! Indexed {len(md_files)} articles")
    print(f"   Storage: {CHROMA_PATH}")

    # Test search
    print("\n🔍 Testing search...")
    results = collection.query(
        query_texts=["RSI divergence trading"],
        n_results=3
    )

    print(f"   Top results for 'RSI divergence':")
    for i, doc in enumerate(results['documents'][0], 1):
        print(f"   {i}. {doc[:100]}...")

if __name__ == "__main__":
    main()

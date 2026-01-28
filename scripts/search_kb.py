#!/usr/bin/env python3
"""
Quick KB search for Claude Code
Usage: python scripts/search_kb.py "your search query"
"""

import sys
import json
import chromadb
from pathlib import Path

def search_kb(query: str, n_results: int = 5):
    """Search the MQL5 knowledge base."""
    client = chromadb.PersistentClient(path=str(Path("data/chromadb")))
    collection = client.get_collection("mql5_knowledge")

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    formatted = []
    if results['documents'] and len(results['documents'][0]) > 0:
        for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
            # Extract title from frontmatter
            title = "Untitled"
            if doc.startswith('---'):
                lines = doc.split('\n')
                for line in lines[1:10]:
                    if line.startswith('title:'):
                        title = line.split(':', 1)[1].strip()
                        break
                    if line == '---':
                        break

            # Get preview
            preview = doc[:300] + "..." if len(doc) > 300 else doc

            formatted.append({
                "rank": i,
                "title": title,
                "file_path": meta.get('file_path', ''),
                "categories": meta.get('categories', ''),
                "preview": preview
            })

    return formatted

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "trading strategies"
    results = search_kb(query)

    print(f"📚 KB Search: \"{query}\"")
    print("=" * 80)

    for r in results:
        print(f"\n[{r['rank']}] {r['title']}")
        print(f"    File: {r['file_path']}")
        print(f"    Categories: {r['categories']}")
        print(f"    Preview: {r['preview'][:200]}...")

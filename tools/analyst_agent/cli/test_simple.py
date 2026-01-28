#!/usr/bin/env python3
"""
Simple test script for Analyst Agent CLI.
Run this to verify KB connection and see available files.
"""

import sys
from pathlib import Path

# Add to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "tools" / "analyst_agent"))

from kb.client import ChromaKBClient

# Test KB connection
print("=" * 60)
print("QuantMindX Analyst Agent - Test Kit")
print("=" * 60)

kb = ChromaKBClient()
print(f"\nChromaDB connected: {kb.db_path}")

collections = kb.list_collections()
print(f"Collections: {collections}")

for col in collections:
    stats = kb.get_collection_stats(col)
    print(f"  - {col}: {stats['count']} documents")

# Test searches
print("\n" + "-" * 60)
print("Knowledge Base Search Test")
print("-" * 60)

test_queries = [
    ("ORB strategy", 3),
    ("Kelly criterion", 2),
    ("position sizing", 2),
    ("risk management", 2),
]

for query, n in test_queries:
    results = kb.search(query, collection="mql5_knowledge", n=n)
    print(f'\nQuery: "{query}"')
    for r in results:
        print(f"  - {r['title'][:45]}... (score: {r['score']:.2f})")

# Show NPRD files
print("\n" + "-" * 60)
print("NPRD Files")
print("-" * 60)

import json
nprd_dir = project_root / "outputs" / "videos"
nprd_files = list(nprd_dir.rglob("*.json"))

if nprd_files:
    nprd_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    print(f"\nFound {len(nprd_files)} NPRD files:")
    for i, f in enumerate(nprd_files[:10], 1):
        size_kb = f.stat().st_size / 1024
        print(f"  [{i}] {f.name[:45]} ({size_kb:.1f} KB)")
else:
    print("No NPRD files found")

# Show commands
print("\n" + "-" * 60)
print("Available Commands")
print("-" * 60)
print("""
  python3 tools/analyst_agent/cli/main.py interactive
      - Launch interactive test environment (requires terminal)

  python3 tools/analyst_agent/cli/main.py list --nprd
      - List NPRD files

  python3 tools/analyst_agent/cli/main.py stats
      - Show KB statistics

  export OPENROUTER_API_KEY="sk-..."
      - Set API key for TRD generation
""")

print("=" * 60)
print("Test complete!")
print("=" * 60)

#!/usr/bin/env python3
"""
Create filtered ChromaDB collection for Analyst Agent.

Filters mql5_knowledge collection to create analyst_kb with ~330 articles
relevant to trading strategy analysis.

Usage:
    python scripts/filter_analyst_kb.py
"""

import chromadb
from pathlib import Path
import sys

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_COLLECTION = "mql5_knowledge"
ANALYST_COLLECTION = "analyst_kb"
CHROMA_PATH = PROJECT_ROOT / "data" / "chromadb"

# Filter criteria
ACCEPTED_CATEGORIES = [
    "Trading Systems",
    "Trading",
    "Expert Advisors",
    "Indicators"
]

REJECTED_COMBINATIONS = [
    "Machine Learning",
    "Integration"
]


def should_include(article_meta: dict) -> bool:
    """Check if article should be in analyst KB.

    Args:
        article_meta: Article metadata dictionary

    Returns:
        True if article should be included
    """
    categories = article_meta.get('categories', '')

    # Reject if has Machine Learning or Integration
    for rejected in REJECTED_COMBINATIONS:
        if rejected in categories:
            return False

    # Accept if matches target categories
    for accepted in ACCEPTED_CATEGORIES:
        if accepted in categories:
            return True

    return False


def create_analyst_kb():
    """Create filtered analyst_kb collection from mql5_knowledge."""

    print("=" * 60)
    print("Analyst KB Filter Script")
    print("=" * 60)

    # Initialize ChromaDB client
    print(f"\nConnecting to ChromaDB at: {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Get main collection
    print(f"Accessing main collection: {MAIN_COLLECTION}")
    try:
        main_col = client.get_collection(MAIN_COLLECTION)
        main_count = main_col.count()
        print(f"Main collection has {main_count} articles")
    except Exception as e:
        print(f"Error accessing main collection: {e}")
        sys.exit(1)

    # Check if analyst_kb exists
    existing_collections = [col.name for col in client.list_collections()]
    if ANALYST_COLLECTION in existing_collections:
        print(f"\nCollection '{ANALYST_COLLECTION}' already exists.")
        response = input("Delete and recreate? (y/N): ").strip().lower()
        if response == 'y':
            client.delete_collection(ANALYST_COLLECTION)
            print(f"Deleted existing collection.")
        else:
            print("Aborted.")
            sys.exit(0)

    # Create analyst collection
    print(f"\nCreating collection: {ANALYST_COLLECTION}")
    analyst_col = client.get_or_create_collection(
        name=ANALYST_COLLECTION,
        metadata={
            "hnsw:space": "cosine",
            "purpose": "analyst_agent",
            "source": MAIN_COLLECTION,
            "filter_criteria": str(ACCEPTED_CATEGORIES)
        }
    )

    # Get all articles from main collection
    print("\nFetching all articles from main collection...")
    results = main_col.get(include=['metadatas', 'documents'])

    print(f"Total articles fetched: {len(results['ids'])}")

    # Filter articles
    print("\nApplying filter criteria:")
    for criterion in ACCEPTED_CATEGORIES:
        print(f"  + {criterion}")
    for rejected in REJECTED_COMBINATIONS:
        print(f"  - {rejected}")

    filtered_ids = []
    filtered_docs = []
    filtered_metas = []

    for article_id, meta, doc in zip(results['ids'], results['metadatas'], results['documents']):
        if should_include(meta):
            # Use original ID to avoid duplicates
            filtered_ids.append(article_id)
            filtered_docs.append(doc)
            filtered_metas.append(meta)

    # Add to collection
    if filtered_ids:
        print(f"\nAdding {len(filtered_ids)} articles to {ANALYST_COLLECTION}...")

        # Batch add to avoid memory issues
        batch_size = 100
        for i in range(0, len(filtered_ids), batch_size):
            batch_ids = filtered_ids[i:i+batch_size]
            batch_docs = filtered_docs[i:i+batch_size]
            batch_metas = filtered_metas[i:i+batch_size]

            analyst_col.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas
            )
            print(f"  Added batch {i//batch_size + 1}/{(len(filtered_ids)-1)//batch_size + 1}")

        final_count = analyst_col.count()
    else:
        print("\nNo articles matched the filter criteria!")
        final_count = 0

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Source collection: {MAIN_COLLECTION}")
    print(f"  Total articles: {main_count}")
    print(f"\nFiltered collection: {ANALYST_COLLECTION}")
    print(f"  Total articles: {final_count}")
    print(f"  Filter ratio: {final_count/main_count*100:.1f}%")
    print(f"\nStorage path: {CHROMA_PATH}")
    print("=" * 60)

    # Test query
    print("\nTesting search with sample query...")
    test_results = analyst_col.query(
        query_texts=["stop loss strategy"],
        n_results=3
    )
    print(f"Found {len(test_results['documents'][0])} results for 'stop loss strategy'")

    print("\nDone!")


if __name__ == "__main__":
    try:
        create_analyst_kb()
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
Migration script: Qdrant to ChromaDB data conversion.

Migrates existing mql5_knowledge collection data from Qdrant to ChromaDB.
Preserves metadata: title, categories, file_path, relevance_score.

Usage:
    python scripts/migrate_qdrant_to_chroma.py [--dry-run]
"""

import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_qdrant_available() -> bool:
    """Check if Qdrant client is available."""
    try:
        from qdrant_client import QdrantClient
        return True
    except ImportError:
        return False


def check_chromadb_available() -> bool:
    """Check if ChromaDB is available."""
    try:
        import chromadb
        return True
    except ImportError:
        return False


def init_qdrant_client(dry_run: bool = False):
    """Initialize Qdrant client."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    qdrant_path = PROJECT_ROOT / "data" / "qdrant_db"

    if not qdrant_path.exists() and not dry_run:
        print(f"Warning: Qdrant DB path not found: {qdrant_path}")
        return None

    try:
        # Try to connect to local Qdrant instance
        client = QdrantClient(path=str(qdrant_path))
        return client
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        return None


def init_chromadb_client() -> Optional[Any]:
    """Initialize ChromaDB client."""
    import chromadb

    chroma_path = PROJECT_ROOT / "data" / "chromadb"
    chroma_path.mkdir(parents=True, exist_ok=True)

    HNSW_CONFIG = {
        "hnsw:space": "cosine",
        "hnsw:M": 16,
        "hnsw:construction_ef": 100
    }

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(
        name="mql5_knowledge",
        metadata=HNSW_CONFIG
    )

    return client, collection


def fetch_qdrant_data(qdrant_client, collection_name: str = "mql5_knowledge") -> Optional[Dict[str, Any]]:
    """Fetch all data from Qdrant collection."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    try:
        # Check if collection exists
        collections = qdrant_client.get_collections()
        collection_names = [coll.name for coll in collections.collections]

        if collection_name not in collection_names:
            print(f"Collection '{collection_name}' not found in Qdrant")
            print(f"Available collections: {collection_names}")
            return None

        # Get collection info
        collection_info = qdrant_client.get_collection(collection_name)
        total_points = collection_info.points_count
        print(f"Found {total_points} points in Qdrant collection '{collection_name}'")

        # Fetch all points using offset batching
        BATCH_SIZE = 100
        all_points = []
        offset = 0

        while offset < total_points:
            records = qdrant_client.scroll(
                collection_name=collection_name,
                limit=BATCH_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )[0]

            if not records:
                break

            all_points.extend(records)
            offset += len(records)
            print(f"Fetched {len(all_points)}/{total_points} points...")

        return {
            "points": all_points,
            "total": total_points
        }

    except Exception as e:
        print(f"Error fetching data from Qdrant: {e}")
        return None


def transform_qdrant_to_chroma(qdrant_points: List[Any]) -> Dict[str, Any]:
    """Transform Qdrant data format to ChromaDB format."""
    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for point in qdrant_points:
        # Extract ID
        ids.append(str(point.id))

        # Extract payload as document
        payload = point.payload or {}
        content = payload.get("content", payload.get("text", ""))
        documents.append(content)

        # Extract metadata
        metadata = {
            "title": payload.get("title", ""),
            "categories": payload.get("categories", ""),
            "file_path": payload.get("file_path", ""),
            "relevance_score": payload.get("relevance_score", 0.0)
        }
        metadatas.append(metadata)

        # Extract vector (Qdrant uses .vector, ChromaDB will re-embed if not provided)
        if hasattr(point, 'vector') and point.vector is not None:
            embeddings.append(point.vector)
        else:
            embeddings.append(None)

    return {
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
        "embeddings": embeddings
    }


def migrate_to_chromadb(
    chroma_collection,
    chroma_data: Dict[str, Any],
    dry_run: bool = False
) -> bool:
    """Migrate data to ChromaDB collection."""
    try:
        if dry_run:
            print(f"\n[DRY RUN] Would migrate {len(chroma_data['ids'])} documents to ChromaDB")
            print(f"Sample metadata: {chroma_data['metadatas'][0] if chroma_data['metadatas'] else 'None'}")
            return True

        # Check for existing data
        existing_count = chroma_collection.count()
        if existing_count > 0:
            print(f"Warning: ChromaDB collection already has {existing_count} documents")
            response = input("Continue with migration? (y/N): ")
            if response.lower() != 'y':
                print("Migration cancelled")
                return False

        # Add data to ChromaDB
        # Note: We'll let ChromaDB handle embeddings automatically if not provided
        print(f"\nMigrating {len(chroma_data['ids'])} documents to ChromaDB...")

        chroma_collection.add(
            ids=chroma_data["ids"],
            documents=chroma_data["documents"],
            metadatas=chroma_data["metadatas"]
            # embeddings omitted - ChromaDB will auto-generate
        )

        print(f"Successfully migrated {len(chroma_data['ids'])} documents")
        return True

    except Exception as e:
        print(f"Error migrating to ChromaDB: {e}")
        return False


def verify_migration(chroma_collection, expected_count: int) -> bool:
    """Verify that migration was successful."""
    try:
        actual_count = chroma_collection.count()
        print(f"\nMigration verification:")
        print(f"  Expected documents: {expected_count}")
        print(f"  Actual documents: {actual_count}")

        if actual_count >= expected_count:
            print("  Status: SUCCESS")
            return True
        else:
            print(f"  Status: PARTIAL ({expected_count - actual_count} documents missing)")
            return False

    except Exception as e:
        print(f"Error during verification: {e}")
        return False


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description="Migrate Qdrant data to ChromaDB")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without making changes")
    parser.add_argument("--skip-verification", action="store_true", help="Skip post-migration verification")
    args = parser.parse_args()

    print("=" * 60)
    print("Qdrant to ChromaDB Migration Script")
    print("=" * 60)

    # Check dependencies
    if not check_qdrant_available():
        print("\nError: Qdrant client not installed")
        print("Install with: pip install qdrant-client")
        sys.exit(1)

    if not check_chromadb_available():
        print("\nError: ChromaDB not installed")
        print("Install with: pip install chromadb")
        sys.exit(1)

    # Initialize clients
    print("\n1. Initializing clients...")
    qdrant_client = init_qdrant_client(args.dry_run)
    if not qdrant_client and not args.dry_run:
        print("Warning: Could not connect to Qdrant. Attempting to continue...")

    chroma_client, chroma_collection = init_chromadb_client()
    print(f"   ChromaDB: {chroma_collection}")

    # Fetch Qdrant data
    print("\n2. Fetching data from Qdrant...")
    if qdrant_client:
        qdrant_data = fetch_qdrant_data(qdrant_client)
        if not qdrant_data:
            print("No data found in Qdrant. Exiting.")
            sys.exit(0)

        qdrant_points = qdrant_data["points"]
        print(f"   Fetched {len(qdrant_points)} points")
    else:
        print("   Skipped (Qdrant client not available)")
        qdrant_points = []

    # Transform data
    print("\n3. Transforming data format...")
    if qdrant_points:
        chroma_data = transform_qdrant_to_chroma(qdrant_points)
        print(f"   Transformed {len(chroma_data['ids'])} documents")
    else:
        chroma_data = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}

    # Migrate to ChromaDB
    print("\n4. Migrating to ChromaDB...")
    success = migrate_to_chromadb(chroma_collection, chroma_data, args.dry_run)

    if success and not args.dry_run and not args.skip_verification:
        verify_migration(chroma_collection, len(qdrant_points) if qdrant_points else 0)

    print("\n" + "=" * 60)
    if args.dry_run:
        print("Dry run completed successfully")
    elif success:
        print("Migration completed successfully")
    else:
        print("Migration failed")
    print("=" * 60)


if __name__ == "__main__":
    main()

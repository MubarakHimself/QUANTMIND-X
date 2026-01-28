"""
ChromaDB Knowledge Base Client for QuantMindX Analyst Agent

Provides semantic search over trading knowledge bases with support for
multiple collections and category filtering.

Collections:
- mql5_knowledge: Full MQL5 article database
- analyst_kb: Filtered collection for trading strategies, indicators, EAs

Requirements:
    pip install chromadb
"""

import logging
from pathlib import Path
from typing import Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases for clarity
SearchResult = dict[str, Any]
CollectionStats = dict[str, Any]


class ChromaKBError(Exception):
    """Base exception for ChromaKB client errors."""
    pass


class CollectionNotFoundError(ChromaKBError):
    """Raised when a requested collection does not exist."""
    pass


class InvalidQueryError(ChromaKBError):
    """Raised when a query is invalid."""
    pass


class ChromaKBClient:
    """
    Client for ChromaDB knowledge base operations.

    Supports multiple collections with cosine similarity search (HNSW).
    Provides methods for semantic search, statistics, and collection management.

    Attributes:
        db_path: Path to ChromaDB storage directory
        _client: ChromaDB persistent client instance

    Example:
        >>> kb = ChromaKBClient()
        >>> results = kb.search("RSI divergence strategy", collection="analyst_kb", n=5)
        >>> for r in results:
        ...     print(f"{r['title']}: {r['score']:.2f}")
    """

    # Valid categories for analyst_kb collection
    ANALYST_CATEGORIES = {
        "Trading Systems",
        "Trading",
        "Expert Advisors",
        "Indicators",
    }

    # Categories to exclude from analyst_kb
    EXCLUDED_CATEGORIES = {
        "Machine Learning",
        "Integration",
    }

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        """
        Initialize ChromaKB client.

        Args:
            db_path: Path to ChromaDB storage. Defaults to data/chromadb.

        Raises:
            ChromaKBError: If ChromaDB is not installed or initialization fails.
        """
        try:
            import chromadb
        except ImportError as e:
            raise ChromaKBError(
                "ChromaDB not installed. Run: pip install chromadb"
            ) from e

        if db_path is None:
            # Default path relative to project root
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            db_path = project_root / "data" / "chromadb"
        elif isinstance(db_path, str):
            db_path = Path(db_path)

        self.db_path: Path = db_path
        self._client = chromadb.PersistentClient(path=str(self.db_path))

        logger.info(f"ChromaKB client initialized: {self.db_path}")

    def search(
        self,
        query: str,
        collection: str = "analyst_kb",
        n: int = 5,
        category_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Search knowledge base using semantic similarity.

        Args:
            query: Natural language search query
            collection: Collection name (default: analyst_kb)
            n: Maximum number of results (default: 5)
            category_filter: Optional category filter (e.g., "Trading Systems")

        Returns:
            List of search results with keys:
                - title: Article title
                - file_path: Relative path to article file
                - categories: Article categories
                - score: Relevance score (0-1, higher is better)
                - preview: Content preview (first ~300 chars)

        Raises:
            CollectionNotFoundError: If collection does not exist
            InvalidQueryError: If query is empty or invalid

        Example:
            >>> results = kb.search("moving average crossover", n=3)
            >>> print(results[0]['title'])
            'Moving Average Trading Strategy'
        """
        if not query or not isinstance(query, str):
            raise InvalidQueryError("Query must be a non-empty string")

        if n <= 0:
            raise InvalidQueryError("n must be greater than 0")

        try:
            chroma_collection = self._client.get_collection(name=collection)
        except Exception as e:
            raise CollectionNotFoundError(
                f"Collection '{collection}' not found"
            ) from e

        # Query with HNSW cosine similarity
        # Fetch more results for filtering
        query_results = chroma_collection.query(
            query_texts=[query],
            n_results=n * 2 if category_filter else n,
        )

        results: list[SearchResult] = []
        seen_titles: set[str] = set()

        documents = query_results.get("documents", [[]])
        metadatas = query_results.get("metadatas", [[]])
        distances = query_results.get("distances", [[]])

        if not documents or not documents[0]:
            return results

        for i, doc in enumerate(documents[0]):
            metadata = metadatas[0][i] if metadatas and i < len(metadatas[0]) else {}
            distance = distances[0][i] if distances and i < len(distances[0]) else 0.0

            # Extract title from document
            title = self._extract_title(doc)

            # Deduplicate by title
            if title in seen_titles:
                continue

            # Apply category filter if specified
            if category_filter:
                categories = str(metadata.get("categories", "") or "")
                if category_filter.lower() not in categories.lower():
                    continue

            seen_titles.add(title)

            # Convert cosine distance to similarity score
            score = 1.0 - distance

            # Extract preview (skip YAML frontmatter)
            preview = self._extract_preview(doc)

            results.append({
                "title": title,
                "file_path": metadata.get("file_path", ""),
                "categories": metadata.get("categories", ""),
                "score": round(score, 3),
                "preview": preview,
            })

            # Return only requested number of results
            if len(results) >= n:
                break

        logger.info(f"Search '{query}' in '{collection}': {len(results)} results")
        return results

    def get_collection_stats(self, collection: str) -> CollectionStats:
        """
        Get statistics for a specific collection.

        Args:
            collection: Collection name

        Returns:
            Dictionary with keys:
                - name: Collection name
                - count: Number of documents
                - metadata: Collection metadata (includes HNSW config)

        Raises:
            CollectionNotFoundError: If collection does not exist

        Example:
            >>> stats = kb.get_collection_stats("analyst_kb")
            >>> print(stats['count'])
            1250
        """
        try:
            chroma_collection = self._client.get_collection(name=collection)
        except Exception as e:
            raise CollectionNotFoundError(
                f"Collection '{collection}' not found"
            ) from e

        count = chroma_collection.count()
        metadata = chroma_collection.metadata

        return {
            "name": collection,
            "count": count,
            "metadata": metadata or {},
        }

    def list_collections(self) -> list[str]:
        """
        List all available collections.

        Returns:
            List of collection names

        Example:
            >>> collections = kb.list_collections()
            >>> print(collections)
            ['mql5_knowledge', 'analyst_kb']
        """
        collections = self._client.list_collections()
        return [c.name for c in collections]

    def create_analyst_kb(
        self,
        source_collection: str = "mql5_knowledge",
        target_collection: str = "analyst_kb",
    ) -> dict[str, int]:
        """
        Create filtered analyst_kb collection from source collection.

        Filters mql5_knowledge by allowed categories:
        - Trading Systems, Trading, Expert Advisors, Indicators

        Excludes:
        - Machine Learning, Integration

        Args:
            source_collection: Source collection name (default: mql5_knowledge)
            target_collection: Target collection name (default: analyst_kb)

        Returns:
            Dictionary with stats:
                - total: Total documents in source
                - included: Documents included in analyst_kb
                - excluded: Documents excluded

        Raises:
            CollectionNotFoundError: If source collection does not exist

        Example:
            >>> stats = kb.create_analyst_kb()
            >>> print(f"Included: {stats['included']}, Excluded: {stats['excluded']}")
        """
        try:
            source = self._client.get_collection(name=source_collection)
        except Exception as e:
            raise CollectionNotFoundError(
                f"Source collection '{source_collection}' not found"
            ) from e

        # Get all documents from source
        all_data = source.get(include=["documents", "metadatas", "embeddings"])

        # Create or get target collection
        target = self._client.get_or_create_collection(
            name=target_collection,
            metadata={"hnsw:space": "cosine"},
        )

        # Clear existing data
        try:
            target.delete(where={})
        except Exception:
            pass  # Collection might be empty

        included_docs: list[str] = []
        included_metadatas: list[dict[str, Any]] = []
        included_ids: list[str] = []
        included_embeddings: list[Any] = []

        excluded_count = 0

        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])
        embeddings = all_data.get("embeddings", [])
        ids = [f"doc_{i}" for i in range(len(documents))]

        # Check if we have embeddings (handle numpy arrays)
        has_embeddings = embeddings is not None and len(embeddings) > 0

        for i, doc in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            # Get categories and ensure it's a string (handles numpy arrays, lists, etc.)
            categories_val = metadata.get("categories", "")
            if categories_val is None:
                categories = ""
            elif isinstance(categories_val, (list, tuple)):
                # Handle list/tuple of categories
                categories = ", ".join(str(c) for c in categories_val)
            else:
                # Convert to string, handling numpy arrays and other types
                try:
                    categories = str(categories_val)
                except Exception:
                    categories = ""

            # Check if document should be included
            if self._should_include_for_analyst(categories):
                included_docs.append(doc)
                included_metadatas.append(metadata)
                included_ids.append(ids[i])
                if has_embeddings and i < len(embeddings):
                    included_embeddings.append(embeddings[i])
            else:
                excluded_count += 1

        # Add filtered documents to target collection
        if len(included_embeddings) > 0:
            # Use existing embeddings for faster insertion
            target.add(
                documents=included_docs,
                metadatas=included_metadatas,
                ids=included_ids,
                embeddings=included_embeddings,
            )
        else:
            # Let ChromaDB generate embeddings
            target.add(
                documents=included_docs,
                metadatas=included_metadatas,
                ids=included_ids,
            )

        stats = {
            "total": len(documents),
            "included": len(included_docs),
            "excluded": excluded_count,
        }

        logger.info(
            f"Created '{target_collection}': {stats['included']} included, "
            f"{stats['excluded']} excluded"
        )

        return stats

    def _should_include_for_analyst(self, categories: str) -> bool:
        """
        Determine if article should be included in analyst_kb.

        Args:
            categories: Categories string (comma-separated)

        Returns:
            True if article should be included, False otherwise
        """
        # Convert to string if not already (handles numpy arrays, lists, etc.)
        categories_str = str(categories) if categories is not None else ""

        # Exclude Machine Learning and Integration
        for excluded in self.EXCLUDED_CATEGORIES:
            if excluded.lower() in categories_str.lower():
                return False

        # Include if it matches any analyst category
        # If no categories specified, include by default
        if not categories_str or categories_str == "" or categories_str == "None":
            return True

        for allowed in self.ANALYST_CATEGORIES:
            if allowed.lower() in categories_str.lower():
                return True

        return False

    @staticmethod
    def _extract_title(doc: str) -> str:
        """
        Extract title from document.

        Tries YAML frontmatter first, then first # heading.
        Falls back to "Untitled" if not found.

        Args:
            doc: Document content

        Returns:
            Extracted title
        """
        import re

        # Try YAML frontmatter
        if doc.startswith("---"):
            lines = doc.split("\n")
            for line in lines[1:10]:
                if line.startswith("title:"):
                    return line.split(":", 1)[1].strip()
                if line == "---":
                    break

        # Try first # heading
        match = re.search(r"^#\s+(.+)$", doc, re.MULTILINE)
        if match:
            return match.group(1).strip()

        return "Untitled"

    @staticmethod
    def _extract_preview(doc: str, max_chars: int = 300) -> str:
        """
        Extract content preview, skipping YAML frontmatter.

        Args:
            doc: Document content
            max_chars: Maximum characters for preview

        Returns:
            Content preview
        """
        lines = doc.split("\n")
        preview_start = 0

        # Skip YAML frontmatter
        for i, line in enumerate(lines):
            if i > 0 and line == "---":
                preview_start = i + 1
                break

        preview = "\n".join(lines[preview_start:preview_start + 10]).strip()

        if len(preview) > max_chars:
            preview = preview[:max_chars] + "..."

        return preview

#!/usr/bin/env python3
"""
ChromaDB Client for QuantMind Hybrid Core v7.
Provides persistent vector storage using sentence-transformers all-MiniLM-L6-v2 (384-dim).
Uses existing ChromaDB storage at data/chromadb/ with cosine similarity.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    import chromadb
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
except ImportError:
    raise ImportError("ChromaDB not installed. Run: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")


logger = logging.getLogger(__name__)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_PATH = PROJECT_ROOT / "data" / "chromadb"


class SentenceTransformerEmbedding(EmbeddingFunction):
    """
    Custom embedding function using sentence-transformers all-MiniLM-L6-v2.
    Produces 384-dimensional embeddings with cosine similarity support.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model identifier
        """
        self._model: Optional[SentenceTransformer] = None
        self._model_name = model_name

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            logger.info(f"Loading sentence-transformers model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            logger.info(f"Model loaded. Embedding dimension: {self._model.get_sentence_embedding_dimension()}")
        return self._model

    def __call__(self, texts: Documents) -> Embeddings:
        """
        Generate embeddings for input texts.

        Args:
            texts: List of text documents to embed

        Returns:
            List of embedding vectors (384-dimensional)
        """
        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
            show_progress_bar=False
        )
        return embeddings.tolist()


class ChromaDBClient:
    """
    ChromaDB client for QuantMind Hybrid Core v7.
    Manages vector collections for strategies, knowledge, and market patterns.
    """

    # Collection names
    COLLECTION_STRATEGIES = "strategy_dna"
    COLLECTION_KNOWLEDGE = "market_research"
    COLLECTION_PATTERNS = "agent_memory"

    # HNSW index configuration for cosine similarity
    HNSW_CONFIG = {
        "hnsw:space": "cosine",
        "hnsw:M": 16,  # Connections per node (balanced recall/speed)
        "hnsw:construction_ef": 100,  # Build-time accuracy
        "hnsw:search_ef": 50  # Search-time accuracy
    }

    def __init__(self, persist_directory: Optional[Path] = None):
        """
        Initialize ChromaDB client.

        Args:
            persist_directory: Path to ChromaDB storage (defaults to data/chromadb/)
        """
        self._persist_directory = persist_directory or CHROMA_PATH
        self._client: Optional[chromadb.PersistentClient] = None
        self._embedding_function = SentenceTransformerEmbedding()
        self._collections: Dict[str, Any] = {}

        logger.info(f"ChromaDB client initialized with storage: {self._persist_directory}")

    @property
    def client(self) -> chromadb.PersistentClient:
        """Get or create ChromaDB client."""
        if self._client is None:
            logger.info(f"Initializing ChromaDB PersistentClient at {self._persist_directory}")
            self._client = chromadb.PersistentClient(path=str(self._persist_directory))
        return self._client

    @property
    def embedding_function(self) -> SentenceTransformerEmbedding:
        """Get the embedding function instance."""
        return self._embedding_function

    def get_or_create_collection(self, name: str) -> Any:
        """
        Get or create a ChromaDB collection.

        Args:
            name: Collection name

        Returns:
            ChromaDB collection object
        """
        if name not in self._collections:
            collection = self.client.get_or_create_collection(
                name=name,
                metadata=self.HNSW_CONFIG,
                embedding_function=self._embedding_function
            )
            self._collections[name] = collection
            logger.info(f"Collection '{name}' ready (count: {collection.count()})")

        return self._collections[name]

    @property
    def strategies_collection(self) -> Any:
        """Get the strategy_dna collection."""
        return self.get_or_create_collection(self.COLLECTION_STRATEGIES)

    @property
    def knowledge_collection(self) -> Any:
        """Get the market_research collection."""
        return self.get_or_create_collection(self.COLLECTION_KNOWLEDGE)

    @property
    def patterns_collection(self) -> Any:
        """Get the agent_memory collection."""
        return self.get_or_create_collection(self.COLLECTION_PATTERNS)

    def add_strategy(
        self,
        strategy_id: str,
        code: str,
        strategy_name: str,
        code_hash: str,
        performance_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a trading strategy to the strategies collection.

        Args:
            strategy_id: Unique identifier for the strategy
            code: Strategy code/document
            strategy_name: Human-readable name
            code_hash: Hash of the code for deduplication
            performance_metrics: Optional performance data (win_rate, profit_factor, etc.)
        """
        from datetime import datetime

        metadata = {
            "strategy_name": strategy_name,
            "code_hash": code_hash,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "performance_metrics": str(performance_metrics) if performance_metrics else "{}"
        }

        self.strategies_collection.add(
            ids=[strategy_id],
            documents=[code],
            metadatas=[metadata]
        )
        logger.info(f"Added strategy '{strategy_name}' ({strategy_id})")

    def add_knowledge(
        self,
        article_id: str,
        content: str,
        title: str,
        url: str,
        categories: str,
        relevance_score: float = 0.5
    ) -> None:
        """
        Add knowledge article to the knowledge collection.

        Args:
            article_id: Unique identifier for the article
            content: Article content
            title: Article title
            url: Article URL
            categories: Comma-separated category tags
            relevance_score: Relevance score (0-1)
        """
        metadata = {
            "title": title,
            "url": url,
            "categories": categories,
            "relevance_score": relevance_score
        }

        self.knowledge_collection.add(
            ids=[article_id],
            documents=[content],
            metadatas=[metadata]
        )
        logger.info(f"Added knowledge article '{title}' ({article_id})")

    def add_market_pattern(
        self,
        pattern_id: str,
        description: str,
        pattern_type: str,
        volatility_level: str,
        timestamp: Optional[str] = None
    ) -> None:
        """
        Add market pattern to the patterns collection.

        Args:
            pattern_id: Unique identifier for the pattern
            description: Pattern description
            pattern_type: Type of pattern (trend_following, mean_reversion, breakout, etc.)
            volatility_level: Market volatility (low, medium, high)
            timestamp: ISO timestamp of pattern detection
        """
        from datetime import datetime

        if timestamp is None:
            timestamp = datetime.utcnow().isoformat() + "Z"

        metadata = {
            "pattern_type": pattern_type,
            "timestamp": timestamp,
            "volatility_level": volatility_level
        }

        self.patterns_collection.add(
            ids=[pattern_id],
            documents=[description],
            metadatas=[metadata]
        )
        logger.info(f"Added market pattern '{pattern_type}' ({pattern_id})")

    def add_agent_memory(
        self,
        memory_id: str,
        content: str,
        agent_type: str,
        memory_type: str,
        context: str,
        importance: float = 0.5
    ) -> None:
        """
        Add agent memory to the agent_memory collection.

        Args:
            memory_id: Unique identifier for the memory
            content: Memory content
            agent_type: Type of agent (analyst/quant/executor)
            memory_type: Type of memory (semantic/episodic/procedural)
            context: Context in which memory was created
            importance: Importance score (0-1)
        """
        from datetime import datetime

        metadata = {
            "agent_type": agent_type,
            "memory_type": memory_type,
            "context": context,
            "importance": importance,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        self.patterns_collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[metadata]
        )
        logger.info(f"Added agent memory '{memory_type}' for {agent_type} ({memory_id})")

    def search_strategies(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search strategies by semantic similarity.

        Args:
            query: Search query text
            limit: Maximum number of results

        Returns:
            List of search results with documents and metadata
        """
        results = self.strategies_collection.query(
            query_texts=[query],
            n_results=limit
        )

        formatted_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                formatted_results.append({
                    "id": doc_id,
                    "document": results["documents"][0][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": results["distances"][0][i] if results.get("distances") else 0.0
                })

        return formatted_results

    def search_knowledge(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search knowledge articles by semantic similarity.

        Args:
            query: Search query text
            limit: Maximum number of results

        Returns:
            List of search results with documents and metadata
        """
        results = self.knowledge_collection.query(
            query_texts=[query],
            n_results=limit
        )

        formatted_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                formatted_results.append({
                    "id": doc_id,
                    "document": results["documents"][0][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": results["distances"][0][i] if results.get("distances") else 0.0
                })

        return formatted_results

    def search_patterns(
        self,
        query: str,
        limit: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search market patterns by semantic similarity.

        Args:
            query: Search query text
            limit: Maximum number of results
            where: Optional metadata filter

        Returns:
            List of search results with documents and metadata
        """
        results = self.patterns_collection.query(
            query_texts=[query],
            n_results=limit,
            where=where
        )

        formatted_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                formatted_results.append({
                    "id": doc_id,
                    "document": results["documents"][0][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": results["distances"][0][i] if results.get("distances") else 0.0
                })

        return formatted_results

    def list_collections(self) -> List[str]:
        """Get list of all collection names."""
        return [coll.name for coll in self.client.list_collections()]

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all QuantMind collections.

        Returns:
            Dictionary with collection counts and metadata
        """
        stats = {
            "persist_directory": str(self._persist_directory),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "similarity": "cosine",
            "collections": {}
        }

        for collection_name in [self.COLLECTION_STRATEGIES, self.COLLECTION_KNOWLEDGE, self.COLLECTION_PATTERNS]:
            collection = self.get_or_create_collection(collection_name)
            stats["collections"][collection_name] = {
                "count": collection.count(),
                "metadata": collection.metadata
            }

        return stats


# Singleton instance for global access
_chroma_client_instance: Optional[ChromaDBClient] = None


def get_chroma_client() -> ChromaDBClient:
    """
    Get the singleton ChromaDB client instance.

    Returns:
        ChromaDBClient instance
    """
    global _chroma_client_instance
    if _chroma_client_instance is None:
        _chroma_client_instance = ChromaDBClient()
    return _chroma_client_instance


# Initialize collections on module import
def init_collections() -> ChromaDBClient:
    """
    Initialize all required ChromaDB collections.

    Returns:
        ChromaDBClient instance with initialized collections
    """
    client = get_chroma_client()

    # Initialize all collections
    _ = client.strategies_collection
    _ = client.knowledge_collection
    _ = client.patterns_collection

    logger.info("ChromaDB collections initialized:")
    logger.info(f"  - {client.COLLECTION_STRATEGIES}")
    logger.info(f"  - {client.COLLECTION_KNOWLEDGE}")
    logger.info(f"  - {client.COLLECTION_PATTERNS}")

    return client


if __name__ == "__main__":
    # CLI for testing ChromaDB client
    import json

    logging.basicConfig(level=logging.INFO)

    print("ChromaDB Client for QuantMind Hybrid Core v7")
    print("=" * 50)

    client = init_collections()

    print("\nCollection Statistics:")
    stats = client.get_collection_stats()
    print(json.dumps(stats, indent=2))

    print("\nAvailable Collections:")
    for coll_name in client.list_collections():
        print(f"  - {coll_name}")

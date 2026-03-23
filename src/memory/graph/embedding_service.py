"""Embedding Service for Graph Memory System.

This module provides embedding generation and vector similarity search
using sentence-transformers (all-MiniLM-L6-v2) and optional ChromaDB backend.
"""
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 output dimension


class EmbeddingService:
    """Service for generating and managing vector embeddings.

    Uses sentence-transformers (all-MiniLM-L6-v2) for embedding generation.
    Supports both direct SQLite storage and ChromaDB backend.
    """

    def __init__(
        self,
        chroma_path: Optional[str] = None,
        use_chroma: bool = False,
    ) -> None:
        """Initialize the embedding service.

        Args:
            chroma_path: Path to ChromaDB persistence directory.
            use_chroma: If True, use ChromaDB for embedding storage.
        """
        self._model = None
        self._use_chroma = use_chroma
        self._chroma_collection = None
        self._chroma_path = chroma_path

    def _load_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(EMBEDDING_MODEL)
                logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
            except ImportError:
                logger.error(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                raise

        return self._model

    def _init_chroma(self):
        """Initialize ChromaDB backend if enabled."""
        if self._chroma_collection is not None:
            return

        if not self._use_chroma:
            return

        try:
            import chromadb

            if self._chroma_path:
                self._chroma_client = chromadb.PersistentClient(
                    path=self._chroma_path
                )
            else:
                self._chroma_client = chromadb.Client()

            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name="memory_embeddings",
                metadata={"dimension": EMBEDDING_DIMENSION}
            )
            logger.info(f"Initialized ChromaDB collection at {self._chroma_path}")
        except ImportError:
            logger.warning(
                "chromadb not installed. Using SQLite fallback. "
                "Install with: pip install chromadb"
            )
            self._use_chroma = False

    def generate_embedding(self, text: str) -> bytes:
        """Generate embedding vector for text.

        Args:
            text: Text to generate embedding for.

        Returns:
            Embedding as bytes (numpy array serialized).
        """
        model = self._load_model()
        embedding = model.encode(text, convert_to_numpy=True)

        # Serialize to bytes
        return embedding.tobytes()

    def generate_embeddings_batch(self, texts: list[str]) -> list[bytes]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to generate embeddings for.

        Returns:
            List of embeddings as bytes.
        """
        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        return [emb.tobytes() for emb in embeddings]

    def compute_similarity(self, embedding1: bytes, embedding2: bytes) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding as bytes.
            embedding2: Second embedding as bytes.

        Returns:
            Cosine similarity score (-1 to 1).
        """
        emb1 = np.frombuffer(embedding1, dtype=np.float32)
        emb2 = np.frombuffer(embedding2, dtype=np.float32)

        # Normalize
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)

        # Cosine similarity
        return float(np.dot(emb1, emb2))

    def search_similar(
        self,
        query_text: str,
        stored_embeddings: list[tuple[str, bytes]],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Search for similar embeddings using text query.

        Args:
            query_text: Query text.
            stored_embeddings: List of (id, embedding_bytes) tuples.
            top_k: Number of results to return.

        Returns:
            List of (id, similarity_score) tuples.
        """
        query_embedding = self.generate_embedding(query_text)

        similarities = []
        for node_id, embedding in stored_embeddings:
            sim = self.compute_similarity(query_embedding, embedding)
            similarities.append((node_id, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def add_to_chroma(
        self,
        node_id: str,
        text: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Add embedding to ChromaDB.

        Args:
            node_id: Node ID for the embedding.
            text: Text to embed.
            metadata: Optional metadata dict.

        Returns:
            True if successful.
        """
        self._init_chroma()

        if not self._use_chroma or self._chroma_collection is None:
            logger.warning("ChromaDB not initialized, cannot add embedding")
            return False

        try:
            embedding = self.generate_embedding(text)
            embedding_list = np.frombuffer(embedding, dtype=np.float32).tolist()

            self._chroma_collection.add(
                ids=[node_id],
                embeddings=[embedding_list],
                metadatas=[metadata or {}],
                documents=[text],
            )
            return True
        except Exception as e:
            logger.error(f"Failed to add embedding to ChromaDB: {e}")
            return False

    def search_chroma(
        self,
        query_text: str,
        n_results: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[dict]:
        """Search ChromaDB for similar embeddings.

        Args:
            query_text: Query text.
            n_results: Number of results.
            filter_metadata: Optional metadata filter.

        Returns:
            List of result dicts with id, distance, metadata.
        """
        self._init_chroma()

        if not self._use_chroma or self._chroma_collection is None:
            logger.warning("ChromaDB not initialized")
            return []

        try:
            model = self._load_model()
            query_embedding = model.encode(query_text, convert_to_numpy=True).tolist()

            results = self._chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata,
            )

            return [
                {
                    "id": results["ids"][0][i],
                    "distance": results["distances"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "document": results["documents"][0][i],
                }
                for i in range(len(results["ids"][0]))
            ]
        except Exception as e:
            logger.error(f"Failed to search ChromaDB: {e}")
            return []


# Global singleton
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service(
    chroma_path: Optional[str] = None,
    use_chroma: bool = False,
) -> EmbeddingService:
    """Get or create the global embedding service.

    Args:
        chroma_path: Path to ChromaDB persistence.
        use_chroma: If True, use ChromaDB backend.

    Returns:
        EmbeddingService instance.
    """
    global _embedding_service

    if _embedding_service is None:
        _embedding_service = EmbeddingService(
            chroma_path=chroma_path,
            use_chroma=use_chroma,
        )

    return _embedding_service


__all__ = [
    "EmbeddingService",
    "get_embedding_service",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
]

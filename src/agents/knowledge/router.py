"""
QuantMind Knowledge Router
Manages isolated namespaces in the Vector Database (Qdrant).

Features:
- Isolated "Private Offices" for agents (Analyst, QuantCode).
- Global "Town Hall" for shared standards.
- Super-User access for Co-pilot.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)

class KnowledgeRouter:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KnowledgeRouter, cls).__new__(cls)
            cls._instance._init_client()
        return cls._instance

    def _init_client(self):
        # Default to local Qdrant
        self.url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        self.api_key = os.environ.get("QDRANT_API_KEY", None)
        
        try:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
            logger.info(f"Connected to Knowledge Router (Qdrant at {self.url})")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self.client = None

    def search(self, namespace: str, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search within a specific agent's namespace.
        Also optionally searches the 'global' namespace for shared standards.
        """
        if not self.client: return []

        # 1. Search Private Namespace
        private_results = self.client.search(
            collection_name=namespace,
            query_vector=query_vector,
            limit=limit
        )

        # 2. Search Global Namespace (hardcoded 'quantmind_global')
        global_results = self.client.search(
            collection_name="quantmind_global",
            query_vector=query_vector,
            limit=2 # Add top 2 global standards
        )

        # Merge and Format
        combined = private_results + global_results
        return [
            {
                "content": hit.payload.get("content", ""),
                "source": hit.payload.get("source", "unknown"),
                "score": hit.score,
                "namespace": namespace if hit in private_results else "global"
            }
            for hit in combined
        ]

    def ingest(self, namespace: str, content: str, metadata: Dict[str, Any], vector: List[float]):
        """Save a memory/document to a specific namespace."""
        if not self.client: return

        self._ensure_collection(namespace)
        
        self.client.upsert(
            collection_name=namespace,
            points=[
                models.PointStruct(
                    id=models.generate_uuid(), # Auto-generate UUID
                    vector=vector,
                    payload={
                        "content": content,
                        **metadata
                    }
                )
            ]
        )

    def _ensure_collection(self, name: str):
        """Lazy creation of collections."""
        try:
            self.client.get_collection(name)
        except Exception:
            logger.info(f"Creating new knowledge namespace: {name}")
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
            )

# Global Singleton
kb_router = KnowledgeRouter()

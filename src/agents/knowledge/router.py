"""
QuantMind Knowledge Router
Manages knowledge retrieval via PageIndex services.

Features:
- Reasoning-based retrieval through PageIndex API.
- Three isolated collections: articles, books, logs.
- HTTP-based communication with PageIndex services.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import httpx

logger = logging.getLogger(__name__)


class PageIndexClient:
    """HTTP client for PageIndex services."""
    
    # Collection to port mapping
    COLLECTION_PORTS = {
        "articles": 3000,
        "books": 3001,
        "logs": 3002,
    }
    
    def __init__(
        self,
        articles_url: Optional[str] = None,
        books_url: Optional[str] = None,
        logs_url: Optional[str] = None,
    ):
        """Initialize PageIndex client with service URLs from environment or parameters.
        
        Args:
            articles_url: Optional override for articles service URL
            books_url: Optional override for books service URL
            logs_url: Optional override for logs service URL
        """
        # Set public attributes with parameter overrides or environment defaults
        self.articles_url = articles_url or os.environ.get("PAGEINDEX_ARTICLES_URL", "http://localhost:3000")
        self.books_url = books_url or os.environ.get("PAGEINDEX_BOOKS_URL", "http://localhost:3001")
        self.logs_url = logs_url or os.environ.get("PAGEINDEX_LOGS_URL", "http://localhost:3002")
        
        # Maintain base_urls dict for internal use
        self.base_urls = {
            "articles": self.articles_url,
            "books": self.books_url,
            "logs": self.logs_url,
        }
        self.timeout = httpx.Timeout(30.0)
        logger.info(f"PageIndexClient initialized with URLs: {self.base_urls}")
    
    def _get_port_for_collection(self, collection: str) -> int:
        """Get the port number for a specific collection.
        
        Args:
            collection: Collection name (articles, books, logs)
            
        Returns:
            Port number (3000, 3001, or 3002), defaults to 3000 for unknown collections
        """
        return self.COLLECTION_PORTS.get(collection, 3000)
    
    def _get_url_for_collection(self, collection: str) -> str:
        """Get the base URL for a specific collection."""
        if collection not in self.base_urls:
            raise ValueError(f"Unknown collection: {collection}. Valid collections: {list(self.base_urls.keys())}")
        return self.base_urls[collection]
    
    def search(self, query: str, collection: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search within a specific PageIndex collection.
        
        Args:
            query: The search query string
            collection: Collection name (articles, books, logs)
            limit: Maximum number of results to return
            
        Returns:
            List of search results with content, source, and page references
        """
        base_url = self._get_url_for_collection(collection)
        search_url = f"{base_url}/search"
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    search_url,
                    json={
                        "query": query,
                        "limit": limit,
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                # Format results with page/section references
                results = []
                for item in data.get("results", []):
                    results.append({
                        "content": item.get("content", ""),
                        "source": item.get("source", "unknown"),
                        "score": item.get("score", 0.0),
                        "page": item.get("page"),
                        "section": item.get("section"),
                        "collection": collection,
                    })
                return results
                
        except httpx.HTTPStatusError as e:
            logger.error(f"PageIndex search failed with status {e.response.status_code}: {e}")
            return []
        except httpx.RequestError as e:
            logger.error(f"PageIndex request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during PageIndex search: {e}")
            return []
    
    def health_check(self, collection: str) -> Dict[str, Any]:
        """Check health status of a PageIndex service."""
        base_url = self._get_url_for_collection(collection)
        health_url = f"{base_url}/health"
        
        try:
            with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
                response = client.get(health_url)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Health check failed for {collection}: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    def get_stats(self, collection: str) -> Dict[str, Any]:
        """Get statistics from a PageIndex service."""
        base_url = self._get_url_for_collection(collection)
        stats_url = f"{base_url}/stats"
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(stats_url)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Stats request failed for {collection}: {e}")
            return {"error": str(e)}


class KnowledgeRouter:
    """
    Singleton router for knowledge retrieval via PageIndex.
    Provides unified access to articles, books, and logs collections.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KnowledgeRouter, cls).__new__(cls)
            cls._instance._init_client()
        return cls._instance

    def _init_client(self):
        """Initialize the PageIndex client."""
        try:
            self.client = PageIndexClient()
            logger.info("Connected to Knowledge Router (PageIndex)")
        except Exception as e:
            logger.error(f"Failed to initialize PageIndex client: {e}")
            self.client = None

    def search(
        self,
        query: str,
        collection: str = "articles",
        limit: int = 5,
        include_global: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search within a specific collection.
        
        Args:
            query: The search query string
            collection: Collection name (articles, books, logs)
            limit: Maximum number of results
            include_global: Whether to include results from global standards
            
        Returns:
            List of search results with content, source, and page references
        """
        if not self.client:
            logger.warning("PageIndex client not initialized")
            return []

        # Search the specified collection
        results = self.client.search(query, collection, limit)
        
        # Optionally search global standards (books collection for shared knowledge)
        if include_global and collection != "books":
            global_results = self.client.search(query, "books", limit=2)
            for result in global_results:
                result["namespace"] = "global"
            results.extend(global_results)
        
        return results

    def search_all(self, query: str, limit_per_collection: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across all collections.
        
        Args:
            query: The search query string
            limit_per_collection: Max results per collection
            
        Returns:
            Dictionary mapping collection names to their results
        """
        if not self.client:
            return {}
        
        results = {}
        for collection in ["articles", "books", "logs"]:
            results[collection] = self.client.search(query, collection, limit_per_collection)
        
        return results

    def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all PageIndex services."""
        if not self.client:
            return {}
        
        health_status = {}
        for collection in ["articles", "books", "logs"]:
            health_status[collection] = self.client.health_check(collection)
        
        return health_status

    def get_stats_all(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all PageIndex services."""
        if not self.client:
            return {}
        
        stats = {}
        for collection in ["articles", "books", "logs"]:
            stats[collection] = self.client.get_stats(collection)
        
        return stats


# Global Singleton
kb_router = KnowledgeRouter()
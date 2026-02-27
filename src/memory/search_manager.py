"""
QuantMindX Search Manager

Advanced search functionality with:
- Vector similarity search
- Full-text search (FTS)
- Hybrid search combining both
- MMR (Maximal Marginal Relevance) for diverse results
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Literal, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class SearchMethod(str, Enum):
    """Search methods."""
    VECTOR = "vector"  # Vector similarity search
    FTS = "fts"  # Full-text search
    HYBRID = "hybrid"  # Combined vector + FTS
    MMR = "mmr"  # Maximal Marginal Relevance


@dataclass
class SearchResult:
    """
    A search result with relevance score.
    
    Attributes:
        entry_id: Memory entry ID
        content: Memory content
        score: Relevance/similarity score (0-1, higher is better)
        method: Search method that produced this result
        metadata: Additional metadata
        source: Memory source
    """
    entry_id: str
    content: str
    score: float
    method: SearchMethod
    metadata: Dict[str, Any]
    source: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "content": self.content,
            "score": self.score,
            "method": self.method.value,
            "metadata": self.metadata,
            "source": self.source,
        }


class SearchManager:
    """
    Advanced search manager with multiple search strategies.
    
    Supports:
    - Vector similarity search using cosine similarity
    - Full-text search using FTS5
    - Hybrid search combining both methods
    - MMR for diverse, non-redundant results
    
    Example:
        >>> manager = SearchManager(memory_manager)
        >>> 
        >>> # Vector search
        >>> results = await manager.search(
        ...     query_embedding=[0.1, 0.2, ...],
        ...     method=SearchMethod.VECTOR,
        ...     limit=10
        ... )
        >>> 
        >>> # Hybrid search
        >>> results = await manager.search(
        ...     query_text="trading strategy",
        ...     query_embedding=[0.1, 0.2, ...],
        ...     method=SearchMethod.HYBRID,
        ...     limit=10
        ... )
        >>> 
        >>> # MMR for diversity
        >>> results = await manager.search(
        ...     query_embedding=[0.1, 0.2, ...],
        ...     method=SearchMethod.MMR,
        ...     limit=10,
        ...     lambda_mult=0.5  # Balance relevance/diversity
        ... )
    """
    
    def __init__(
        self,
        memory_manager,
        default_method: SearchMethod = SearchMethod.VECTOR,
        default_limit: int = 10,
    ):
        """
        Initialize search manager.
        
        Args:
            memory_manager: MemoryManager instance
            default_method: Default search method
            default_limit: Default result limit
        """
        self.memory_manager = memory_manager
        self.default_method = default_method
        self.default_limit = default_limit
        
        logger.info(
            f"SearchManager initialized: method={default_method}, limit={default_limit}"
        )
    
    async def search(
        self,
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        method: Optional[SearchMethod] = None,
        limit: Optional[int] = None,
        source: Optional[str] = None,
        min_importance: float = 0.0,
        lambda_mult: float = 0.5,
        fetch_k: int = 50,
        **kwargs
    ) -> List[SearchResult]:
        """
        Perform search using specified method.
        
        Args:
            query_text: Text query for FTS
            query_embedding: Query embedding for vector search
            method: Search method (default: default_method)
            limit: Maximum results (default: default_limit)
            source: Filter by memory source
            min_importance: Minimum importance threshold
            lambda_mult: MMR lambda parameter (0=diversity, 1=relevance)
            fetch_k: Candidate pool size for MMR
            **kwargs: Additional method-specific parameters
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        method = method or self.default_method
        limit = limit or self.default_limit
        
        if method == SearchMethod.VECTOR:
            return await self._search_vector(
                query_embedding=query_embedding,
                limit=limit,
                source=source,
                min_importance=min_importance,
                **kwargs
            )
        
        elif method == SearchMethod.FTS:
            return await self._search_fts(
                query_text=query_text,
                limit=limit,
                source=source,
                **kwargs
            )
        
        elif method == SearchMethod.HYBRID:
            return await self._search_hybrid(
                query_text=query_text,
                query_embedding=query_embedding,
                limit=limit,
                source=source,
                min_importance=min_importance,
                **kwargs
            )
        
        elif method == SearchMethod.MMR:
            return await self._search_mmr(
                query_embedding=query_embedding,
                limit=limit,
                source=source,
                min_importance=min_importance,
                lambda_mult=lambda_mult,
                fetch_k=fetch_k,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unknown search method: {method}")
    
    async def _search_vector(
        self,
        query_embedding: List[float],
        limit: int,
        source: Optional[str] = None,
        min_importance: float = 0.0,
    ) -> List[SearchResult]:
        """Vector similarity search."""
        if query_embedding is None:
            raise ValueError("query_embedding required for vector search")
        
        # Search via memory manager
        results = await self.memory_manager.search_similar(
            query_embedding=query_embedding,
            limit=limit,
            source=source,
            min_importance=min_importance,
        )
        
        # Convert to SearchResult
        search_results = []
        for entry, similarity in results:
            search_results.append(SearchResult(
                entry_id=entry.id,
                content=entry.content,
                score=similarity,
                method=SearchMethod.VECTOR,
                metadata=entry.metadata,
                source=entry.source.value,
            ))
        
        return search_results
    
    async def _search_fts(
        self,
        query_text: str,
        limit: int,
        source: Optional[str] = None,
    ) -> List[SearchResult]:
        """Full-text search."""
        if query_text is None:
            raise ValueError("query_text required for FTS")
        
        # Search via memory manager
        results = await self.memory_manager.search_fts(
            query=query_text,
            limit=limit,
            source=source,
        )
        
        # Convert to SearchResult
        search_results = []
        for entry, rank in results:
            search_results.append(SearchResult(
                entry_id=entry.id,
                content=entry.content,
                score=rank,
                method=SearchMethod.FTS,
                metadata=entry.metadata,
                source=entry.source.value,
            ))
        
        return search_results
    
    async def _search_hybrid(
        self,
        query_text: Optional[str],
        query_embedding: Optional[List[float]],
        limit: int,
        source: Optional[str] = None,
        min_importance: float = 0.0,
        vector_weight: float = 0.7,
        fts_weight: float = 0.3,
    ) -> List[SearchResult]:
        """
        Hybrid search combining vector and FTS.
        
        Scores are combined using weighted average:
        final_score = vector_weight * vector_score + fts_weight * fts_score
        """
        results_by_id: Dict[str, SearchResult] = {}
        
        # Get vector results
        if query_embedding is not None:
            vector_results = await self._search_vector(
                query_embedding=query_embedding,
                limit=limit * 2,  # Get more candidates
                source=source,
                min_importance=min_importance,
            )
            
            for result in vector_results:
                results_by_id[result.entry_id] = result
        
        # Get FTS results
        if query_text is not None:
            fts_results = await self._search_fts(
                query_text=query_text,
                limit=limit * 2,
                source=source,
            )
            
            for result in fts_results:
                if result.entry_id in results_by_id:
                    # Combine scores
                    existing = results_by_id[result.entry_id]
                    combined_score = (
                        vector_weight * existing.score +
                        fts_weight * result.score
                    )
                    existing.score = combined_score
                    existing.method = SearchMethod.HYBRID
                else:
                    result.score = fts_weight * result.score
                    result.method = SearchMethod.HYBRID
                    results_by_id[result.entry_id] = result
        
        # Sort by combined score
        sorted_results = sorted(
            results_by_id.values(),
            key=lambda r: r.score,
            reverse=True
        )
        
        return sorted_results[:limit]
    
    async def _search_mmr(
        self,
        query_embedding: List[float],
        limit: int,
        source: Optional[str] = None,
        min_importance: float = 0.0,
        lambda_mult: float = 0.5,
        fetch_k: int = 50,
    ) -> List[SearchResult]:
        """
        Maximal Marginal Relevance search.
        
        MMR balances relevance to query with diversity among results.
        
        Formula:
            MMR = argmax [lambda * Sim(d, q) - (1 - lambda) * max Sim(d, d_selected)]
        
        Where:
            - d: candidate document
            - q: query
            - d_selected: already selected documents
            - lambda: balance parameter (0=diversity, 1=relevance)
        """
        if query_embedding is None:
            raise ValueError("query_embedding required for MMR")
        
        # Fetch candidate pool
        candidates = await self.memory_manager.search_similar(
            query_embedding=query_embedding,
            limit=fetch_k,
            source=source,
            min_importance=min_importance,
        )
        
        if not candidates:
            return []
        
        # Convert to numpy for efficient computation
        candidate_embeddings = np.array([
            e[0].embedding for e in candidates if e[0].embedding
        ])
        candidate_entries = [e[0] for e in candidates]
        candidate_similarities = np.array([e[1] for e in candidates])
        
        query_vec = np.array(query_embedding)
        
        selected_indices = []
        remaining_indices = list(range(len(candidates)))
        
        # Select first result (most similar to query)
        if remaining_indices:
            first_idx = np.argmax(candidate_similarities)
            selected_indices.append(first_idx)
            remaining_indices.remove(first_idx)
        
        # Iteratively select results using MMR
        while len(selected_indices) < limit and remaining_indices:
            best_score = -float('inf')
            best_idx = None
            
            selected_vecs = candidate_embeddings[selected_indices]
            
            for idx in remaining_indices:
                candidate_vec = candidate_embeddings[idx]
                relevance = candidate_similarities[idx]
                
                # Compute max similarity to selected documents
                if len(selected_vecs) > 0:
                    # Cosine similarity
                    sims = np.dot(selected_vecs, candidate_vec) / (
                        np.linalg.norm(selected_vecs, axis=1) *
                        np.linalg.norm(candidate_vec)
                    )
                    max_sim_to_selected = np.max(sims)
                else:
                    max_sim_to_selected = 0.0
                
                # MMR score
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim_to_selected
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        # Convert selected to SearchResult
        search_results = []
        for idx in selected_indices:
            entry = candidate_entries[idx]
            similarity = candidate_similarities[idx]
            
            search_results.append(SearchResult(
                entry_id=entry.id,
                content=entry.content,
                score=similarity,
                method=SearchMethod.MMR,
                metadata=entry.metadata,
                source=entry.source.value,
            ))
        
        return search_results
    
    async def diverse_search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        source: Optional[str] = None,
        diversity_buckets: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Search with diversity across buckets/categories.
        
        Ensures results are distributed across specified buckets/tags.
        
        Args:
            query_embedding: Query embedding
            limit: Total results
            source: Filter by source
            diversity_buckets: Tags/categories to diversify across
            
        Returns:
            Diversified search results
        """
        # Fetch candidates
        candidates = await self.memory_manager.search_similar(
            query_embedding=query_embedding,
            limit=limit * 3,  # Get more for diversity
            source=source,
        )
        
        # Group by bucket (tags)
        if diversity_buckets:
            buckets = {bucket: [] for bucket in diversity_buckets}
            buckets["other"] = []
            
            for entry, score in candidates:
                matched = False
                for bucket in diversity_buckets:
                    if bucket in entry.tags:
                        buckets[bucket].append((entry, score))
                        matched = True
                        break
                if not matched:
                    buckets["other"].append((entry, score))
            
            # Distribute results across buckets
            results = []
            per_bucket = max(1, limit // (len(buckets)))
            
            for bucket_entries in buckets.values():
                results.extend(bucket_entries[:per_bucket])
            
            # Sort by score
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:limit]
        else:
            results = candidates[:limit]
        
        # Convert to SearchResult
        search_results = []
        for entry, score in results:
            search_results.append(SearchResult(
                entry_id=entry.id,
                content=entry.content,
                score=score,
                method=SearchMethod.VECTOR,
                metadata=entry.metadata,
                source=entry.source.value,
            ))
        
        return search_results


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Similarity score (-1 to 1, where 1 is identical)
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same dimension")
    
    import numpy as np
    
    a = np.array(vec1)
    b = np.array(vec2)
    
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize scores to 0-1 range.
    
    Args:
        scores: List of scores
        
    Returns:
        Normalized scores
    """
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [1.0] * len(scores)
    
    return [(s - min_score) / (max_score - min_score) for s in scores]

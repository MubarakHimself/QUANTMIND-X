#!/usr/bin/env python3
"""
Focused tests for ChromaDB operations for QuantMind Hybrid Core v7.
Tests cover: collection creation, embedding generation, vector search, metadata storage.

Run only these tests: pytest tests/database/test_chromadb_collections.py -v
"""

import pytest
import sys
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestChromaDBCollections:
    """Test ChromaDB collection creation for QuantMind Hybrid Core."""

    @pytest.fixture
    def temp_chroma_path(self):
        """Create temporary directory for test ChromaDB data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def chroma_client(self, temp_chroma_path):
        """Create ChromaDB client with temporary storage."""
        try:
            import chromadb
        except ImportError:
            pytest.skip("ChromaDB not installed")

        client = chromadb.PersistentClient(path=temp_chroma_path)
        return client

    def test_collection_creation_with_cosine_similarity(self, chroma_client):
        """Test collection creation with cosine similarity configuration."""
        collection = chroma_client.get_or_create_collection(
            name="quantmind_strategies",
            metadata={"hnsw:space": "cosine"}
        )

        assert collection is not None
        assert collection.name == "quantmind_strategies"
        assert collection.count() == 0

    def test_quantmind_strategies_collection_metadata(self, chroma_client):
        """Test quantmind_strategies collection accepts strategy metadata."""
        collection = chroma_client.get_or_create_collection(
            name="quantmind_strategies",
            metadata={"hnsw:space": "cosine"}
        )

        # Insert strategy with required metadata
        collection.add(
            ids=["strategy_001"],
            documents=["RSI mean reversion strategy with dynamic position sizing"],
            metadatas=[{
                "strategy_name": "RSI Dynamic",
                "code_hash": "abc123def456",
                "created_at": "2026-01-30T00:00:00Z",
                "performance_metrics": '{"win_rate": 0.65, "profit_factor": 1.8}'
            }]
        )

        assert collection.count() == 1

        # Verify retrieval with metadata
        results = collection.get(ids=["strategy_001"], include=["metadatas"])
        assert results["metadatas"][0]["strategy_name"] == "RSI Dynamic"
        assert results["metadatas"][0]["code_hash"] == "abc123def456"
        assert "performance_metrics" in results["metadatas"][0]

    def test_quantmind_knowledge_collection_metadata(self, chroma_client):
        """Test quantmind_knowledge collection accepts article metadata."""
        collection = chroma_client.get_or_create_collection(
            name="quantmind_knowledge",
            metadata={"hnsw:space": "cosine"}
        )

        # Insert knowledge article with required metadata
        collection.add(
            ids=["article_001"],
            documents=["MQL5 article about RSI divergence trading strategies"],
            metadatas=[{
                "title": "RSI Divergence Trading",
                "url": "https://mql5.com/articles/rsi-divergence",
                "categories": "Trading Systems,Indicators",
                "relevance_score": 0.92
            }]
        )

        assert collection.count() == 1

        # Verify retrieval with metadata
        results = collection.get(ids=["article_001"], include=["metadatas"])
        assert results["metadatas"][0]["title"] == "RSI Divergence Trading"
        assert results["metadatas"][0]["categories"] == "Trading Systems,Indicators"
        assert results["metadatas"][0]["relevance_score"] == 0.92

    def test_market_patterns_collection_metadata(self, chroma_client):
        """Test market_patterns collection accepts regime metadata."""
        collection = chroma_client.get_or_create_collection(
            name="market_patterns",
            metadata={"hnsw:space": "cosine"}
        )

        # Insert market pattern with required metadata
        collection.add(
            ids=["pattern_001"],
            documents=["Bullish trend with low volatility regime detected"],
            metadatas=[{
                "pattern_type": "trend_following",
                "timestamp": "2026-01-30T12:00:00Z",
                "volatility_level": "low"
            }]
        )

        assert collection.count() == 1

        # Verify retrieval with metadata
        results = collection.get(ids=["pattern_001"], include=["metadatas"])
        assert results["metadatas"][0]["pattern_type"] == "trend_following"
        assert results["metadatas"][0]["volatility_level"] == "low"

    def test_all_collections_created(self, chroma_client):
        """Test that all 3 required collections are created and accessible."""
        expected_collections = {
            "quantmind_strategies",
            "quantmind_knowledge",
            "market_patterns"
        }

        # Create all required collections
        for collection_name in expected_collections:
            chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

        # List all collections
        existing_collections = {coll.name for coll in chroma_client.list_collections()}

        assert existing_collections == expected_collections

    def test_vector_search_with_cosine_similarity(self, chroma_client):
        """Test vector search returns relevant results using cosine similarity."""
        collection = chroma_client.get_or_create_collection(
            name="quantmind_strategies",
            metadata={"hnsw:space": "cosine"}
        )

        # Insert multiple strategies
        collection.add(
            ids=["strat_001", "strat_002", "strat_003"],
            documents=[
                "RSI mean reversion strategy for range-bound markets",
                "Moving average crossover trend following strategy",
                "Bollinger bands breakout strategy for volatile markets"
            ],
            metadatas=[
                {"strategy_name": "RSI Mean Reversion", "pattern_type": "range"},
                {"strategy_name": "MA Crossover", "pattern_type": "trend"},
                {"strategy_name": "BB Breakout", "pattern_type": "breakout"}
            ]
        )

        # Query for similar strategies
        results = collection.query(
            query_texts=["mean reversion trading strategy"],
            n_results=2
        )

        assert len(results["ids"][0]) == 2
        # RSI Mean Reversion should be most relevant
        assert "strat_001" in results["ids"][0]

    def test_metadata_filtering(self, chroma_client):
        """Test metadata filtering on collections."""
        collection = chroma_client.get_or_create_collection(
            name="market_patterns",
            metadata={"hnsw:space": "cosine"}
        )

        # Insert patterns with different volatility levels
        collection.add(
            ids=["pat_001", "pat_002", "pat_003"],
            documents=[
                "Low volatility bullish trend",
                "High volatility breakout",
                "Medium volatility consolidation"
            ],
            metadatas=[
                {"pattern_type": "trend", "volatility_level": "low"},
                {"pattern_type": "breakout", "volatility_level": "high"},
                {"pattern_type": "range", "volatility_level": "medium"}
            ]
        )

        # Query with metadata filter
        results = collection.query(
            query_texts=["market regime"],
            n_results=10,
            where={"volatility_level": "low"}
        )

        assert len(results["ids"][0]) == 1
        assert results["metadatas"][0][0]["volatility_level"] == "low"

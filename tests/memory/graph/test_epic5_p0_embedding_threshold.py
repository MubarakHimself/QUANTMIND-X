"""P0 Tests: Vector Embedding Semantic Search Relevance (Epic 5 - R-003)

These tests verify that vector embedding semantic search returns relevant
results based on cosine similarity thresholds.

Test Design Reference: _bmad-output/test-artifacts/test-design-epic-5.md
Risk: R-003 (Score 6) - Vector embedding quality degradation
Priority: P0 - High risk (>=6)
"""

import pytest
import tempfile

from src.memory.graph.embedding_service import EmbeddingService


class TestEmbeddingSimilarityThreshold:
    """Test R-003: Vector embedding semantic search relevance threshold."""

    def test_semantic_search_returns_nodes_above_threshold(self):
        """P0: Semantic search MUST return nodes with cosine similarity >= 0.7.

        When searching for semantically similar content, the system must
        return only results that meet the minimum relevance threshold
        to avoid noise in search results.
        """
        # Create embedding service with in-memory mode
        service = EmbeddingService(use_chroma=False)

        # Generate embeddings for semantically related content
        query_text = "EURUSD trading strategy for trending markets"

        # These texts are semantically related to the query
        relevant_texts = [
            "EURUSD trend following strategy for strong momentum",
            "Trading EURUSD using trend analysis",
            "EURUSD momentum-based entry signals",
        ]

        # This text is NOT semantically related
        irrelevant_text = "How to cook pasta al dente"

        # Generate query embedding
        query_embedding = service.generate_embedding(query_text)

        # Generate embeddings for relevant texts
        relevant_embeddings = [
            (f"node-{i}", service.generate_embedding(text))
            for i, text in enumerate(relevant_texts)
        ]

        # Generate embedding for irrelevant text
        irrelevant_embedding = ("node-irrelevant", service.generate_embedding(irrelevant_text))

        # Search for similar
        results = service.search_similar(
            query_text=query_text,
            stored_embeddings=relevant_embeddings + [irrelevant_embedding],
            top_k=5,
        )

        # Extract similarity scores
        result_ids = [r[0] for r in results]
        result_scores = {r[0]: r[1] for r in results}

        # CRITICAL ASSERTION: All returned results must have similarity >= 0.7
        SIMILARITY_THRESHOLD = 0.7
        for node_id, score in result_scores.items():
            assert score >= SIMILARITY_THRESHOLD, (
                f"Node {node_id} returned with similarity {score:.3f}, "
                f"which is below threshold {SIMILARITY_THRESHOLD}. "
                f"Low-quality search results compromise system reliability."
            )

        # Verify irrelevant text is NOT in results (or has very low score)
        irrelevant_score = result_scores.get("node-irrelevant", 0)
        assert irrelevant_score < SIMILARITY_THRESHOLD, (
            f"Irrelevant text 'pasta al dente' was returned with score {irrelevant_score:.3f}! "
            f"Semantic search is returning noise."
        )

    def test_semantic_search_returns_empty_for_irrelevant_query(self):
        """P0: Semantic search MUST return empty/low results for irrelevant queries.

        When searching with a query that has no semantic relationship to
        stored content, the system should not return false positives.
        """
        service = EmbeddingService(use_chroma=False)

        # Query about trading
        query_text = "Best GBPUSD momentum strategy for intraday trading"

        # Stored content is completely unrelated
        unrelated_content = [
            ("recipe-1", "How to make perfect chocolate chip cookies"),
            ("recipe-2", "Best pasta sauce recipe with tomatoes"),
            ("weather-1", "Sunny weather forecast for the weekend"),
        ]

        # Generate embeddings for unrelated content
        stored_embeddings = []
        for node_id, text in unrelated_content:
            emb = service.generate_embedding(text)
            stored_embeddings.append((node_id, emb))

        # Search
        results = service.search_similar(
            query_text=query_text,
            stored_embeddings=stored_embeddings,
            top_k=3,
        )

        # All results should have very low similarity
        SIMILARITY_THRESHOLD = 0.7
        for node_id, score in results:
            assert score < SIMILARITY_THRESHOLD, (
                f"Query '{query_text}' returned unrelated node '{node_id}' "
                f"with score {score:.3f}. Semantic search is producing false positives!"
            )

    def test_embedding_similarity_is_symmetric(self):
        """P0: Cosine similarity MUST be symmetric (sim(a,b) == sim(b,a)).

        This is a mathematical property of cosine similarity that must hold
        for the embedding service to be consistent.
        """
        service = EmbeddingService(use_chroma=False)

        text_a = "EURUSD trading strategy development"
        text_b = "Developing forex trading strategies for EURUSD"

        emb_a = service.generate_embedding(text_a)
        emb_b = service.generate_embedding(text_b)

        sim_ab = service.compute_similarity(emb_a, emb_b)
        sim_ba = service.compute_similarity(emb_b, emb_a)

        assert abs(sim_ab - sim_ba) < 0.001, (
            f"Cosine similarity is not symmetric: sim(a,b)={sim_ab:.4f}, sim(b,a)={sim_ba:.4f}. "
            f"This indicates a bug in the embedding service."
        )

    def test_embedding_quality_deterministic(self):
        """P0: Same text MUST produce same embedding (deterministic).

        Embeddings must be deterministic for the system to be testable
        and reproducible.
        """
        service = EmbeddingService(use_chroma=False)

        text = "Consistent EURUSD analysis for trading decisions"

        emb1 = service.generate_embedding(text)
        emb2 = service.generate_embedding(text)

        similarity = service.compute_similarity(emb1, emb2)

        # Same text should produce identical embeddings (similarity = 1.0)
        assert similarity > 0.999, (
            f"Same text produced non-identical embeddings (similarity={similarity:.4f}). "
            f"Embeddings must be deterministic for reproducibility."
        )

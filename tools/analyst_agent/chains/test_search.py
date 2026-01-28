"""
Test file for search chain implementation.
"""

import asyncio
from unittest.mock import Mock, AsyncMock
from typing import List, Dict, Any

# Mock the necessary imports for testing
class MockLLM:
    async def ainvoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "queries": [
                "opening range breakout strategy implementation MQL5",
                "how to code breakout entry signals with volume confirmation",
                "ATR based stop loss placement forex expert advisor"
            ],
            "query_count": 3,
            "rationale": "Generated queries covering implementation, entry signals, and risk management"
        }

class MockChromaKBClient:
    async def search(self, query: str, n: int = 5) -> List[Dict[str, Any]]:
        return [
            {
                "title": "Opening Range Breakout Strategy Implementation",
                "file_path": "articles/orb_strategy.md",
                "categories": "Trading Systems, Expert Advisors",
                "score": 0.92,
                "preview": "This article provides a complete implementation of the ORB strategy..."
            },
            {
                "title": "Volume Confirmation for Breakout Entries",
                "file_path": "articles/volume_breakout.md",
                "categories": "Trading, Indicators",
                "score": 0.88,
                "preview": "Learn how to implement volume confirmation for breakout entries..."
            }
        ]

async def test_search_chain():
    """Test the search chain functionality."""
    # Create mock objects
    mock_llm = MockLLM()
    mock_kb_client = MockChromaKBClient()

    # Test data
    extracted_concepts = {
        "strategy_name": "ORB Breakout Scalper",
        "mentioned_concepts": [
            "opening range breakout",
            "London session",
            "volume confirmation",
            "ATR-based stops"
        ]
    }

    # Run the search chain
    results = await search_chain(
        extracted_concepts=extracted_concepts,
        kb_client=mock_kb_client,
        llm=mock_llm,
        max_results_per_query=2
    )

    # Verify results
    assert len(results) > 0, "Should return search results"
    assert "title" in results[0], "Results should have title field"
    assert "score" in results[0], "Results should have score field"
    assert results[0]["score"] >= results[-1]["score"], "Results should be sorted by score"

    print(f"Test passed! Found {len(results)} unique results")
    for i, result in enumerate(results[:3], 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.2f})")

if __name__ == "__main__":
    asyncio.run(test_search_chain())
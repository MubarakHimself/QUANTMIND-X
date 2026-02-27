"""
QuantMindX Memory System - Usage Examples

This script demonstrates how to use the memory system components.
"""

import asyncio
from pathlib import Path
from datetime import datetime, timezone, timedelta

from src.memory import (
    # Memory Manager
    MemoryManager,
    MemorySource,
    create_memory_manager,
    # Embeddings
    get_embedding_provider,
    # Search
    SearchManager,
    SearchMethod,
    # Session Memory
    SessionMemory,
    SessionManager,
    MessageRole,
    # Temporal Decay
    TemporalDecay,
    DecayConfig,
    DecayType,
    DECAY_DEFAULT,
)


async def example_basic_memory():
    """Basic memory storage and retrieval."""
    print("\n=== Basic Memory Example ===\n")
    
    # Create memory manager
    manager = await create_memory_manager(
        db_path=Path("data/test_memory.db"),
        embedding_dim=1536,
    )
    
    # Add a memory
    entry = await manager.add_memory(
        source=MemorySource.MEMORY,
        content="EURUSD shows mean reversion behavior on H1 timeframe",
        importance=0.8,
        tags=["trading", "forex", "eurusd"],
    )
    print(f"Added memory: {entry.id}")
    
    # Retrieve the memory
    retrieved = await manager.get_memory(entry.id)
    print(f"Retrieved: {retrieved.content if retrieved else 'Not found'}")
    
    # Get stats
    stats = await manager.get_stats()
    print(f"Stats: {stats.total_entries} entries")
    
    await manager.close()


async def example_with_embeddings():
    """Memory with embeddings for similarity search."""
    print("\n=== Embeddings Example ===\n")
    
    # Create embedding provider (uses mock for demo)
    provider = get_embedding_provider("mock", dimension=1536)
    
    # Generate embeddings
    texts = [
        "EURUSD shows mean reversion behavior",
        "GBPUSD is trending upward",
        "USDJPY is range-bound",
        "Trading strategy: Bollinger Bands breakout",
    ]
    
    embeddings = await provider.embed_batch(texts)
    print(f"Generated {len(embeddings)} embeddings")
    
    # Create memory manager
    manager = await create_memory_manager(
        db_path=Path("data/test_memory_embeddings.db"),
        embedding_dim=1536,
    )
    
    # Add memories with embeddings
    for text, embedding in zip(texts, embeddings):
        await manager.add_memory(
            source=MemorySource.MEMORY,
            content=text,
            embedding=embedding,
            importance=0.7,
        )
    
    print("Added memories with embeddings")
    
    # Search by similarity
    query_embedding = embeddings[0]  # Search for similar to first
    results = await manager.search_similar(
        query_embedding=query_embedding,
        limit=3,
    )
    
    print(f"\nSimilar results:")
    for entry, score in results:
        print(f"  [{score:.3f}] {entry.content}")
    
    await manager.close()


async def example_session_memory():
    """Session-based conversation tracking."""
    print("\n=== Session Memory Example ===\n")
    
    # Create session
    session = SessionMemory(
        session_id="trading_chat_001",
        persist_path=Path("sessions/trading_chat_001.json"),
    )
    
    # Add conversation messages
    session.add_message(
        role=MessageRole.SYSTEM,
        content="You are a trading strategy assistant.",
    )
    
    session.add_message(
        role=MessageRole.USER,
        content="What's a good strategy for EURUSD?",
    )
    
    session.add_message(
        role=MessageRole.ASSISTANT,
        content="For EURUSD, consider mean reversion strategies using Bollinger Bands.",
    )
    
    session.add_message(
        role=MessageRole.USER,
        content="What timeframes work best?",
    )
    
    # Get conversation history
    messages = session.get_messages()
    print(f"Conversation has {len(messages)} messages")
    
    # Get OpenAI format
    openai_msgs = session.to_openai_format()
    print(f"OpenAI format: {len(openai_msgs)} messages")
    
    # Persist session
    await session.persist()
    print("Session persisted")


async def example_temporal_decay():
    """Time-based relevance scoring."""
    print("\n=== Temporal Decay Example ===\n")
    
    # Create temporal decay with default config (24h half-life)
    decay = TemporalDecay(config=DECAY_DEFAULT)
    
    now = datetime.now(timezone.utc)
    
    # Calculate decayed importance for memories of different ages
    memories = [
        ("Just now", 0.8, now),
        ("1 hour ago", 0.8, now - timedelta(hours=1)),
        ("6 hours ago", 0.8, now - timedelta(hours=6)),
        ("12 hours ago", 0.8, now - timedelta(hours=12)),
        ("24 hours ago", 0.8, now - timedelta(hours=24)),
        ("48 hours ago", 0.8, now - timedelta(hours=48)),
    ]
    
    print("Time-based importance decay (24h half-life):")
    print(f"{'Age':<15} {'Original':>10} {'Decayed':>10} {'Factor':>10}")
    print("-" * 50)
    
    for label, importance, created_at in memories:
        decayed = decay.apply_decay(importance, created_at)
        factor = decay.calculate_decay_factor(created_at)
        print(f"{label:<15} {importance:>10.2f} {decayed:>10.2f} {factor:>10.2f}")


async def example_hybrid_search():
    """Hybrid search combining vector and full-text."""
    print("\n=== Hybrid Search Example ===\n")
    
    # Setup memory manager with embeddings
    provider = get_embedding_provider("mock", dimension=1536)
    manager = await create_memory_manager(
        db_path=Path("data/test_hybrid_search.db"),
        embedding_dim=1536,
    )
    
    # Add sample memories
    memories = [
        "EURUSD mean reversion strategy using Bollinger Bands",
        "GBPUSD trend following with moving averages",
        "USDJPY range trading strategy",
        "Risk management: position sizing with Kelly criterion",
        "Backtesting tips: use walk-forward analysis",
    ]
    
    embeddings = await provider.embed_batch(memories)
    
    for text, emb in zip(memories, embeddings):
        await manager.add_memory(
            source=MemorySource.MEMORY,
            content=text,
            embedding=emb,
            tags=["trading"],
        )
    
    # Create search manager
    search = SearchManager(manager)
    
    # Vector search
    print("Vector search:")
    vector_results = await search.search(
        query_embedding=embeddings[0],
        method=SearchMethod.VECTOR,
        limit=3,
    )
    for r in vector_results:
        print(f"  [{r.score:.3f}] {r.content}")
    
    # FTS search
    print("\nFull-text search:")
    fts_results = await search.search(
        query_text="EURUSD strategy",
        method=SearchMethod.FTS,
        limit=3,
    )
    for r in fts_results:
        print(f"  [{r.score:.3f}] {r.content}")
    
    # Hybrid search
    print("\nHybrid search:")
    hybrid_results = await search.search(
        query_text="EURUSD",
        query_embedding=embeddings[0],
        method=SearchMethod.HYBRID,
        limit=3,
    )
    for r in hybrid_results:
        print(f"  [{r.score:.3f}] {r.content}")
    
    # MMR for diversity
    print("\nMMR search (diverse results):")
    mmr_results = await search.search(
        query_embedding=embeddings[0],
        method=SearchMethod.MMR,
        limit=3,
        lambda_mult=0.5,
    )
    for r in mmr_results:
        print(f"  [{r.score:.3f}] {r.content}")
    
    await manager.close()


async def main():
    """Run all examples."""
    print("QuantMindX Memory System - Usage Examples")
    print("=" * 50)
    
    await example_basic_memory()
    await example_with_embeddings()
    await example_session_memory()
    await example_temporal_decay()
    await example_hybrid_search()
    
    print("\n" + "=" * 50)
    print("Examples complete!")


if __name__ == "__main__":
    asyncio.run(main())

# QuantMindX Memory System

A comprehensive Python memory management system inspired by the openclaw architecture, featuring SQLite persistence, vector embeddings, and temporal decay for intelligent memory relevance.

## Features

- **SQLite Backend**: Persistent storage with aiosqlite for async operations
- **Vector Search**: Similarity search using sqlite-vec
- **Multiple Embedding Providers**: OpenAI, Z.AI, local sentence-transformers, and mock
- **Full-Text Search**: FTS5 support for text-based queries
- **Hybrid Search**: Combine vector and full-text search
- **MMR (Maximal Marginal Relevance)**: Diverse, non-redundant results
- **Temporal Decay**: Time-based relevance scoring
- **Session Memory**: Conversation tracking with persistence
- **Dirty Tracking**: Sync operations for incremental updates

## Installation

```bash
# Install dependencies
pip install -r src/memory/requirements.txt

# For local embeddings (optional)
pip install sentence-transformers

# For vector search (optional)
pip install sqlite-vec
```

## Quick Start

```python
import asyncio
from pathlib import Path
from src.memory import (
    create_memory_manager,
    MemorySource,
    get_embedding_provider,
    SearchManager,
    SearchMethod,
)

async def main():
    # Create memory manager
    manager = await create_memory_manager(
        db_path=Path("data/memory.db"),
        embedding_dim=1536,
    )
    
    # Create embedding provider
    provider = get_embedding_provider("openai", model="text-embedding-3-small")
    
    # Add memory with embedding
    text = "EURUSD shows mean reversion behavior"
    embedding = await provider.embed(text)
    
    await manager.add_memory(
        source=MemorySource.MEMORY,
        content=text,
        embedding=embedding,
        importance=0.8,
        tags=["trading", "forex"],
    )
    
    # Search by similarity
    search = SearchManager(manager)
    results = await search.search(
        query_embedding=embedding,
        method=SearchMethod.VECTOR,
        limit=10,
    )
    
    for result in results:
        print(f"[{result.score:.3f}] {result.content}")

asyncio.run(main())
```

## Components

### MemoryManager

Core memory manager with SQLite persistence and vector search.

```python
from src.memory import MemoryManager, MemorySource

manager = MemoryManager(
    db_path=Path("memory.db"),
    embedding_dim=1536,
)
await manager.initialize()

# Add memory
entry = await manager.add_memory(
    source=MemorySource.MEMORY,
    content="Memory content",
    embedding=[...],  # Optional
    importance=0.8,
    tags=["tag1", "tag2"],
)

# Retrieve memory
entry = await manager.get_memory(entry_id)

# Update memory
await manager.update_memory(entry_id, content="Updated content")

# Delete memory
await manager.delete_memory(entry_id)

# Get statistics
stats = await manager.get_stats()
```

### Embedding Providers

Multiple embedding providers with unified interface.

```python
from src.memory import get_embedding_provider

# OpenAI
openai_provider = get_embedding_provider(
    "openai",
    model="text-embedding-3-small",
    api_key="your-api-key",
)

# Z.AI (Anthropic-compatible)
zai_provider = get_embedding_provider(
    "zai",
    model="text-embedding-3-small",
    api_key="your-api-key",
)

# Local sentence-transformers
local_provider = get_embedding_provider(
    "local",
    model="all-MiniLM-L6-v2",  # 384 dimensions
)

# Mock for testing
mock_provider = get_embedding_provider("mock", dimension=1536)

# Generate embeddings
embedding = await provider.embed("Single text")
embeddings = await provider.embed_batch(["Text 1", "Text 2", "Text 3"])
```

### SearchManager

Advanced search with multiple methods.

```python
from src.memory import SearchManager, SearchMethod

search = SearchManager(manager)

# Vector similarity search
results = await search.search(
    query_embedding=[...],
    method=SearchMethod.VECTOR,
    limit=10,
)

# Full-text search
results = await search.search(
    query_text="search query",
    method=SearchMethod.FTS,
    limit=10,
)

# Hybrid search (combined)
results = await search.search(
    query_text="search query",
    query_embedding=[...],
    method=SearchMethod.HYBRID,
    limit=10,
    vector_weight=0.7,
    fts_weight=0.3,
)

# MMR for diverse results
results = await search.search(
    query_embedding=[...],
    method=SearchMethod.MMR,
    limit=10,
    lambda_mult=0.5,  # 0=diversity, 1=relevance
    fetch_k=50,  # Candidate pool size
)
```

### SessionMemory

Conversation/session tracking with persistence.

```python
from src.memory import SessionMemory, MessageRole

session = SessionMemory(
    session_id="chat_001",
    persist_path=Path("sessions/chat_001.json"),
)

# Add messages
session.add_message(
    role=MessageRole.USER,
    content="Hello!",
)

session.add_message(
    role=MessageRole.ASSISTANT,
    content="Hi there!",
)

# Get conversation history
messages = session.get_messages()
last_user_msg = session.get_last_message(role=MessageRole.USER)

# Export to OpenAI format
openai_msgs = session.to_openai_format()

# Persist to disk
await session.persist()
```

### TemporalDecay

Time-based relevance scoring.

```python
from src.memory import TemporalDecay, DECAY_DEFAULT
from datetime import datetime, timezone, timedelta

decay = TemporalDecay(config=DECAY_DEFAULT)

# Calculate decay factor
factor = decay.calculate_decay_factor(
    created_at=datetime.now(timezone.utc) - timedelta(hours=12),
)

# Apply decay to importance score
decayed_score = decay.apply_decay(
    importance=0.8,
    created_at=datetime.now(timezone.utc) - timedelta(hours=12),
)

# Batch processing
scores = decay.calculate_batch_decay(
    importances=[0.8, 0.7, 0.6],
    created_at_times=[...],
)
```

## Preset Decay Configurations

```python
from src.memory import (
    DECAY_FAST,     # 6 hour half-life
    DECAY_DEFAULT,  # 24 hour half-life
    DECAY_SLOW,     # 1 week half-life
    DECAY_VERY_SLOW,  # 30 day half-life
    DecayConfig,
    DecayType,
)

# Custom config
custom_config = DecayConfig(
    decay_type=DecayType.EXPONENTIAL,
    half_life=timedelta(hours=12),
    boost_recent=timedelta(hours=2),
    boost_factor=1.5,
)
```

## File Structure

```
src/memory/
├── __init__.py              # Package exports
├── memory_manager.py        # Core memory manager
├── embeddings.py            # Embedding providers
├── search_manager.py        # Search functionality
├── session_memory.py        # Session tracking
├── temporal_decay.py        # Time-based decay
├── langmem_integration.py   # Legacy LangMem integration
├── langmem_manager.py       # LangMem manager
├── requirements.txt         # Dependencies
├── example_usage.py         # Usage examples
└── README.md               # This file
```

## API Reference

### MemoryManager

| Method | Description |
|--------|-------------|
| `add_memory()` | Add new memory entry |
| `get_memory()` | Retrieve by ID |
| `update_memory()` | Update existing entry |
| `delete_memory()` | Delete entry |
| `search_similar()` | Vector similarity search |
| `search_fts()` | Full-text search |
| `get_dirty_entries()` | Get pending sync entries |
| `get_stats()` | Memory statistics |

### EmbeddingProvider

| Provider | Model | Dimension |
|----------|-------|-----------|
| OpenAI | text-embedding-3-small | 1536 |
| OpenAI | text-embedding-3-large | 3072 |
| Z.AI | text-embedding-3-small | 1536 |
| Local | all-MiniLM-L6-v2 | 384 |
| Local | all-mpnet-base-v2 | 768 |

### SearchMethod

| Method | Description |
|--------|-------------|
| `VECTOR` | Vector similarity search |
| `FTS` | Full-text search |
| `HYBRID` | Combined vector + FTS |
| `MMR` | Maximal Marginal Relevance |

## License

MIT License - See LICENSE file for details.

## Sources

- [sqlite-vec Documentation](https://github.com/asg017/sqlite-vec) - Vector search for SQLite
- [sentence-transformers](https://www.sbert.net/) - Local embedding models
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings) - OpenAI embeddings

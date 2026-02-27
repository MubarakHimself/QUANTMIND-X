# QuantMindX Memory System - Architecture

## Overview

This memory system is inspired by the openclaw architecture and provides a comprehensive solution for managing memories with vector embeddings, temporal decay, and session tracking.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Memory System                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Memory     │  │   Search     │  │   Session    │          │
│  │   Manager    │  │   Manager    │  │   Memory     │          │
│  │              │  │              │  │              │          │
│  │ - CRUD       │  │ - Vector     │  │ - Messages   │          │
│  │ - Sync       │  │ - FTS        │  │ - Persist    │          │
│  │ - Stats      │  │ - Hybrid     │  │ - Delta      │          │
│  │              │  │ - MMR        │  │              │          │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘          │
│         │                  │                                       │
│         └──────────────────┴──────┐                              │
│                                    ▼                              │
│                     ┌──────────────────┐                          │
│                     │  SQLite Backend  │                          │
│                     │                  │                          │
│                     │ - memories table │                          │
│                     │ - FTS5 index     │                          │
│                     │ - vec0 index     │                          │
│                     └──────────────────┘                          │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Embedding Providers                          │   │
│  │                                                           │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │   │
│  │  │  OpenAI  │  │   Z.AI   │  │  Local   │  │   Mock   │ │   │
│  │  │          │  │          │  │          │  │          │ │   │
│  │  │ - GPT-3  │  │ - Claude  │  │ - BERT   │  │ - Test   │ │   │
│  │  │ - Ada    │  │ - Compatible│ │ - SBERT  │  │          │ │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Temporal Decay                              │   │
│  │                                                           │   │
│  │  - Exponential decay function                            │   │
│  │  - Configurable half-life                                │   │
│  │  - Recent memory boost                                   │   │
│  │  - Batch processing                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. MemoryManager

**Purpose**: Core CRUD operations with SQLite persistence

**Key Features**:
- Async operations via aiosqlite
- Vector storage via sqlite-vec
- FTS5 full-text search
- Dirty tracking for sync
- Automatic schema creation

**Database Schema**:
```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,           -- 'memory' or 'session'
    content TEXT NOT NULL,
    embedding BLOB,                 -- Serialized float array
    metadata TEXT,                  -- JSON
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    accessed_at TEXT NOT NULL,
    access_count INTEGER DEFAULT 0,
    importance REAL DEFAULT 0.5,
    tags TEXT,                      -- JSON array
    sync_status TEXT DEFAULT 'clean',
    checksum TEXT
);

CREATE VIRTUAL TABLE memories_fts USING fts5(content, id);
CREATE VIRTUAL TABLE memories_vec USING vec0(embedding float[1536], id);
```

### 2. Embedding Providers

**Purpose**: Generate vector embeddings for semantic search

**Provider Interface**:
```python
class EmbeddingProvider(ABC):
    async def embed(text: str) -> List[float]
    async def embed_batch(texts: List[str]) -> List[List[float]]
    def get_dimension() -> int
    def get_model_name() -> str
```

**Available Providers**:

| Provider | Model | Dimension | Latency | Cost |
|----------|-------|-----------|---------|------|
| OpenAI | text-embedding-3-small | 1536 | ~500ms | $0.02/1M tokens |
| OpenAI | text-embedding-3-large | 3072 | ~1s | $0.13/1M tokens |
| Z.AI | text-embedding-3-small | 1536 | ~500ms | Variable |
| Local | all-MiniLM-L6-v2 | 384 | ~50ms | Free |
| Local | all-mpnet-base-v2 | 768 | ~100ms | Free |
| Mock | - | Configurable | <1ms | Free |

### 3. SearchManager

**Purpose**: Advanced search with multiple strategies

**Search Methods**:

1. **Vector Search**: Cosine similarity using sqlite-vec
2. **FTS Search**: Full-text search using FTS5
3. **Hybrid Search**: Weighted combination of vector + FTS
4. **MMR Search**: Maximal Marginal Relevance for diversity

**MMR Formula**:
```
MMR = argmax [λ * Sim(d, q) - (1 - λ) * max Sim(d, d_selected)]
```

Where:
- `d` = candidate document
- `q` = query
- `d_selected` = already selected documents
- `λ` = balance parameter (0=diversity, 1=relevance)

### 4. SessionMemory

**Purpose**: Track conversation history with persistence

**Message Roles**:
- `system`: System instructions
- `user`: User messages
- `assistant`: Assistant responses
- `tool`: Tool call results

**Features**:
- Delta tracking for incremental updates
- Auto-compression for long sessions
- Token counting for context management
- OpenAI format export

### 5. TemporalDecay

**Purpose**: Time-based relevance scoring

**Decay Types**:

1. **Exponential**: `exp(-λ * t)` - Standard decay
2. **Linear**: `max(0, 1 - t/half_life)` - Linear falloff
3. **Logarithmic**: `1 / (1 + log(1 + t))` - Slow decay
4. **Step**: `1 / (2 ^ steps)` - Discrete steps

**Preset Configurations**:

| Config | Half-Life | Recent Boost | Use Case |
|--------|-----------|--------------|----------|
| DECAY_FAST | 6 hours | 1 hour | Volatile data |
| DECAY_DEFAULT | 24 hours | 2 hours | General use |
| DECAY_SLOW | 1 week | 12 hours | Stable data |
| DECAY_VERY_SLOW | 30 days | 24 hours | Long-term |

## Data Flow

### Adding a Memory

```
User Request
    │
    ▼
Generate Embedding (via Provider)
    │
    ▼
Create MemoryEntry
    │
    ▼
Serialize to SQLite
    │
    ├─► Main Table (memories)
    ├─► FTS Index (memories_fts)
    └─► Vec Index (memories_vec)
    │
    ▼
Return MemoryEntry
```

### Searching Memories

```
Search Request
    │
    ▼
┌──────────────────┐
│  Search Method?  │
└────┬───────┬─────┘
     │       │
     ▼       ▼
┌─────┴─────┐ ┌────────┐
│  Vector   │ │  FTS   │
│  Search   │ │ Search │
└─────┬─────┘ └───┬────┘
      │           │
      └─────┬─────┘
            ▼
      ┌──────────┐
      │  Hybrid  │
      │  Combine │
      └─────┬────┘
            │
            ▼
      ┌──────────┐
      │   MMR    │
      │ Diversify│
      └─────┬────┘
            ▼
      Rank & Return
```

## Performance Considerations

### Vector Search
- **Index**: sqlite-vec provides approximate nearest neighbor
- **Latency**: O(log n) for indexed search
- **Memory**: 4 bytes per dimension per vector

### Full-Text Search
- **Index**: FTS5 with Porter stemming
- **Latency**: O(log n) for indexed search
- **Memory**: Minimal (tokenized text)

### Hybrid Search
- **Combination**: Weighted average of scores
- **Optimization**: Fetch from both indices in parallel
- **Memory**: Same as individual methods

### MMR Search
- **Complexity**: O(k * n) where k=results, n=candidates
- **Optimization**: Use numpy for vector operations
- **Memory**: O(n * d) for candidate embeddings

## Usage Patterns

### Pattern 1: Short-Term Working Memory

```python
# Use fast decay for volatile data
decay = TemporalDecay(config=DECAY_FAST)
manager = await create_memory_manager()
```

### Pattern 2: Long-Term Knowledge Base

```python
# Use slow decay for stable knowledge
decay = TemporalDecay(config=DECAY_SLOW)
# Store with high importance
await manager.add_memory(..., importance=0.9)
```

### Pattern 3: Conversational Context

```python
# Use session memory for conversations
session = SessionMemory(session_id="chat_001")
session.add_message(role=MessageRole.USER, content="...")
```

## Extension Points

### Custom Embedding Provider

```python
class MyEmbeddingProvider(EmbeddingProvider):
    async def embed(self, text: str) -> List[float]:
        # Your implementation
        pass
```

### Custom Decay Function

```python
class CustomDecay(TemporalDecay):
    def _custom_decay(self, elapsed: float) -> float:
        # Your decay formula
        return 1.0 / (1.0 + elapsed ** 0.5)
```

### Custom Search Strategy

```python
class CustomSearchManager(SearchManager):
    async def _search_custom(self, ...):
        # Your search logic
        pass
```

## References

- [sqlite-vec](https://github.com/asg017/sqlite-vec) - Vector search for SQLite
- [FTS5 Documentation](https://www.sqlite.org/fts5.html) - Full-text search
- [sentence-transformers](https://www.sbert.net/) - Local embeddings
- [MMR Paper](https://dl.acm.org/doi/10.1145/276646.276788) - Carbonell & Goldstein, 1998

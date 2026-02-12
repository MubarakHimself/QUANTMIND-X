# Market Data Integration Discussion - QUANTMINDX

**Date:** February 10, 2026  
**Purpose:** Architectural discussion on market data sources, latency, providers, and strategy router integration

---

## 1. Current Market Data Architecture

### 1.1 Data Layer Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUANTMINDX DATA LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐ │
│  │ BrokerRegistry  │  │ DataManager     │  │ StreamClient  │ │
│  │ (YAML Config)   │  │                  │  │               │ │
│  └────────┬────────┘  └────────┬────────┘  └───────┬───────┘ │
│           │                     │                    │         │
│           ▼                     ▼                    ▼         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Broker Adapters                       │   │
│  ├──────────────┬──────────────┬──────────────────────────┤   │
│  │ MT5Socket     │ BinanceSpot  │ BinanceFutures           │   │
│  │ (ZMQ Socket)  │ (REST/WS)    │ (REST/WS)                │   │
│  │ <5ms latency  │ ~50-100ms    │ ~50-100ms                │   │
│  └──────────────┴──────────────┴──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Current Components

| Component | Location | Purpose |
|-----------|----------|---------|
| BrokerRegistry | [`src/data/brokers/registry.py`](src/data/brokers/registry.py) | Factory pattern for broker connections |
| MT5SocketAdapter | [`src/data/brokers/mt5_socket_adapter.py`](src/data/brokers/mt5_socket_adapter.py) | ZMQ socket bridge to MT5 VPS |
| BinanceSpotAdapter | [`src/data/brokers/binance_adapter.py`](src/data/brokers/binance_adapter.py) | Binance Spot trading |
| BinanceFuturesAdapter | [`src/data/brokers/binance_adapter.py`](src/data/brokers/binance_adapter.py) | Binance Futures trading |
| TickStreamer | [`mcp-metatrader5-server/src/mcp_mt5/streaming.py`](mcp-metatrader5-server/src/mcp_mt5/streaming.py) | WebSocket real-time tick streaming |

### 1.3 Strategy Router Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                  STRATEGY ROUTER                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐ │
│  │   Sentinel  │──▶│   Governor   │──▶│     Commander       │ │
│  │ (Market Diag)│   │ (Risk Mgmt) │   │ (Strategy Auction)  │ │
│  └─────────────┘   └─────────────┘   └─────────────────────┘ │
│         │                │                    │               │
│         ▼                ▼                    ▼               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Market Data Input                          │   │
│  │  process_tick(symbol, price, account_data)             │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Market Data Sources & Providers

### 2.1 Current Providers

| Provider | Asset Classes | Data Types | Latency | Pros | Cons |
|----------|--------------|------------|---------|------|------|
| **MT5** | Forex, CFDs, Commodities, Indices | OHLCV, Tick, Order Book | <5ms (socket) | Direct broker feed, native execution | Windows required, limited symbols |
| **Binance** | Crypto Spot/Futures | OHLCV, Tick, Depth | ~50-100ms | 24/7 trading, high liquidity | REST/WebSocket overhead |

### 2.2 Alternative Providers to Consider

| Provider | Type | Use Case | Latency | Notes |
|----------|------|----------|---------|-------|
| **Interactive Brokers** | Direct | Global equities, forex | ~10-50ms | Extensive market coverage |
| **Alpaca** | API | US equities, crypto | ~20-100ms | Developer-friendly, paper trading |
| **Polygon.io** | API | US market data | ~10-50ms | WebSocket streaming available |
| **Binance Direct** | FIX | Crypto derivatives | ~5-20ms | Lower latency than REST |
| **Exante** | FIX | Multi-asset | ~5-15ms | Institutional-grade |
| **CQG** | Propietary | Futures, forex | <5ms | Ultra-low latency |

### 2.3 Provider Selection Criteria

```
┌────────────────────────────────────────────────────────────┐
│              PROVIDER EVALUATION MATRIX                     │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. LATENCY REQUIREMENTS                                   │
│     □ Sub-millisecond: Direct exchange, co-location       │
│     □ 1-5ms: FIX protocol, binary sockets                  │
│     □ 10-50ms: WebSocket, optimized REST                   │
│     □ 100ms+: Standard API, polling                       │
│                                                            │
│  2. MARKET COVERAGE                                        │
│     □ Forex: MT5, Interactive Brokers, CQG                 │
│     □ Crypto: Binance, Coinbase Pro, Kraken               │
│     □ Equities: Alpaca, Polygon.io, IBKR                  │
│     □ Futures: CQG, Rithmic, TT                            │
│                                                            │
│  3. DATA QUALITY                                           │
│     □ Tick-level: Raw exchange data                       │
│     □ Level 2: Order book depth                           │
│     □ Level 3: Full order flow                            │
│                                                            │
│  4. COST CONSIDERATIONS                                    │
│     □ Free tier: Polygon, Alpaca (limited)                 │
│     □ Subscription: $50-500/month for real-time            │
│     □ Enterprise: Exchange data fees apply                │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 3. Latency Analysis

### 3.1 Current Latency Profile

```
┌─────────────────────────────────────────────────────────────────┐
│                    LATENCY STACK (Current)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Exchange ──▶ Broker ──▶ MT5 Terminal ──▶ ZMQ Bridge ──▶ App  │
│                │            ~1-2ms          <5ms         │     │
│               ~1ms          (Windows)                      ▼     │
│                                                   Python Router │
│                                                          ~1ms   │
│                                                                 │
│  Total: ~3-8ms (MT5)                                          │
│                                                                 │
│  Exchange ──▶ Binance API ──▶ REST/WS ──▶ Python Adapter      │
│               ~50-100ms              │                     │    │
│                                     ~20ms                  ▼    │
│                                                   Python Router │
│                                                          ~1ms   │
│                                                                 │
│  Total: ~70-120ms (Binance)                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Latency Optimization Options

| Layer | Optimization | Latency Reduction | Complexity |
|-------|--------------|-------------------|------------|
| **Network** | Co-location | -2-5ms | High |
| **Protocol** | FIX 4.4/5.0 SP2 | -1-3ms | Medium |
| **Serialization** | Protocol Buffers | -0.5-1ms | Low |
| **Transport** | UDP Multicast | -1-2ms | Medium |
| **Data Format** | Binary (FlatBuffers) | -0.2-0.5ms | Low |
| **Processing** | Zero-copy deserialization | -0.1-0.3ms | Medium |

### 3.3 Latency Budget Example

```
┌─────────────────────────────────────────────────────────────────┐
│              LATENCY BUDGET (HFT Strategy - 10ms)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Network transit          ████████████░░░░░░░░░░░░░   4ms       │
│  Exchange processing      ██████░░░░░░░░░░░░░░░░░░░   2ms       │
│  Broker internal          ████░░░░░░░░░░░░░░░░░░░░░   1ms       │
│  Data serialization       ██░░░░░░░░░░░░░░░░░░░░░░░   0.5ms     │
│  Router processing        ██░░░░░░░░░░░░░░░░░░░░░░░   0.5ms     │
│  Strategy computation     ████████░░░░░░░░░░░░░░░░░░   2ms       │
│  ─────────────────────────────────────────────────────          │
│  Total                    10ms                                 │
│                                                                 │
│  Margin                   ██░░░░░░░░░░░░░░░░░░░░░░░   TBD       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Connection Methods

### 4.1 Protocol Comparison

| Protocol | Type | Latency | Reliability | Use Case |
|----------|------|---------|-------------|----------|
| **REST** | Synchronous | 50-200ms | Medium | Historical data, order mgmt |
| **WebSocket** | Async | 20-100ms | High | Real-time streaming |
| **ZMQ** | Message Queue | 1-10ms | High | Inter-process comm |
| **FIX 4.2** | Protocol | 5-20ms | Very High | Trading, institutional |
| **FIX 4.4** | Protocol | 3-15ms | Very High | Enhanced FIX |
| **FIX 5.0 SP2** | Protocol | 2-10ms | Very High | Latest FIX standard |
| **Binary (Protobuf)** | Serialization | 1-5ms | High | Low-latency data |
| **UDP** | Transport | <1ms | Low | Market data only |

### 4.2 Connection Architecture Patterns

#### Pattern A: Direct Broker Connection (Current MT5)
```
┌──────────┐    ┌──────────┐    ┌────────────┐
│  MT5     │───▶│  ZMQ     │───▶│  Python    │
│ Terminal │    │  Bridge  │    │  Router    │
└──────────┘    └──────────┘    └────────────┘
                 (Windows)
```
- **Pros:** Native broker integration, low latency
- **Cons:** Platform-dependent (Windows)

#### Pattern B: FIX Protocol Bridge
```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────────┐
│ Exchange │───▶│  FIX     │───▶│  FIX     │───▶│  Python    │
│ / Broker │    │  Gateway │    │  Engine  │    │  Router    │
└──────────┘    └──────────┘    └──────────┘    └────────────┘
```
- **Pros:** Standardized, multi-broker, platform independent
- **Cons:** More complex setup

#### Pattern C: Aggregated Data Feed
```
┌──────────┐    ┌──────────┐
│  MT5     │    │ Binance  │
└────┬─────┘    └────┬─────┘
     │               │
     ▼               ▼
┌────────────────────────────────────┐
│        Data Aggregator             │
│  (Normalization, deduplication)    │
└──────────────┬─────────────────────┘
               │
               ▼
        ┌────────────┐
        │  Python    │
        │  Router    │
        └────────────┘
```
- **Pros:** Multi-source validation, failover
- **Cons:** Additional latency overhead

---

## 5. Strategy Router Data Integration

### 5.1 Current Data Flow

```
Tick Data → process_tick() → Sentinel → Governor → Commander
            (symbol, price)
```

### 5.2 Enhanced Data Integration Options

| Data Type | Current | Enhancement Option | Use Case |
|-----------|---------|-------------------|----------|
| **Price** | ✓ Tick | Add timestamp, spread, volume | Basic trading |
| **OHLCV** | ✗ | Add bar aggregation | Technical analysis |
| **Order Book** | ✗ | Add bid/ask depth, liquidity | Market making |
| **Volume Profile** | ✗ | Add volume at price | Volume analysis |
| **Sentiment** | ✗ | Add news/scores | Regime detection |
| **Order Flow** | ✗ | Add delta, aggression | Microstructure |
| **Correlations** | ✗ | Add cross-asset data | Portfolio routing |

### 5.3 Proposed Enhanced Data Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENHANCED DATA PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐   │
│  │  Price  │  │  Order  │  │ Volume  │  │ Alternative     │   │
│  │  Stream │  │  Book   │  │ Profile │  │ Data            │   │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────────┬────────┘   │
│       │            │            │                 │             │
│       ▼            ▼            ▼                 ▼             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Market Data Normalizer                      │   │
│  │  (Format conversion, validation, timestamp sync)        │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Data Enrichment Layer                        │   │
│  │  (Indicators, regime flags, quality scores)              │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Strategy Router                             │   │
│  │  Sentinel → Governor → Commander                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Discussion Questions

### 6.1 Immediate Priorities

1. **What are your latency requirements?**
   - Sub-5ms: Requires FIX + co-location
   - 5-20ms: Binary protocols, optimized sockets
   - 50ms+: Current architecture sufficient

2. **Which markets are you focusing on?**
   - Forex: MT5, CQG, Interactive Brokers
   - Crypto: Binance Direct, Coinbase Pro
   - Equities: Alpaca, Polygon.io, IBKR

3. **What data types do you need?**
   - ✓ Tick data (current)
   - ☐ Order book depth
   - ☐ Volume profile
   - ☐ Alternative data

### 6.2 Architecture Decisions

4. **Single-source vs. Multi-source?**
   - Single: Simpler, lower latency, single point of failure
   - Multi: Redundancy, validation, higher latency

5. **Protocol preference?**
   - ZMQ: Current, good for internal routing
   - FIX: Industry standard, multi-broker
   - WebSocket: Good for web integration
   - Binary: Best performance

6. **Integration with strategy router?**
   - Current: Price + basic metadata
   - Enhanced: Full market depth + alternative data

---

## 7. Action Items & Next Steps

### For Discussion

- [ ] Confirm latency requirements and use cases
- [ ] Select target markets and data sources
- [ ] Choose connection protocols
- [ ] Define data requirements for strategy router

### Potential Enhancements

| Priority | Enhancement | Effort | Impact |
|----------|-------------|--------|--------|
| 1 | FIX Protocol Adapter | Medium | High |
| 2 | Multi-Provider Aggregation | Medium | High |
| 3 | Order Book Integration | Low | Medium |
| 4 | Binary Serialization | Low | Medium |
| 5 | Alternative Data Feed | High | High |

---

*This document is a discussion framework. Please provide feedback on areas you'd like to explore further.*

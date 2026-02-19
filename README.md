# 🌌 QUANTMIND-X: The Quant Engineering Factory

> An autonomous, physics-aware proprietary trading ecosystem that bridges neural pattern recognition and industrial-grade high-frequency trading through a tri-layer sentient architecture.

**⚠️ Development Status:** Active Development - Not Production Ready

---

## 🏗️ Architecture Overview

QuantMind-X operates as a dual-hemisphere system:

### 🧠 Intelligence Hub (The Brain)
A multi-agent system powered by **OpenRouter** and **LangGraph** that handles the end-to-end strategy lifecycle.

```
┌─────────────────────────────────────────────────────────────┐
│                    INTELLIGENCE HUB                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Analyst   │───▶│  QuantCode   │───▶│ QuantMind    │ │
│  │   Agent     │    │    Agent     │    │ Co-pilot     │ │
│  │             │    │              │    │              │ │
│  │ • NPRD      │    │ • MQL5/Py    │    │ • Orchestrate│ │
│  │ • TRD       │    │ • Compile    │    │ • Queue Mgmt │ │
│  │ • Synthesis │    │ • Refine     │    │ • Handoffs   │ │
│  └─────────────┘    └──────────────┘    └──────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 🛡️ Strategy Router (The Body)
High-performance execution floor managing real-time risk and trade dispatch.

```
┌─────────────────────────────────────────────────────────────┐
│                    STRATEGY ROUTER                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Sentinel  │───▶│   Governor   │───▶│  Commander   │ │
│  │             │    │              │    │              │ │
│  │ • Chaos     │    │ • Risk       │    │ • Auction    │ │
│  │ • Lyapunov  │    │ • Compliance │    │ • Dispatch   │ │
│  │ • Regime    │    │ • Caps       │    │ • Selection  │ │
│  └─────────────┘    └──────────────┘    └──────────────┘ │
│          │                   │                   │       │
│          ▼                   ▼                   ▼       │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │ Market Data │    │ Risk Engine  │    │ Bot Registry │ │
│  │   Streams   │    │              │    │              │ │
│  └─────────────┘    └──────────────┘    └──────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🌀 Risk Management System

### Physics-Aware Risk Detection

```
┌─────────────────────────────────────────────────────────────┐
│                  CHAOS DETECTION SYSTEM                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Market Data ──▶ Lyapunov Calculator ──▶ Chaos Score        │
│      │                │                    │               │
│      ▼                ▼                    ▼               │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │ Price   │    │ Divergence  │    │ Risk Floor  │        │
│  │ Series  │    │ Rate        │    │ 0.2x Scale  │        │
│  │         │    │             │    │             │        │
│  │ • OHLC  │    │ • λ > 0     │    │ • Auto      │        │
│  │ • Ticks │    │ • Chaos     │    │ • Protect   │        │
│  │ • Volume│    │ • Predict   │    │ • Capital    │        │
│  └─────────┘    └─────────────┘    └─────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3-Layer Enhanced Kelly Position Sizing

```
┌─────────────────────────────────────────────────────────────┐
│                ENHANCED KELLY SYSTEM                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Win Rate, Avg Win/Loss, Volatility                 │
│          │                                                 │
│          ▼                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              LAYER 1: Kelly Fractioning             │   │
│  │  • Full Kelly: f* = p - q/b                        │   │
│  │  • Fractional: 0.25-0.5 f*                        │   │
│  │  • Growth without ruin                            │   │
│  └─────────────────────────────────────────────────────┘   │
│          │                                                 │
│          ▼                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              LAYER 2: Hard Risk Caps                 │   │
│  │  • Max 2% per trade                                │   │
│  │  • Daily loss limits                               │   │
│  │  │  • Circuit breakers                             │   │
│  └─────────────────────────────────────────────────────┘   │
│          │                                                 │
│          ▼                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           LAYER 3: Dynamic Volatility Adj          │   │
│  │  • ATR-based scaling                               │   │
│  │  • Real-time volatility adjustment                 │   │
│  │  • Market regime awareness                         │   │
│  └─────────────────────────────────────────────────────┘   │
│          │                                                 │
│          ▼                                                 │
│  Output: Final Position Size                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔌 ZeroMQ Bridge Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ZERO-MQ BRIDGE                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Python Strategy Engine          MetaTrader 5 Terminal     │
│  ┌─────────────────┐            ┌─────────────────┐         │
│  │  ZMQ Publisher  │◀───5ms───▶│  ZMQ Subscriber │         │
│  │                 │            │                 │         │
│  │ • Trade Signals │            │ • Execute       │         │
│  │ • Risk Updates  │            │ • Market Data   │         │
│  │ • Bot Status    │            │ • Position Mgmt │         │
│  └─────────────────┘            └─────────────────┘         │
│          │                                 │                │
│          ▼                                 ▼                │
│  ┌─────────────────┐            ┌─────────────────┐         │
│  │  Message Queue  │            │  MT5 Gateway    │         │
│  │                 │            │                 │         │
│  │ • Async I/O     │            │ • Native API    │         │
│  │ • Reliable      │            │ • Low Latency   │         │
│  │ • Ordered       │            │ • Real-time     │         │
│  └─────────────────┘            └─────────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Development Setup

### Prerequisites

- Python 3.9+
- Node.js 18+
- MetaTrader 5 Terminal
- OpenRouter API Key

### Backend Setup

```bash
# Clone repository
git clone https://github.com/yourusername/quantmind-x.git
cd quantmind-x

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your OpenRouter API key
```

### Frontend IDE Setup

```bash
# Navigate to IDE directory
cd quantmind-ide

# Install dependencies
npm install

# Start development server
npm run dev
```

### Running Services

```bash
# Terminal 1: Start API server
python3 -m src.api.server

# Terminal 2: Start strategy router
python3 -m src.router.commander

# Terminal 3: Start IDE (in quantmind-ide directory)
npm run dev
```

### Starting PageIndex Services

PageIndex provides knowledge retrieval services for articles, books, and trading logs.

```bash
# Start PageIndex services (articles, books, logs)
docker-compose -f docker-compose.pageindex.yml up -d

# Services run on:
# - Articles API: http://localhost:3000
# - Books API: http://localhost:3001
# - Logs API: http://localhost:3002
```

### Starting Monitoring Stack (Production)

For production monitoring with Grafana, Prometheus, and Loki:

```bash
# Start full monitoring stack
docker-compose -f docker-compose.production.yml up -d

# Access Grafana at your configured instance URL
```

---

## 📊 Core Technologies

### Backend Stack
- **Python 3.9+** - Core language
- **FastAPI** - REST API framework
- **LangGraph** - Agent orchestration
- **ZeroMQ** - Low-latency messaging
- **PageIndex** - Knowledge retrieval (articles, books, logs)
- **Pydantic** - Data validation

### Frontend Stack
- **SvelteKit** - Web framework
- **Tauri** - Desktop app framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling

### Trading Infrastructure
- **MetaTrader 5** - Trading platform
- **OpenRouter** - LLM API gateway
- **WebSocket** - Real-time communication

---

## 🗺️ Development Roadmap

### ✅ Completed

- [x] **Phase 4**: Strategy Router Core (Sentinel, Governor, Commander)
- [x] **Phase 4.5**: Agent Framework (BaseAgent, Skill Management)
- [x] **Phase 5**: Analyst & QuantCode V1 (Node Graphs & Refinement)
- [x] **Phase 5.5**: IDE Foundation (SvelteKit + Tauri)

### 🚧 In Progress

- [ ] **Phase 6**: QuantMind Co-pilot (Master Orchestrator)
- [ ] **Phase 6.5**: Advanced Risk Analytics
- [ ] **Phase 7**: Enhanced IDE Features (Debugging, Profiling)

### 📋 Planned

- [ ] **Phase 8**: Live Prop Deployment
- [ ] **Phase 9**: Multi-Asset Support (Crypto, Futures)
- [ ] **Phase 10**: Community Strategy Marketplace

---

## 🔍 Current Implementation Status

### Working Components
- ✅ Basic agent communication
- ✅ Strategy router framework
- ✅ Risk management core
- ✅ ZeroMQ bridge to MT5
- ✅ IDE interface foundation
- ✅ WebSocket streaming

### In Development
- 🔄 Advanced chaos detection
- 🔄 Enhanced Kelly implementation
- 🔄 Agent skill system
- 🔄 Backtesting engine
- 🔄 Strategy auction system

### TODO
- ⏳ Production deployment
- ⏳ Performance optimization
- ⏳ Error handling
- ⏳ Logging system
- ⏳ Unit tests coverage

---

## 📚 Documentation

- [API Reference](docs/api/README.md) - REST API documentation
- [Agent System](docs/agents/README.md) - Agent architecture
- [Risk Management](docs/risk/README.md) - Risk system details
- [Strategy Development](docs/strategies/README.md) - Strategy creation guide

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

© 2026 QuantMind Labs. Built for the era of Machine Intelligence.
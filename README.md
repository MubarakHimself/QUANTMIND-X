# 🌌 QUANTMIND-X: The Quant Engineering Factory

QuantMind-X is an autonomous, physics-aware proprietary trading ecosystem. It bridges the gap between neural pattern recognition (NPRD) and industrial-grade high-frequency trading through a tri-layer sentient architecture.

---

## 🏗️ Architecture: The Brain & The Body

QuantMind-X is divided into two primary hemispheres:

### 🧠 1. The Intelligence Hub (The Brain)
A multi-agent system powered by **OpenRouter** and **LangGraph** that handles the end-to-end strategy lifecycle.
- **🕵️‍♂️ Analyst Agent**: Consumes voice-to-text transcripts (NPRD) and synthesizes technical requirement documents (TRDs).
- **🤖 QuantCode Agent**: A specialized engineering agent that uses a **Trial & Reflection** loop to write, compile, and refine MQL5 and Python strategies.
- **🚀 QuantMind Co-pilot**: The master orchestrator. Sits at the highest hierarchical level, managing agent handoffs and global task queuing.

### 🛡️ 2. The Strategy Router (The Body)
A high-performance execution floor that manages risk and trade dispatch in real-time.
- **Sentinel**: Real-time market diagnostics using **Lyapunov-based Chaos Sensors** to detect market instability.
- **Governor**: Enforces compliance and risk stacking (including **Prop Firm Survival** quadratic throttling).
- **Commander**: Runs a **Strategy Auction** to dispatch the best bots for the current market regime.

---

## 🔬 Core Technologies

### 🌀 Physics-Aware Risk
We don't just measure volatility; we measure **Chaos**. The system uses rate-of-divergence (Lyapunov proxies) to identify when market predictability collapses, automatically engaging a 0.2x risk floor.

### 📊 3-Layer Enhanced Kelly
Optimal position sizing that protects capital through:
1.  **Kelly Fractioning**: Capturing growth without the ruin of full Kelly.
2.  **Hard Risk Caps**: Absolute circuit breakers on a per-trade basis.
3.  **Dynamic Volatility Adjustment**: Real-time scaling based on ATR divergence.

### 🔌 Native Bridge (ZMQ)
Low-latency communication layer using ZeroMQ for sub-5ms integration between Python's intelligence and MetaTrader 5's execution.

---

## 🚀 Getting Started

### 1. Environment Configuration
Ensure your `.env` is configured for **OpenRouter**:
```bash
OPENROUTER_API_KEY=your_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

### 2. Interaction
Interact with the specialized agents via the QuantMind CLI:
```bash
# Chat with the Architect
python3 -m src.agents.cli chat --agent analyst

# Chat with the Engineer
python3 -m src.agents.cli chat --agent quantcode
```

### 3. Stress Testing
Verify the router's responsiveness to chaos:
```bash
python3 stress_test_router.py
```

---

## 🗺️ Roadmap: The Path to Sentience

- [x] **Phase 4: Strategy Router Core** (Sentinel, Governor, Commander)
- [x] **Phase 4.5: Agent Framework** (BaseAgent, Skill Management)
- [x] **Phase 5: Analyst & QuantCode V1** (Node Graphs & Refinement)
- [ ] **Phase 6: QuantMind Co-pilot** (The Master Orchestrator)
- [ ] **Phase 7: IDE Integration** (VS Code-style Desktop App)
- [ ] **Phase 8: Live Prop Deployment**

---

© 2026 QuantMind Labs. Built for the era of Machine Intelligence.

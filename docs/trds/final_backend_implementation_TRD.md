# Technical Requirements Document: `myCodingAgents.com` & QuantMindX Architecture

> **Version:** 1.0
> **Target System:** myCodingAgents.com Integration with QuantMindX
> **Purpose:** To provide a deterministic, context-rich specification for the implementation of the `myCodingAgents.com` platform, ensuring all probabilistic models have sufficient context to execute without hallucination.

---

## 1. Executive Summary

This document defines the technical architecture for `myCodingAgents.com`, a platform enabling the deployment of autonomous trading agents within the **QuantMindX** ecosystem. The core requirement is to unify the **Knowledge Library (Assets Hub)**, **Quant Code Agent**, **Backtest Engine**, and **Paper Trading** modules into a seamless, automated workflow where code generation leads directly to verification and deployment.

### Core Philosophy
*   **Context-First:** The system relies on a massive, structured Knowledge Library to ground all AI decisions.
*   **Automated Pipeline:** `Quant Code Agent` -> `Backtest Engine` -> `Paper Trading` must occur without human friction, using MCP for inter-agent communication.
*   **Guild-Based:** Adheres to the 5-Guild stricture of QuantMindX (Research, Engineering, Operations, Evolution, Orchestration).

---

## 2. The Assets Hub (Knowledge Library)

The **Assets Hub** is the central repository for all intellectual property, logic, and context required by the agents. It replaces and expands upon the "Research Guild KB".

### 2.1 Component Structure

The Assets Hub serves five distinct categories of assets:

1.  **Algorithms & EAs (MetaTrader 5)**
    *   **Storage:** `Tier 3 (ChromaDB/Qdrant)` + `Git Storage`
    *   **Format:** `.mq5` source files, `.ex5` compiled binaries (references), and `metadata.json` (logic description).
    *   **Attributes:** Strategy Logic, Indicators Used, Timeframe, Risk Profile.

2.  **Articles & Theory**
    *   **Source:** Scraped MQL5 articles, internal strategy notes, PDF books.
    *   **Structure:** Classified into `ml_strategy`, `indicator_usage`, `market_mechanics`.
    *   **Function:** Provides semantic context for RAG (Retrieval Augmented Generation) calls by the Quant Code Agent.

3.  **Knowledge Bases (Guild-Specific)**
    *   **Research KB:** Raw data, transcripts, market observations.
    *   **Engineering KB:** Code templates, snippets, "What works/What fails" patterns.
    *   **Operations KB:** Trade journals, execution logs, broker quirks.

4.  **Resources**
    *   External API Connector definitions.
    *   Docker container images for specific environments.
    *   Datafeed configurations.

5.  **Agentic Skills Library (NEW)**
    *   **Definition:** Re-usable, atomic capabilities that agents can "equip".
    *   **Examples:** `deploy_vps`, `optimize_genetic_algo`, `parse_financial_statement`, `execute_mt5_order`.
    *   **Implementation:** MCP Tools defined in `server_definitions`.
    *   **Storage:** `docs/skills/` registry with standardized input/output schemas.

### 2.2 Access Layer
*   **MCP Server (`assets-hub-server`):** Exposes a secure API for agents to query and retrieve assets.
    *   `get_strategy_template(type="scalper")`
    *   `search_library(query="RSI + MACD divergence")`
    *   `load_skill(skill_name="backtest_runner")`

---

## 3. Quant Code Agent & Backtest Engine

This section details the critical automation from code generation to verification.

### 3.1 Quant Code Agent ("The Builder")
*   **Role:** Specialized implementation of the `Engineering Guild: Code Developer`.
*   **Responsibility:** specific focus on writing high-performance, error-free MQL5 and Python trading logic.
*   **Context Awareness:**
    *   Must load "Coding Standards" from Assets Hub before writing.
    *   Must checks "Bad Patterns" (Graveyard) to avoid repeating failures.

**Automation Workflow:**
1.  Receives `Strategy Specification` (JSON) from Research Guild or User.
2.  Retrieves `Code Template` from Engineering KB.
3.  Generates Source Code.
4.  **CRITICAL STEP:** Automatically invokes `Backtest Engine` via MCP. *No manual copying.*

### 3.2 Backtest Engine Integration
*   **Reference:** Internal QuantMindX Backtest Engine (based on Backtrader/Custom Python Engine described in `QuantMindX_ Complete Technical Documen.md`).
*   **Upgrade:** An **MCP Server Wrapper** around the existing Backtest Engine.

**MCP Tool Definition:**
```json
{
  "name": "run_backtest",
  "description": "Executes a backtest for a given strategy code and configuration.",
  "inputs": {
    "code_content": "string (The full source code)",
    "language": "string (python | mq5)",
    "config": {
      "symbol": "EURUSD",
      "period": "Start2023-End2024",
      "timeframe": "H1",
      "initial_capital": 10000
    }
  },
  "outputs": {
    "metrics": {
      "sharpe": 1.5,
      "drawdown": 8.4,
      "profit": 1200
    },
    "equity_curve": "[Array...]",
    "logs": "string"
  }
}
```

### 3.3 The Loop
The Quant Code Agent operates in a loop:
1.  **Generate** -> 2. **Send to Backtest MCP** -> 3. **Analyze Result**
    *   *If Result < Threshold:* **Refine Code** (Self-Correction) -> Goto 1.
    *   *If Result > Threshold:* **Tag `@primal`** -> **Dispatch to Paper Trading**.

---

## 4. Paper Trading (Sub-Agents)

Once a strategy passes backtesting, it enters the **Paper Trading** phase.

### 4.1 Deployment Architecture
*   **Sub-Agents:** Lightweight, containerized instances of the strategy spun up within the QuantMind Platform.
*   **Host:** `Operations Guild` Service (Docker Swarm / Kubernetes).

### 4.2 Lifecycle
1.  **Spin Up:** Quant Code Agent triggers `deploy_paper_agent` tool.
2.  **Runtime:** The sub-agent connects to the `MT5 Connector` (Operations Guild) using read-only credentials or a demo account.
3.  **Monitoring:** The sub-agent reports heartbeat and trade events to the **Dashboard (IDE)** via Redis Pub/Sub.
4.  **Promotion:** If the sub-agent survives 50 trades with metrics intact, it requests promotion to Live Trading (`@pending`).

---

## 5. Implementation Roadmap for myCodingAgents.com

To enable this platform, the following implementations are prioritized:

1.  **Assets Hub MCP Server**: Build the API to serve the Knowledge Library and Agentic Skills.
2.  **Quant Code Agent Logic**: Refine the prompt engineering and tool usage of the standard engineering agent to strictly follow the `Generate -> Backtest` loop.
3.  **Backtest Engine Wrapper**: Create the `backtest-mcp-server` that exposes the internal Python backtester logic to the agents.
4.  **Paper Trading Container**: Define the `Dockerfile` for ephemeral strategy agents.

---

## 6. Technical Context for Models

**Probabilistic Model Guiderails:**
To prevent hallucination, all Agents must be initialized with access to the **Assets Hub**. They should never "invent" a library function. They must always:
1.  Query `Assets Hub` for available libraries/snippets.
2.  Read the `Skill Definition`.
3.  Execute the `Skill`.

This specific `Retrieval-First` architecture ensures deterministic behavior in a probabilistic system.

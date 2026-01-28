# Gap Analysis: Backend Implementation vs. Final TRD

> **Date:** 2026-01-28
> **Reference:** `docs/trds/final_backend_implementation_TRD.md`

## 1. Assets Hub (Knowledge Library)
*   **Current State:** `quantmindx-kb-chroma` MCP server exists but is currently designed as a knowledge base for development/reference purposes (scraped articles), not for active agent asset retrieval.
*   **Gap:** Missing structured retrieval for operational assets:
    *   Algorithm Templates (`.mq5` source files).
    *   Agentic Skills (The `load_skill` tool is missing).
    *   Guild-Specific Pattern Libraries (Graveyard, Best Practices).
*   **Impact:** Agents cannot "equip" new skills or reliably load coding standards/templates.

## 2. Quant Code Agent
*   **Current State:** `QuantCodeAgent` (in `src/agents/implementations/quant_code.py`) uses a basic `planner -> coder -> validator` LangGraph.
*   **Gap:**
    *   **Context:** `planning_mode` (planning node) does not fetch "Coding Standards" or "Bad Patterns" from the Assets Hub.
    *   **Validation:** `validation_mode` (validation node) uses a mock check ("logic detected") instead of running a backtest.
    *   **Loop:** No feedback loop from backtest results to code refinement.
*   **Impact:** Agent is "blind" to past failures and cannot verify if its code actually works.

## 3. Backtest Engine Integration
*   **Current State:** No specific backtest engine MCP server found. `quant-traderr-lab` contains research notebooks.
    *   *Reference:* The architectural design for the Backtest Engine is detailed in `/# QuantMindX_ Complete Technical Documen.md`.
*   **Gap:** Missing the `backtest-mcp-server` wrapper (TRD Section 3.2) that exposes the engine described in the technical documentation to the Quant Code Agent.
*   **Impact:** Code generation is theoretical; no way to automatically prove profitability before deployment.

## 4. Paper Trading
*   **Current State:** No infrastructure for deploying containerized sub-agents.
*   **Gap:** Missing `deploy_paper_agent` tool and the associated Docker container definitions.
*   **Impact:** No automated path from code to live-like testing.

## Recommendations
To support the "Context-First" architecture:
1.  **High Priority:** Upgrade the existing `quantmindx-kb` (currently a development knowledge base) to function as the full **Assets Hub**, capable of serving Skills and Templates to agents.
2.  **High Priority:** Build the `backtest-mcp-server` based on the specifications in the technical document.
3.  **Medium Priority:** Update `QuantCodeAgent` to use the above two tools.

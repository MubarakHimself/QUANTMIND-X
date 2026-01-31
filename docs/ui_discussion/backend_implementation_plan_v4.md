# Backend "Bridge the Gap" Implementation Plan (v4 - Comprehensive)

**Objective:** Implement the missing backend assets required to support the "Hybrid Risk" architecture, "Shared Asset Library", and "Agent Capabilities" upgrades.

**Target Audience:** Coding Agent / Backend Engineer / Copilot.

---

## 🤖 Component 1: Agent Capabilities Upgrade

**Goal:** empower all agents with standardized tools and specific meta-skills.

### 1.1 Universal Slash Commands
**Location:** `src/agents/core/base_agent.py`
**Changes:**
*   Implement `CommandParser` middleware.
*   Support standard commands:
    *   `/code [snippet]`: Execute code directly.
    *   `/file [path]`: Read/Write file operations.
    *   `/run [cmd]`: Execute shell commands.
*   **Security:** These must be gated by `SafeToAutoRun` logic or user confirmation.

### 1.2 The "Skill Creator" Meta-Skill (For Copilot)
**Location:** `src/agents/skills/system_skills/skill_creator.py`
**Source:** [Anthropic Skill Standard](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md)
**Role:** Allows the Copilot to scaffold *new* skills for other agents.
**Logic:**
1.  **Scaffold Structure:** Create directory `src/agents/skills/[category]/[name]/`.
2.  **Generate `SKILL.md`:** Write YAML frontmatter + Instructions.
3.  **Generate `__init__.py`:** Create the `SkillDefinition` object (Pydantic) to register it.

---

## 🏗️ Component 2: The Shared Asset Library (`QuantMind_Risk.mqh`)

**Context:** The Data Access Layer (DAO) for risk parameters.

**Location:** `extensions/mql5_library/Include/QuantMind/Risk.mqh`

### 2.1 Core Responsibilities
1.  **Read Global State:** Check `GlobalVariableGet("QM_RISK_MULTIPLIER")`.
2.  **Read Hybrid State (Fallback):** Read `risk_matrix.json`.
3.  **Provide Clean Interface:** `GetRiskMultiplier(symbol)`.

### 2.2 *New* Feature: REST Heartbeat
**Source:** `developing_an_mql5_reinforcement_learning_agent_with_restapi_integration...`
**Logic:**
*   Use `WebRequest("POST", ...)` to send a JSON heartbeat to the Python Server (`http://localhost:8000/heartbeat`).
*   **Why:** Allows the "System Monitor" UI to show which EAs are active *without* complex ZMQ sockets.

---

## 🔄 Component 3: Data & Sync Layer

### 3.1 FireCrawl Sync (Librarian Skill)
**Location:** `src/agents/skills/data_skills/firecrawl_sync.py`
**Role:** Keeps local knowledge base in sync with MQL5.com articles.
**Logic:**
*   **Weekly Job:** Check `https://www.mql5.com/en/articles` for new entries in "Integration" or "Expert Advisors".
*   **Parser:** If part of a series (e.g., "Part 6"), download and tag it.

### 3.2 Hybrid Sync (DiskSyncer)
**Location:** `src/router/sync.py`
**Role:** Writes risk parameters to `risk_matrix.json` for the `@swing` bots.

---

## 🛠️ Component 4: Expert Coding Skills

### 4.1 Indicator Writer (MQL5 Generator)
**Location:** `src/agents/skills/trading_skills/indicator_writer.py`
**Source:** `mql5_cookbook_-_creating_a_ring_buffer...`
**Constraint:** Must enforce **Ring Buffer** pattern (`CRiBuffDbl`) to prevent lag.

### 4.2 Python Backtester Engine
**Location:** `src/backtesting/engine.py`
**Source:** `python-metatrader_5_strategy_tester__part_02...`
**Logic:**
*   Implement `PythonStrategyTester` class.
*   Overload standard MT5 testing functions to run purely in Python with tick-level precision.

---

## 🧪 Verification Steps

1.  **Agent Check:** Run `Copilot` and try `/run ls -la`.
2.  **Skill Gen:** Ask Copilot to "Create a skill for calculating Pivot Points". Verify it creates valid directory + Pydantic object.
3.  **Risk Link:** Compile `TestRisk.mq5` (using `QuantMind_Risk.mqh`) and check if it posts a heartbeat to the server.

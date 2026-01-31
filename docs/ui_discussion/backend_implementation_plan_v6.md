# Backend "Bridge the Gap" Implementation Plan (v6 - Final)

**Objective:** Implement the missing backend assets required to support the "Hybrid Risk" architecture, "Shared Asset Library", and "Agent Capabilities" upgrades.

**Target Audience:** Coding Agent / Backend Engineer / Copilot.

---

## 🤖 Component 1: Agent Capabilities Upgrade

**Goal:** empower all agents with standardized tools and specific meta-skills.

### 1.1 Universal Slash Commands
**Location:** `src/agents/core/base_agent.py`
**Changes:**
*   Implement `CommandParser` middleware.
*   Support standard commands: `/code`, `/file`, `/run`.
*   **Reference:** Existing logic in `src/agents/core/base_agent.py` needs to be wrapped.

### 1.2 The "Skill Creator" Meta-Skill (For Copilot)
**Location:** `src/agents/skills/system_skills/skill_creator.py`
**Source URL:** [Anthropic Skill Creator](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md)
**Role:** Allows the Copilot to scaffold *new* skills for other agents.
**Logic:**
1.  **Scaffold Structure:** Create directory `src/agents/skills/[category]/[name]/`.
2.  **Generate `SKILL.md`:** Write YAML frontmatter + Instructions (Use the template from the GitHub URL).
3.  **Generate `__init__.py`:** Create the `SkillDefinition` object (Pydantic) to register it in `src/agents/skills/skill_schema.py`.

---

## 🏗️ Component 2: The Shared Asset Library (`QuantMind_Risk.mqh`)

**Context:** The Data Access Layer (DAO) for risk parameters.

**Location:** `extensions/mql5_library/Include/QuantMind/Risk.mqh`

### 2.1 Core Responsibilities
1.  **Read Global State:** Check `GlobalVariableGet("QM_RISK_MULTIPLIER")` (Fast Path).
2.  **Read Hybrid State (Fallback):** Read `risk_matrix.json` (Path: `~/.wine/.../Common/Files/risk_matrix.json`).
3.  **Provide Clean Interface:** `GetRiskMultiplier(symbol)`.

### 2.2 *New* Feature: REST Heartbeat
**Source File:** `data/scraped_articles/integration/developing_an_mql5_reinforcement_learning_agent_with_restapi_integration__part_1___how_to_use_restap.md`
**Logic:**
*   Use `WebRequest("POST", ...)` to send a JSON heartbeat to the Python Server (`http://localhost:8000/heartbeat`).
*   **Why:** Allows the "System Monitor" UI to show which EAs are active *without* complex ZMQ sockets.

---

## 🔄 Component 3: Data & Sync Layer

### 3.1 Hybrid Sync (DiskSyncer)
**Location:** `src/router/sync.py`
**Role:** Writes risk parameters to `risk_matrix.json` for the `@swing` bots.
**Logic:**
*   Must implement atomic writing (write to temp, then rename) to prevent read-errors in MT5.

*(Note: Content Syncing is handled by existing CLI scripts, not part of this backend plan.)*

---

## 🛠️ Component 4: Expert Coding Skills

### 4.1 Indicator Writer (MQL5 Generator)
**Location:** `src/agents/skills/trading_skills/indicator_writer.py`
**Source Pattern (Performance):** `data/scraped_articles/integration/mql5_cookbook_-_creating_a_ring_buffer_for_fast_calculation_of_indicators_in_a_sliding_window.md`
**Source Rules (Syntax):** `data/scraped_articles/expert_advisors/introduction_to_mql5__part_13___a_beginner_s_guide_to_building_custom_indicators__ii.md`
**Logic:**
*   **Constraint:** The Prompt MUST enforce the `CRiBuffDbl` (Ring Buffer) class structure found in the Cookbook article.
*   **Syntax:** Use the `OnCalculate` signatures found in the Intro article.

### 4.2 Python Backtester Engine
**Location:** `src/backtesting/engine.py`
**Source File:** `data/scraped_articles/integration/python-metatrader_5_strategy_tester__part_02___dealing_with_bars__ticks__and_overloading_built-in_fu.md`
**Logic:**
*   Implement `PythonStrategyTester` class.
*   **Pattern:** Copy the "Built-in Function Overloading" logic (e.g., `iTime`, `iClose` implemented in Python) from the article.

---

## 🧪 Verification Steps

1.  **Agent Check:** Run `Copilot` and try `/run ls -la`.
2.  **Skill Gen:** Ask Copilot to "Create a skill for calculating Pivot Points". Verify it creates valid directory + Pydantic object.
3.  **Heartbeat Check:** Compile `TestRisk.mq5` (using `QuantMind_Risk.mqh`) and check access logs on `localhost:8000`.

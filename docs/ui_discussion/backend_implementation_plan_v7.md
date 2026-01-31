# Backend Implementation Plan (v7)

**Objective:** Implement the "Shared Asset Library" and "backend bridges" to empower the Agent Swarm.

---

## 🏗️ component 1: QuantMind Shared Library (`Include/QuantMind/`)

**Goal:** Create a unified MQL5 library for Agents to reuse key logic (Risk, Networking, Helpers).

### 1.1 JSON Parser (`Include/QuantMind/Utils/Json.mqh`)
*   **Context:** Required for parsing `risk_matrix.json` and sending Heartbeats.
*   **Action:** Implement `CJAVal` class (or similar lightweight JSON parser).
*   **Source Reference:** *Derived from common MQL5 JSON patterns found in "Integration" articles.*

### 1.2 Network Module (`Include/QuantMind/Network/Requests.mqh`)
*   **Role:** Simplify `WebRequest` into a Python-like `requests` interface.
*   **Source Article:** `data/scraped_articles/integration/implementing_practical_modules_from_other_languages_in_mql5__part_02___building_the_requests_library.md`
*   **Key Class:** `CSession`
*   **Key Methods:** `Get()`, `Post()` (Auto-handles Headers & JSON serialization).

### 1.3 Risk Module (`Include/QuantMind/Risk/RiskMan.mqh`)
*   **Role:** The "Brain" of the EA's safety.
*   **Source Article:** `data/scraped_articles/trading/risk_management__part_1___fundamentals_for_building_a_risk_management_class.md`
*   **Key Functions:**
    *   `GetIdealLot(risk_per_trade, stop_loss)`: Calculates precise lot size based on Free Margin.
    *   `GetNetProfitSince(start_time)`: Calculates "Daily Loss" by iterating `HistoryDeals`.
*   **Logic:**
    *   **Check 1:** Is `DailyLoss > MaxDailyLoss`? -> STOP.
    *   **Check 2:** Is `Heartbeat` active? (If not, reduce Risk by 50% "Safe Mode").

---

## 🤖 Component 2: Agent Capabilities

### 2.1 Universal Slash Commands
**Location:** `src/agents/core/base_agent.py`
**Features:** `/code`, `/file`, `/run` (Shell execution).

### 2.2 Skill Creator Meta-Skill
**Location:** `src/agents/skills/system_skills/skill_creator.py`
**Reference:** [Anthropic Skill Standard](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md)

---

## 🛠️ Component 3: Expert Coding Skills

### 3.1 Indicator Writer (Ring Buffer Pattern)
**Location:** `src/agents/skills/trading_skills/indicator_writer.py`
**Source:** `data/scraped_articles/integration/mql5_cookbook_-_creating_a_ring_buffer_for_fast_calculation_of_indicators_in_a_sliding_window.md`
**Constraint:** Must use `CRiBuffDbl` class for performance.

### 3.2 Python Backtester
**Location:** `src/backtesting/engine.py`
**Source:** `data/scraped_articles/integration/python-metatrader_5_strategy_tester__part_02___dealing_with_bars__ticks__and_overloading_built-in_fu.md`

---

## 🔄 Component 4: Data Sync

### 4.1 Hybrid Sync (DiskSyncer)
**Location:** `src/router/sync.py`
**Action:** Writes `risk_matrix.json` atomicaly.

*(FireCrawl Sync removed - handled by existing CLI tools)*

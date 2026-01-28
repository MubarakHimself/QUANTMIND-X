# Technical Requirements Document: The Prop Firm Module (Extension Layer)

**System Version:** 1.0 (Extension Spec)
**Target Module:** `src/router/prop/`
**Dependency:** `src/router/` (Strategy Router Core)

---

## 1. System Overview

The **Prop Firm Module** is an **Extension Layer** designed to wrap the core Strategy Router with "Survival Logic" for funded accounts.

It utilizes a **Polymorphic Architecture** (Inheritance). Instead of a separate sidecar, we Instantiate *Smarter Versions* of the core components when a Prop Account is detected.

### 1.1 Inheritance & Linkage Specification

The **Prop Firm Module** is technically a **Plugin** that extends the Strategy Router. We strictly forbid rewriting logic.

| Module | Parent Class (Source) | Child Class (Destination) | Linkage Type |
|--------|-----------------------|---------------------------|--------------|
| **Governor** | `src.router.governor.Governor` | `src.router.prop.governor.PropGovernor` | **Inheritance** (Extends `check_risk`) |
| **Commander** | `src.router.commander.Commander` | `src.router.prop.commander.PropCommander` | **Inheritance** (Extends `run_auction`) |

---

## 2. Component 1: The Prop Governor (`src/router/prop/governor.py`)

**Parent Source:** `src/router/governor.py`

### 2.1 The Code Logic (Explicit Inheritance)

The coding agent must implement this **Exact Structure** to ensure linkage:

```python
# 1. LINKAGE: Import the Base Brain
from src.router.governor import Governor 
from src.router.prop.state import PropState

class PropGovernor(Governor):  # <-- INHERITANCE
    def __init__(self):
        super().__init__()     # Initialize Base (VaR, Swarm Logic)
        self.prop_state = PropState() # Add Prop Memory

    def calculate_risk(self, trade_proposal):
        # A. INHERIT: Call Parent to get Standard Risk (Tier 1 & 2)
        # This executes the Portfolio VaR and Strategy Edge logic defined in Phase 4.
        base_risk = super().calculate_risk(trade_proposal)
        
        # B. EXTEND: Apply specific Prop Rules (Tier 3)
        throttle = self._get_quadratic_throttle()
        
        # C. COMBINE: Modify the Parent's answer
        return base_risk * throttle

    def _get_quadratic_throttle(self):
        # Implements Balance-Based Drawdown Logic
        # ... (Safety Buffer: 1%, Hard Limit: 4%, etc)
```

### 2.2 Functional Differences

*   **Inherited (Do Not Rewrite):**
    *   `portfolio_var_check()`
    *   `swarm_cohesion_check()`
    *   `position_sizing_kelly()`
*   **Extended (New Code):**
    *   `balance_based_daily_loss()`
    *   `equity_high_water_mark()`
    *   `news_guard_override()`

---

## 3. Component 2: The Prop Commander (`src/router/prop/commander.py`)

**Parent:** `src.router.commander.Commander`

The Prop Commander inherits the "Strategy Auction" but adds **Goal-Oriented Behavior**.

### 3.1 The Goal Tracker

*   **Target:** 8.0% (Configurable per Firm).
*   **Logic:**
    *   If `Equity >= (StartingBalance + Target)`: **Trigger Preservation Mode.**

### 3.2 Preservation Mode Actions

When triggered:
1.  **Risk Floor:** Maximum Risk per trade capped at **0.25%**.
2.  **Filter Tightening:** Only accepts Strategies with `KellyScore > 0.8` (A+ Setups only).
3.  **Objective:** "Run out the clock" while protecting the win.

### 3.3 The "Coin Flip" Bot (Time Guardian)

*   **Purpose:** Satisfy "Minimum Trading Days" requirement if Target is met early.
*   **Trigger:**
    *   `ProfitTarget == REACHED`
    *   `TradingDays < MinDays`
    *   `Market == QUIET` (Asian Session)
*   **Action:**
    *   Open **0.01 Lot** Trade (Random Direction).
    *   Close after **1 minute** or **1 pip**.
    *   *Result:* Trading Day Counter increments with negligible risk ($0.10).

---

## 4. State Management (`src/router/prop/state.py`)

This module requires persistent state that survives system restarts.

*   **Redis Key:** `prop:account_id:state`
*   **Fields:**
    *   `daily_start_balance`: (Float) Snapshot at 00:00.
    *   `daily_high_water_mark`: (Float) Highest equity seen today.
    *   `trading_days_count`: (Int) Number of active days.
    *   `target_reached_date`: (String/Null) When we passed.

---

## 5. Startup Factory Logic

**Location:** `src/router/factory.py` (New Helper)

```python
def create_router_components(account_tag: str):
    if "PROP" in account_tag:
        logger.info("Initializing PROP MODE Architecture")
        return PropGovernor(), PropCommander()
    else:
        logger.info("Initializing STANDARD Architecture")
        return Governor(), Commander()
```

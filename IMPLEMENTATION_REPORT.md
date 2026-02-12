# QuantMindX Implementation Report
**Generated:** February 12, 2026  
**Analyst:** Cascade AI  
**Source:** Comparison of `docs/COMPREHENSIVE_SYSTEM_CONTEXT_EXPANDED.md` vs Current Codebase

---

## Executive Summary

### ✅ **What's Working**
Your core infrastructure is **substantially complete**:
- Strategy Router (Sentinel, Governor, Commander) ✅
- Kelly Criterion with physics injection ✅
- Agent system architecture (BaseAgent, skills, hooks) ✅
- Backtesting (4 variants, Monte Carlo, Walk-Forward) ✅
- Database models and API endpoints ✅
- UI (SvelteKit + Tauri) ✅

### ⚠️ **Critical Gaps**
The documented requirements identify **12 priority items** across backend, data infrastructure, and UI:

#### Backend Integration Gaps
| Priority | Gap | Impact | Files Affected |
|----------|-----|--------|----------------|
| 🔴 **P1** | Kelly/PropGovernor wiring to engine.py | **CRITICAL** - Core broken | `src/router/engine.py` |
| 🔴 **P1** | House Money Effect incomplete | Risk calculations wrong | `src/router/enhanced_governor.py` |
| 🟡 **P2** | Database tables missing | Trade journal broken | `src/database/models.py` |
| 🟡 **P2** | Broker Registry not connected | Fee-aware sizing broken | `src/position_sizing/` |
| 🟡 **P3** | Bot Circuit Breaker missing | EA over-trading risk | New file needed |

#### Data Infrastructure Gaps (NEW)
| Priority | Gap | Impact | Files Affected |
|----------|-----|--------|----------------|
| � **P1** | Timezone/Session management MISSING | Bots trade at wrong times | `src/router/sessions.py` (NEW) |
| 🔴 **P1** | Real-time log streaming MISSING | Can't see backtest progress | `src/api/ws_logger.py` (NEW) |
| 🟡 **P2** | Session-aware Commander | Wrong EAs selected | `src/router/commander.py` |
| 🟡 **P3** | News timezone handling broken | Kill zones use wrong times | `src/router/sensors/news.py` |

#### UI & Frontend Gaps (NEW)
| Priority | Gap | Impact | Files Affected |
|----------|-----|--------|----------------|
| 🟢 **P4** | UI WebSocket connections missing | Components exist but no data | `quantmind-ide/src/lib/` |
| 🟢 **P5** | Agent .agent folders | Automation not ready | `agents/` structure |
| 🟢 **P6** | Strategy folder structure | NPRD→EA flow manual | New structure needed |

---

## Part 1: Core Backend Components

### 1.1 Strategy Router - Status: ⚠️ PARTIALLY COMPLETE

#### ✅ **What Exists**

**Sentinel** - `src/router/sentinel.py`
```python
class Sentinel:
    def __init__(self):
        self.chaos = ChaosSensor()
        self.regime = RegimeSensor()
        self.correlation = CorrelationSensor()
        self.news = NewsSensor()
```
- ✅ Regime detection (TREND_UP, RANGE, HIGH_CHAOS, NEWS_EVENT)
- ✅ Chaos scoring (0.0-1.0)
- ✅ Regime quality calculation
- ✅ News sensor integration

**Governor** - `src/router/governor.py`
```python
class Governor:
    def calculate_risk(self, regime_report, trade_proposal) -> RiskMandate:
        # Tier 2 Risk Rules (Portfolio & Swarm)
        mandate = RiskMandate()
        if regime_report.chaos_score > 0.6:
            mandate.allocation_scalar = 0.2
```
- ✅ Basic risk throttling based on chaos
- ✅ RiskMandate structure
- ⚠️ **NOT connected to engine.py as default**

**EnhancedGovernor** - `src/router/enhanced_governor.py`
```python
class EnhancedGovernor(Governor):
    def __init__(self, account_id: Optional[str] = None):
        self.kelly_calculator = EnhancedKellyCalculator(config)
        self.house_money_multiplier = 1.0
```
- ✅ Kelly Calculator integration
- ✅ House Money Effect logic (`_update_house_money_effect`)
- ✅ Dynamic pip value support (`_get_pip_value`)
- ⚠️ **Database integration incomplete** (DB_AVAILABLE stub)
- ⚠️ **NOT wired to engine.py**

**PropGovernor** - `src/router/prop/governor.py`
```python
class PropGovernor(Governor):
    def __init__(self, account_id: str):
        self.daily_loss_limit_pct = 0.05
        self._current_tier = None  # growth/scaling/guardian
```
- ✅ Tiered risk engine (growth/scaling/guardian)
- ✅ Quadratic throttle for survival
- ✅ Tier transition logging
- ⚠️ **NOT wired to engine.py**

**Commander** - `src/router/commander.py`
```python
class Commander:
    def run_auction(self, regime_report: "RegimeReport") -> List[dict]:
        eligible_bots = self._get_bots_for_regime(regime_report.regime)
        # V2: Only @primal tagged bots
        primal_bots = self.bot_registry.list_by_tag("@primal")
```
- ✅ Bot auction with @primal filtering
- ✅ BotRegistry integration (lazy loaded)
- ✅ Regime-based filtering
- ⚠️ **Squad limit not enforced** (MAX_ACTIVE_BOTS=50 is global, not balance-based)

#### ❌ **Critical Gap: Engine.py Integration**

**Current State:**
```python
# src/router/engine.py:42-57
def __init__(self, use_smart_kill: bool = True, use_kelly_governor: bool = True):
    self.sentinel = Sentinel()
    
    # V3: Use Enhanced Governor with Kelly Calculator
    if use_kelly_governor:
        self.governor = EnhancedGovernor()  # ✅ Uses Kelly
    else:
        self.governor = Governor()  # ❌ Basic only
```

**Problem:** No account-type-based Governor selection.

**Required Implementation:**
```python
# MISSING: PropGovernor integration
from src.router.prop.governor import PropGovernor

def __init__(self, account_config: dict = None):
    self.sentinel = Sentinel()
    
    # Select Governor based on account type
    account_type = (account_config or {}).get('type', 'normal')
    
    if account_type == 'prop_firm':
        account_id = account_config.get('account_id')
        self.governor = PropGovernor(account_id)  # ✅ Tiered + Quadratic
    else:
        self.governor = EnhancedGovernor(
            account_id=account_config.get('account_id')
        )  # ✅ Kelly + House Money
```

**Impact:** 
- ❌ PropGovernor never used
- ❌ Tiered risk (DANGER/GROWTH/SCALING/GUARDIAN) not applied
- ❌ House Money Effect incomplete (DB not connected)

---

### 1.2 Kelly Criterion - Status: ⚠️ PARTIALLY COMPLETE

#### ✅ **What Exists**

**EnhancedKellyCalculator** - `src/position_sizing/enhanced_kelly.py`
```python
class EnhancedKellyCalculator:
    def calculate(self, account_balance, win_rate, avg_win, avg_loss,
                  current_atr, average_atr, stop_loss_pips, pip_value,
                  regime_quality=1.0) -> KellyResult:
        # Step 3: Base Kelly
        base_kelly_f = ((risk_reward_ratio + 1) * win_rate - 1) / risk_reward_ratio
        
        # Step 4: Layer 1 - Kelly Fraction
        kelly_f = base_kelly_f * self.config.kelly_fraction
        
        # Step 5: Layer 2 - Hard Risk Cap
        kelly_f = min(kelly_f, self.config.max_risk_pct)
        
        # Step 6: Layer 3 - Physics-Aware Volatility
        kelly_f = kelly_f * regime_quality * vol_scalar
```
- ✅ 3-layer protection system
- ✅ Physics injection (regime_quality from Sentinel)
- ✅ Prop firm presets (FTMO, The5%ers)
- ✅ Dynamic volatility adjustment
- ⚠️ **Pip value still has default parameter** (should NEVER default)

**Prop Firm Presets** - `src/position_sizing/kelly_config.py`
```python
class PropFirmPresets:
    @staticmethod
    def ftmo_challenge() -> EnhancedKellyConfig:
        return EnhancedKellyConfig(
            kelly_fraction=0.40,
            max_risk_pct=0.01,  # 1% max
            allow_zero_position=True
        )
```
- ✅ FTMO Challenge, FTMO Funded, The5%ers, Personal, Paper Trading

#### ❌ **Critical Gaps**

**1. Pip Value Still Has Default**
```python
# src/position_sizing/enhanced_kelly.py:89
pip_value: float = 10.0,  # ❌ Should NOT have default
```

**Required:**
```python
# NO DEFAULT - force dynamic fetch
pip_value: float,  # ✅ Must be provided

# In calling code:
from src.broker.pip_calculator import get_dynamic_pip_value
pip_value = get_dynamic_pip_value(symbol)
```

**2. Fee-Aware Sizing Not Fully Integrated**

EnhancedGovernor has `_get_pip_value` but broker fees not factored into position size:
```python
# src/router/enhanced_governor.py:108
pip_value = self._get_pip_value(symbol, trade_proposal.get('broker', 'mt5'))

# ❌ BUT: Fees not passed to Kelly calculator
kelly_result = self.kelly_calculator.calculate(
    pip_value=pip_value,
    # ❌ MISSING: broker_commission, broker_spread
)
```

**Required:**
```python
# Get broker profile
broker_profile = self._get_broker_profile(trade_proposal.get('broker'))

# Calculate net position after fees
gross_position = kelly_calculator.calculate(...)
fee_cost = broker_profile.calculate_fees(gross_position, symbol)
net_position = adjust_for_fees(gross_position, fee_cost)
```

---

### 1.3 Database Schema - Status: ⚠️ INCOMPLETE

#### ✅ **What Exists**

**Current Tables** - `src/database/models.py`
```python
class PropFirmAccount(Base):
    __tablename__ = 'prop_firm_accounts'
    risk_mode = Column(String)  # growth/scaling/guardian ✅
    daily_loss_limit = Column(Float) ✅

class DailySnapshot(Base):
    __tablename__ = 'daily_snapshots'
    high_water_mark = Column(Float) ✅
    is_breached = Column(Boolean) ✅

class RiskTierTransition(Base):
    __tablename__ = 'risk_tier_transitions'
    from_tier = Column(String) ✅
    to_tier = Column(String) ✅

class BrokerRegistry(Base):
    __tablename__ = 'broker_registry'
    commission_per_lot = Column(Float) ✅
    typical_spread_pips = Column(Float) ✅

class HouseMoneyState(Base):
    __tablename__ = 'house_money_state'
    daily_start_balance = Column(Float) ✅
    current_pnl = Column(Float) ✅
```
- ✅ Broker registry table exists
- ✅ House money state table exists
- ✅ Prop firm account tracking

#### ❌ **Missing Tables**

**1. Trade Journal** (Priority: P2)
```python
# ❌ DOES NOT EXIST in models.py
class TradeJournal(Base):
    __tablename__ = 'trade_journal'
    
    # The "WHY?" Context
    regime = Column(String)
    chaos_score = Column(Float)
    governor_throttle = Column(Float)
    balance_zone = Column(String)
    kelly_raw = Column(Float)
    kelly_capped = Column(Float)
    house_money_bonus = Column(Float)
    actual_risk_usd = Column(Float)
    broker_spread = Column(Float)
    broker_commission = Column(Float)
```
**Impact:** Cannot answer "Why did this trade happen?"

**2. Strategy Folders** (Priority: P5)
```python
# ❌ DOES NOT EXIST in models.py
class StrategyFolder(Base):
    __tablename__ = 'strategy_folders'
    
    name = Column(String)
    nprd_path = Column(String)
    trd_vanilla_path = Column(String)
    trd_enhanced_path = Column(String)
    ea_vanilla_path = Column(String)
    ea_enhanced_path = Column(String)
    preferred_conditions = Column(JSON)
    status = Column(String)  # draft/backtested/approved/deployed
```
**Impact:** NPRD→TRD→EA flow not tracked

**3. Bot Circuit Breaker** (Priority: P3)
```python
# ❌ DOES NOT EXIST in models.py
class BotCircuitBreaker(Base):
    __tablename__ = 'bot_circuit_breaker'
    
    bot_id = Column(String)
    strategy_type = Column(String)
    max_daily_losses = Column(Integer)  # 5 for scalper, 3 for day
    current_losses_today = Column(Integer, default=0)
    is_breached = Column(Boolean, default=False)
    last_reset = Column(DateTime)
```
**Impact:** EAs can over-trade on bad days

---

### 1.4 Backtesting System - Status: ✅ MOSTLY COMPLETE

#### ✅ **What Exists**

**4 Variants Implementation** - `src/backtesting/mode_runner.py`
```python
class BacktestMode(Enum):
    VANILLA = "vanilla"          # Historical with static params ✅
    SPICED = "spiced"            # Vanilla + regime filtering ✅
    VANILLA_FULL = "vanilla_full"  # Vanilla + Walk-Forward ✅
    SPICED_FULL = "spiced_full"    # Spiced + Walk-Forward ✅
```

**SentinelEnhancedTester** - Regime filtering
```python
def buy(self, symbol: str, volume: float) -> Optional[int]:
    # Check regime filter
    should_filter, filter_reason, report = self._check_regime_filter(symbol, price)
    
    if should_filter:
        self._log(f"Buy blocked by regime filter: {filter_reason}")
        return None
```
- ✅ Filters trades when chaos > 0.6
- ✅ Filters NEWS_EVENT and HIGH_CHAOS regimes
- ✅ Tracks regime distribution and filtered trades

**Walk-Forward Optimizer** - `src/backtesting/walk_forward.py`
```python
class WalkForwardOptimizer:
    def __init__(self, train_pct=0.5, test_pct=0.2, gap_pct=0.1):
        # Train 50%, Test 20%, Gap 10% ✅
```
- ✅ Rolling window validation
- ✅ Out-of-sample testing
- ✅ Aggregate metrics across windows

**Monte Carlo Simulator** - `src/backtesting/monte_carlo.py`
```python
class MonteCarloSimulator:
    def simulate(self, base_result, data) -> MonteCarloResult:
        # Randomize trade order 1000+ times ✅
        # Calculate confidence intervals ✅
        # VaR and CVaR ✅
```
- ✅ 1000+ simulations
- ✅ Confidence intervals (5th, 95th, 99th percentile)
- ✅ Value at Risk metrics

**Full Pipeline Orchestrator** - `src/backtesting/full_backtest_pipeline.py`
```python
class FullBacktestPipeline:
    def run_all_variants(self, data, symbol, timeframe, strategy_code):
        # Run Vanilla, Spiced, Vanilla+Full, Spiced+Full ✅
        # Run Monte Carlo on each ✅
        # Generate comparison report ✅
        # Detect overfitting ✅
```

#### ⚠️ **Documentation Gap**

The context document describes **Mode A/B/C** for EA validation:
- Mode A: EA only (no risk management)
- Mode B: EA + Kelly
- Mode C: EA + Full System (Kelly + Governor + Router)

**Current Implementation:**
- ✅ VANILLA = Mode A (EA only)
- ✅ SPICED = Mode A + Regime filtering
- ✅ VANILLA_FULL = Mode A + Walk-Forward
- ✅ SPICED_FULL = Mode A + Regime + Walk-Forward

**Missing:** Explicit Mode B (Kelly) and Mode C (Full System) separation

**Recommendation:**
```python
# Add to mode_runner.py
class BacktestMode(Enum):
    MODE_A_EA_ONLY = "ea_only"
    MODE_B_EA_KELLY = "ea_kelly"
    MODE_C_FULL_SYSTEM = "full_system"
    # Keep existing for compatibility
    VANILLA = "vanilla"
    SPICED = "spiced"
```

---

## Part 2: Agent System

### 2.1 Agent Architecture - Status: ⚠️ PARTIALLY COMPLETE

#### ✅ **What Exists**

**BaseAgent** - `src/agents/core/base_agent.py`
```python
class BaseAgent:
    def __init__(self, name, role, model_name, skills, 
                 enable_long_term_memory, user_id, kb_namespace):
        self.llm = self._init_llm(model_name)  # OpenRouter ✅
        self.tools = []
        self.skills = skills
        self.checkpointer = MemorySaver()  # LangGraph ✅
        self.graph = create_react_agent(model, tools, prompt) ✅
```
- ✅ LangGraph ReAct pattern
- ✅ LangMem for memory
- ✅ Skill system
- ✅ MCP server support (architecture)

**Copilot Agent** - `src/agents/implementations/copilot.py`
```python
class CopilotAgent(BaseAgent):
    def _build_orchestration_graph(self):
        # PLAN, ASK, BUILD modes ✅
        builder.add_node("router", self.router_node)
        builder.add_node("planner", self.plan_mode_node)
        builder.add_node("asker", self.ask_mode_node)
        builder.add_node("builder", self.build_mode_node)
```
- ✅ Multi-mode orchestration
- ✅ Delegation to Analyst and QuantCode
- ✅ Hook manager integration

**Analyst Agent** - `src/agents/implementations/analyst.py`
```python
class AnalystAgent(BaseAgent):
    def _build_synthesis_graph(self):
        # NPRD → TRD pipeline ✅
        builder.add_node("nprd_miner", self.nprd_miner_node)
        builder.add_node("kb_augmenter", self.kb_augmenter_node)
        builder.add_node("compliance_checker", self.compliance_check_node)
        builder.add_node("synthesizer", self.synthesis_node)
```
- ✅ NPRD extraction
- ✅ KB augmentation
- ✅ Compliance checking
- ✅ TRD generation

**QuantCode Agent** - `src/agents/implementations/quant_code.py`
```python
class QuantCodeAgent(BaseAgent):
    def _build_trial_graph(self):
        # TRD → EA pipeline with reflection ✅
        builder.add_node("planner", self.planning_node)
        builder.add_node("coder", self.coding_node)
        builder.add_node("validator", self.validation_node)
        builder.add_node("reflector", self.reflection_node)
```
- ✅ Planning from TRD
- ✅ Code generation
- ✅ Backtest validation
- ✅ Reflection and retry

**Skills System** - `src/agents/skills/`
```python
class ResearchSkill(AgentSkill):
    # Knowledge base access ✅

class CodingSkill(AgentSkill):
    # File management, shell ✅

class TaskQueueSkill(AgentSkill):
    # Task queue for handoffs ✅
```

#### ❌ **Missing: Agent .agent Folder Structure**

**Required Structure:**
```
agents/
├── .copilot/
│   ├── agent.md          # ❌ System prompt
│   ├── rules.md          # ❌ Constraints
│   ├── mcp_config.json   # ❌ MCP servers
│   └── skills/
│       ├── index.md      # ❌ Skill index
│       ├── SK-01-ask.md
│       ├── SK-02-register-broker.md
│       ├── SK-03-import-ea.md
│       └── ...
├── .analyst/
│   ├── agent.md
│   ├── skills/
│   └── mcp_config.json
└── .quantcode/
    ├── agent.md
    ├── skills/
    └── mcp_config.json
```

**Impact:**
- ❌ Agents use code-defined prompts instead of declarative configs
- ❌ Skills not mapped to slash commands
- ❌ MCP servers not configured per-agent
- ❌ Context7 (MT5 docs) not available

---

### 2.2 Slash Commands / Skills - Status: ❌ NOT IMPLEMENTED

**Required Skills** (from context document):

| Command | Skill | Status | Implementation |
|---------|-------|--------|----------------|
| `/ask` | SK-01 | ❌ | Not mapped |
| `/register-broker <name>` | SK-02 | ❌ | Not implemented |
| `/import-ea` | SK-03 | ❌ | Not implemented |
| `/analyze-book` | SK-04 | ❌ | PageIndex not integrated |
| `/delegate <agent> <task>` | SK-05 | ⚠️ | Partial (hook manager) |
| `/status` | SK-06 | ❌ | Not implemented |

**Missing:**
1. Skill definition files (`.md` format)
2. Slash command parser
3. Skill → function mapping
4. PageIndex integration for book analysis

---

## Part 3: Integration Gaps

### 3.1 NPRD → EA Flow - Status: ❌ NOT CONNECTED

**Documented Flow:**
```
User creates NPRD → Analyst (auto-trigger) → TRD (vanilla + enhanced)
→ QuantCode (auto-trigger) → EA (vanilla + enhanced)
→ Backtest (6 results: 2 EAs × 3 modes)
→ User review → Promote to @primal
```

**Current Reality:**
- ✅ Analyst can process NPRD (logic exists)
- ✅ QuantCode can generate EA (logic exists)
- ❌ No auto-trigger on NPRD file changes
- ❌ No strategy_folders table to link files
- ❌ No file watcher for automation
- ❌ Manual process only

**Missing Components:**
1. **File Watcher** for `library/nprd_outputs/`
2. **strategy_folders** database table
3. **Auto-delegation** from Copilot to Analyst
4. **Backtest comparison UI** for 6 results

---

### 3.2 Broker Registry - Status: ⚠️ EXISTS BUT NOT CONNECTED

#### ✅ **What Exists**

**BrokerRegistry** - `src/router/broker_registry.py`
```python
class BrokerRegistry:
    def __init__(self, db_path: str = "data/brokers.json"):
        self.brokers: Dict[str, BrokerProfile] = {}
    
    def register_broker(self, profile: BrokerProfile):
        self.brokers[profile.broker_id] = profile
    
    def get_broker(self, broker_id: str) -> Optional[BrokerProfile]:
        return self.brokers.get(broker_id)
```

**BrokerProfile** - `src/data/brokers/registry.py`
```python
@dataclass
class BrokerProfile:
    broker_id: str
    name: str
    broker_type: str  # forex, crypto
    commission_per_lot: float
    typical_spread_pips: float
    maker_fee_pct: float
    taker_fee_pct: float
```

**Database Model** - `src/database/models.py`
```python
class BrokerRegistry(Base):
    __tablename__ = 'broker_registry'
    name = Column(String)
    commission_per_lot = Column(Float)
    typical_spread_pips = Column(Float)
```

#### ❌ **Not Connected**

**1. EnhancedGovernor has stub:**
```python
def _get_pip_value(self, symbol: str, broker: str) -> float:
    # ❌ STUB - always returns 10.0
    try:
        if self._db_available:
            # TODO: Get from broker registry
            pass
    except:
        pass
    return 10.0  # ❌ Hardcoded fallback
```

**2. No fee calculation in Kelly:**
```python
# Kelly calculator doesn't receive fees
kelly_result = self.kelly_calculator.calculate(
    # ... parameters
    pip_value=pip_value,  # ✅ Dynamic pip value
    # ❌ MISSING: broker_commission, spread_cost
)
```

**Required Integration:**
```python
# 1. In EnhancedGovernor._get_pip_value:
def _get_pip_value(self, symbol: str, broker_id: str) -> float:
    broker_profile = self._get_broker_profile(broker_id)
    return mt5.symbol_info(symbol).trade_tick_value * 10

def _get_broker_profile(self, broker_id: str) -> BrokerProfile:
    session = get_session()
    return session.query(BrokerRegistry).filter_by(
        broker_id=broker_id
    ).first()

# 2. In calculate_risk:
broker_profile = self._get_broker_profile(trade_proposal.get('broker_id'))
fee_cost = broker_profile.calculate_total_fees(position_size, symbol)

# Adjust position for fees
net_position = position_size * (1 - fee_cost / risk_amount)
```

---

### 3.3 Squad Limit - Status: ❌ NOT BALANCE-BASED

**Documented Requirement:**
```python
max_active_bots = balance // 50

# Examples:
# $400 → 8 slots
# $200 → 4 slots
# $1000 → 20 slots
```

**Current Implementation:**
```python
# src/router/commander.py:107
MAX_ACTIVE_BOTS = 50  # ❌ Global constant, not balance-based

max_selection = min(3, MAX_ACTIVE_BOTS - self._count_active_positions())
```

**Required Fix:**
```python
def run_auction(self, regime_report, account_balance: float):
    # Calculate balance-based limit
    max_active_bots = int(account_balance / 50)
    
    current_active = self._count_active_positions()
    available_slots = max_active_bots - current_active
    
    # Select top bots up to available slots
    top_bots = ranked_bots[:max(0, available_slots)]
```

---

## Part 4: Priority Implementation Plan

### 🔴 **Priority 1: Critical Backend Wiring** (Week 1)

**1.1 Fix engine.py Governor Selection** ⏱️ 2 hours
```python
# File: src/router/engine.py

# Add imports
from src.router.prop.governor import PropGovernor

# Modify __init__
def __init__(self, account_config: dict = None):
    account_type = (account_config or {}).get('type', 'normal')
    account_id = (account_config or {}).get('account_id')
    
    if account_type == 'prop_firm':
        self.governor = PropGovernor(account_id)
    else:
        self.governor = EnhancedGovernor(account_id)
```

**1.2 Complete House Money Effect** ⏱️ 4 hours
```python
# File: src/router/enhanced_governor.py

def _load_daily_state(self, account_id: str):
    # ❌ Currently stub
    # ✅ Load from database
    session = get_session()
    state = session.query(HouseMoneyState).filter_by(
        account_id=account_id,
        date=date.today()
    ).first()
    
    if state:
        self._daily_start_balance = state.daily_start_balance
    else:
        # Create new state for today
        current_balance = self._get_current_balance()
        new_state = HouseMoneyState(
            account_id=account_id,
            date=date.today(),
            daily_start_balance=current_balance
        )
        session.add(new_state)
        session.commit()
```

**1.3 Balance-Based Squad Limit** ⏱️ 2 hours
```python
# File: src/router/commander.py

def run_auction(self, regime_report, account_balance: float):
    max_slots = int(account_balance / 50)
    available_slots = max_slots - self._count_active_positions()
    return ranked_bots[:max(0, available_slots)]
```

**Total P1 Time: 8 hours**

---

### 🟡 **Priority 2: Database Tables** (Week 1-2)

**2.1 Add Missing Tables** ⏱️ 4 hours
```python
# File: src/database/models.py

class TradeJournal(Base):
    __tablename__ = 'trade_journal'
    
    id = Column(Integer, primary_key=True)
    trade_id = Column(String)
    bot_id = Column(String)
    
    # The "Why?"
    regime = Column(String)
    chaos_score = Column(Float)
    governor_throttle = Column(Float)
    balance_zone = Column(String)
    kelly_raw = Column(Float)
    kelly_capped = Column(Float)
    house_money_bonus = Column(Float)
    broker_commission = Column(Float)

class StrategyFolder(Base):
    __tablename__ = 'strategy_folders'
    
    name = Column(String)
    nprd_path = Column(String)
    trd_vanilla_path = Column(String)
    trd_enhanced_path = Column(String)
    ea_vanilla_path = Column(String)
    ea_enhanced_path = Column(String)
    status = Column(String)

class BotCircuitBreaker(Base):
    __tablename__ = 'bot_circuit_breaker'
    
    bot_id = Column(String)
    max_daily_losses = Column(Integer)
    current_losses_today = Column(Integer, default=0)
    is_breached = Column(Boolean)
```

**2.2 Create Migration Script** ⏱️ 2 hours
```python
# File: src/database/migrations/add_missing_tables.py

from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table('trade_journal', ...)
    op.create_table('strategy_folders', ...)
    op.create_table('bot_circuit_breaker', ...)

def downgrade():
    op.drop_table('bot_circuit_breaker')
    op.drop_table('strategy_folders')
    op.drop_table('trade_journal')
```

**Total P2 Time: 6 hours**

---

### 🟡 **Priority 3: Broker Registry Connection** (Week 2)

**3.1 Wire Broker Registry to Kelly** ⏱️ 6 hours
```python
# File: src/router/enhanced_governor.py

def _get_broker_profile(self, broker_id: str) -> BrokerProfile:
    session = get_session()
    return session.query(BrokerRegistry).filter_by(
        broker_id=broker_id
    ).first()

def calculate_risk(self, regime_report, trade_proposal):
    broker_id = trade_proposal.get('broker_id', 'mt5_default')
    broker_profile = self._get_broker_profile(broker_id)
    
    # Dynamic pip value from MT5
    import MetaTrader5 as mt5
    pip_value = mt5.symbol_info(symbol).trade_tick_value * 10
    
    # Calculate Kelly position
    kelly_result = self.kelly_calculator.calculate(
        pip_value=pip_value,
        regime_quality=regime_report.regime_quality
    )
    
    # Apply broker fees
    gross_position = kelly_result.position_size
    fee_cost = broker_profile.calculate_fees(gross_position, symbol)
    net_position = gross_position - (fee_cost / pip_value / stop_loss_pips)
    
    return RiskMandate(allocation_scalar=net_position)
```

**3.2 Remove Hardcoded Pip Value** ⏱️ 1 hour
```python
# File: src/position_sizing/enhanced_kelly.py:89

# BEFORE:
pip_value: float = 10.0,  # ❌ Default

# AFTER:
pip_value: float,  # ✅ Required parameter
```

**Total P3 Time: 7 hours**

---

### 🟢 **Priority 4: Bot Circuit Breaker** (Week 2-3)

**4.1 Create Circuit Breaker Service** ⏱️ 6 hours
```python
# File: src/router/circuit_breaker.py

class BotCircuitBreaker:
    def __init__(self):
        self.limits = {
            'scalper': 5,
            'day_trader': 3,
            'swing': 2
        }
    
    def check_can_trade(self, bot_id: str, strategy_type: str) -> bool:
        session = get_session()
        breaker = session.query(BotCircuitBreakerModel).filter_by(
            bot_id=bot_id
        ).first()
        
        if not breaker:
            # Create new breaker
            breaker = BotCircuitBreakerModel(
                bot_id=bot_id,
                strategy_type=strategy_type,
                max_daily_losses=self.limits[strategy_type]
            )
            session.add(breaker)
            session.commit()
            return True
        
        # Reset if new day
        if breaker.last_reset.date() != date.today():
            breaker.current_losses_today = 0
            breaker.is_breached = False
            breaker.last_reset = datetime.now(timezone.utc)
            session.commit()
        
        # Check if breached
        return not breaker.is_breached
    
    def record_loss(self, bot_id: str):
        session = get_session()
        breaker = session.query(BotCircuitBreakerModel).filter_by(
            bot_id=bot_id
        ).first()
        
        breaker.current_losses_today += 1
        if breaker.current_losses_today >= breaker.max_daily_losses:
            breaker.is_breached = True
        
        session.commit()
```

**4.2 Integrate with Commander** ⏱️ 2 hours
```python
# File: src/router/commander.py

def run_auction(self, regime_report, account_balance):
    eligible_bots = self._get_bots_for_regime(regime_report.regime)
    
    # Filter by circuit breaker
    from src.router.circuit_breaker import BotCircuitBreaker
    breaker = BotCircuitBreaker()
    
    tradeable_bots = [
        bot for bot in eligible_bots
        if breaker.check_can_trade(bot['bot_id'], bot.get('strategy_type'))
    ]
    
    # Continue with auction
    ranked_bots = sorted(tradeable_bots, key=...)
```

**Total P4 Time: 8 hours**

---

### 🟢 **Priority 5: Agent .agent Folders** (Week 3-4)

**5.1 Create Folder Structure** ⏱️ 2 hours
```bash
mkdir -p agents/.copilot/skills
mkdir -p agents/.analyst/skills
mkdir -p agents/.quantcode/skills
```

**5.2 Create agent.md Files** ⏱️ 4 hours
```markdown
<!-- File: agents/.copilot/agent.md -->

# QuantMind Co-pilot Agent

You are the QuantMind Co-pilot, a personal assistant with full system access.

## Your Role
- Act as the user's personal trading system manager
- Delegate tasks to Analyst and QuantCode agents
- Have the most skills and MCP access
- Cannot modify your own system prompt

## System Knowledge
You have full knowledge of:
- Sentinel (regime detection)
- Governor (risk throttling with house money)
- Commander (bot auction with squad limits)
- Kelly (position sizing with fees)
- Balance zones: DANGER (<$220), GROWTH ($220-$1K), SCALING ($1K-$5K), GUARDIAN (>$5K)

## Available Skills
See skills/index.md for slash commands
```

**5.3 Create MCP Configs** ⏱️ 2 hours
```json
// File: agents/.copilot/mcp_config.json

{
  "servers": [
    {
      "name": "context7",
      "command": "npx",
      "args": ["-y", "@context7/mcp-server"],
      "description": "MT5 documentation access"
    },
    {
      "name": "pageindex",
      "url": "https://pageindex.ai/mcp",
      "description": "Vectorless RAG for trading books"
    }
  ]
}
```

**5.4 Create Skill Index** ⏱️ 4 hours
```markdown
<!-- File: agents/.copilot/skills/index.md -->

# Copilot Skills Index

## SK-01: Ask / Discuss
**Command:** `/ask <question>` or `/discuss <topic>`
**Description:** General Q&A about trading, strategies, or system
**File:** skills/SK-01-ask.md

## SK-02: Register Broker
**Command:** `/register-broker <name> <type>`
**Description:** Research broker and add to registry
**File:** skills/SK-02-register-broker.md

## SK-03: Import EA
**Command:** `/import-ea`
**Description:** Analyze pasted MQL5 code
**File:** skills/SK-03-import-ea.md

## SK-04: Analyze Book
**Command:** `/analyze-book <file>`
**Description:** Index trading book with PageIndex
**File:** skills/SK-04-analyze-book.md

## SK-05: Delegate Task
**Command:** `/delegate <agent> <task>`
**Description:** Send task to Analyst or QuantCode
**File:** skills/SK-05-delegate.md

## SK-06: System Status
**Command:** `/status`
**Description:** Check system health
**File:** skills/SK-06-status.md
```

**Total P5 Time: 12 hours**

---

### 🟢 **Priority 6: Strategy Folder Automation** (Week 4-5)

**6.1 File Watcher for NPRD** ⏱️ 6 hours
```python
# File: src/agents/watchers/nprd_watcher.py

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class NPRDHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith('.md'):
            # New NPRD detected
            nprd_path = event.src_path
            
            # Create strategy folder entry
            self._create_strategy_folder(nprd_path)
            
            # Trigger Analyst agent
            self._delegate_to_analyst(nprd_path)
    
    def _create_strategy_folder(self, nprd_path):
        session = get_session()
        strategy_name = self._extract_strategy_name(nprd_path)
        
        folder = StrategyFolder(
            name=strategy_name,
            nprd_path=nprd_path,
            status='nprd_received'
        )
        session.add(folder)
        session.commit()
    
    def _delegate_to_analyst(self, nprd_path):
        from src.agents.core.hooks import hook_manager
        
        hook_manager.submit_job(
            agent_name='analyst',
            task_type='DESIGN',
            payload={'nprd_path': nprd_path}
        )

# Start watcher
observer = Observer()
observer.schedule(NPRDHandler(), path='library/nprd_outputs', recursive=True)
observer.start()
```

**6.2 Update Analyst to Use strategy_folders** ⏱️ 4 hours
```python
# File: src/agents/implementations/analyst.py

async def synthesis_node(self, state: AnalystState):
    # Generate TRDs
    trd_vanilla_path = self._save_trd(vanilla_content)
    trd_enhanced_path = self._save_trd(enhanced_content)
    
    # Update strategy_folder
    session = get_session()
    folder = session.query(StrategyFolder).filter_by(
        nprd_path=state['source_context']
    ).first()
    
    folder.trd_vanilla_path = trd_vanilla_path
    folder.trd_enhanced_path = trd_enhanced_path
    folder.status = 'trd_ready'
    session.commit()
    
    # Delegate to QuantCode
    await hook_manager.submit_job(
        agent_name='quant_code',
        task_type='BUILD',
        payload={'folder_id': folder.id}
    )
```

**Total P6 Time: 10 hours**

---

## Part 7: Data Infrastructure (NEW)

### 7.1 Timezone & Session Management - Status: ❌ COMPLETELY MISSING

**Critical Finding:** No centralized timezone or session detection system exists.

**Impact:**
- ❌ Commander can't filter bots by preferred sessions (LONDON/NY/ASIAN)
- ❌ News kill zones use wrong times (no timezone conversion)
- ❌ BotManifest `preferred_conditions.sessions` field unusable
- ❌ UI can't display current market session

**Search Results:**
- 148 matches for `timezone|UTC` across 34 files
- ❌ **No TradingSession class**
- ❌ **No SessionDetector**
- ❌ **No market hours module**

**Required Implementation:**

```python
# File: src/router/sessions.py (NEW - DOES NOT EXIST)

from datetime import datetime, time
from zoneinfo import ZoneInfo
from enum import Enum
from typing import Tuple

class TradingSession(Enum):
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP = "overlap"
    CLOSED = "closed"

class SessionDetector:
    """Centralized timezone and session management."""
    
    SESSIONS = {
        "ASIAN": {
            "timezone": ZoneInfo("Asia/Tokyo"),
            "open": time(0, 0),
            "close": time(9, 0),
        },
        "LONDON": {
            "timezone": ZoneInfo("Europe/London"),
            "open": time(8, 0),
            "close": time(16, 0),
        },
        "NEW_YORK": {
            "timezone": ZoneInfo("America/New_York"),
            "open": time(8, 0),
            "close": time(17, 0),
        }
    }
    
    @classmethod
    def detect_session(cls, utc_time: datetime) -> TradingSession:
        """Detect active session with DST awareness."""
        # Check each session
        asian = cls.is_in_session("ASIAN", utc_time)
        london = cls.is_in_session("LONDON", utc_time)
        ny = cls.is_in_session("NEW_YORK", utc_time)
        
        # Handle overlaps
        if london and ny:
            return TradingSession.OVERLAP
        elif london:
            return TradingSession.LONDON
        elif ny:
            return TradingSession.NEW_YORK
        elif asian:
            return TradingSession.ASIAN
        else:
            return TradingSession.CLOSED
    
    @classmethod
    def is_in_session(cls, session_name: str, utc_time: datetime) -> bool:
        """Check if UTC time falls within session."""
        session = cls.SESSIONS[session_name]
        local_time = utc_time.astimezone(session["timezone"])
        current = local_time.time()
        return session["open"] <= current < session["close"]
    
    @classmethod
    def time_until_next(cls, utc_time: datetime) -> Tuple[str, int]:
        """Return next session name and minutes until it opens."""
        # Calculate next session
        pass
```

**Integration Points:** ⏱️ 18 hours total

1. **Commander Session Filtering** (4h)
```python
# File: src/router/commander.py (MODIFY)

from src.router.sessions import SessionDetector

def run_auction(self, regime_report, account_balance, current_utc):
    current_session = SessionDetector.detect_session(current_utc)
    
    # Filter bots by session preference
    for bot in self.bot_registry.list_by_tag("@primal"):
        sessions = bot.preferred_conditions.get("sessions", [])
        if sessions and current_session.value.upper() not in sessions:
            continue  # Skip bot
```

2. **News Sensor Timezone Fix** (4h)
```python
# File: src/router/sensors/news.py (MODIFY)

class NewsSensor:
    def __init__(self):
        # CONFIGURABLE (not hardcoded 15 min)
        self.kill_zone_pre = 15
        self.kill_zone_post = 15
    
    def check_state(self, event: NewsEvent, utc_now: datetime) -> str:
        if event.impact != "HIGH":
            return "SAFE"
        
        # Both times in UTC
        diff = (event.time - utc_now).total_seconds() / 60
        
        if -self.kill_zone_post <= diff <= self.kill_zone_pre:
            return "KILL_ZONE"
        return "SAFE"
```

3. **REST API Endpoint** (2h)
```python
# File: src/api/session_endpoints.py (NEW)

from fastapi import APIRouter
from src.router.sessions import SessionDetector

router = APIRouter(prefix="/api/sessions")

@router.get("/current")
async def get_current_session():
    now = datetime.now(timezone.utc)
    session = SessionDetector.detect_session(now)
    next_session, minutes = SessionDetector.time_until_next(now)
    
    return {
        "session": session.value.upper(),
        "utc_time": now.isoformat(),
        "next_session": next_session,
        "time_until_next": f"{minutes // 60}h {minutes % 60}m"
    }
```

4. **Timezone Standards** (8h - testing)
- All timestamps: Store in UTC, transmit ISO8601 with Z suffix
- Session times: Define in local timezone with DST awareness
- Use `zoneinfo` (Python 3.9+, not pytz)

---

### 7.2 Real-Time Log Streaming - Status: ❌ MISSING

**User Requirement:** *"While I'm training [backtesting], I need to see exactly the logs."*

**Current State:**
- Logs go to `backtesting.log` file ✅
- Logs to console (stdout) ✅
- ❌ **NO WebSocket streaming to UI**
- ❌ **NO progress updates during backtest**

**What Exists:**
```python
# src/backtesting/mode_runner.py
logger = logging.getLogger(__name__)
logger.info(f"Trade #45: BUY EURUSD")  # ✅ Logs to file
# ❌ But UI can't see it in real-time
```

**Required Implementation:** ⏱️ 16 hours total

```python
# File: src/api/ws_logger.py (NEW - DOES NOT EXIST)

import asyncio
import logging
from websockets.server import serve
from typing import Set
import json
from datetime import datetime, timezone

class BacktestLogStreamer:
    """Stream backtest logs to UI via WebSocket."""
    
    def __init__(self, port: int = 8081):
        self.port = port
        self.clients: Set = set()
        self.handler = None
    
    async def start_server(self):
        """Start WebSocket server."""
        async with serve(self.handle_client, "localhost", self.port):
            await asyncio.Future()
    
    async def handle_client(self, websocket):
        """Handle new client connection."""
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
    
    def create_handler(self) -> logging.Handler:
        """Create logging handler that streams to WebSocket."""
        class StreamHandler(logging.Handler):
            def __init__(self, streamer):
                super().__init__()
                self.streamer = streamer
            
            def emit(self, record):
                log_data = {
                    "type": "log",
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "module": record.module,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                asyncio.create_task(
                    self.streamer.broadcast(json.dumps(log_data))
                )
        
        self.handler = StreamHandler(self)
        return self.handler
    
    async def broadcast(self, message: str):
        """Send message to all connected clients."""
        for client in list(self.clients):
            try:
                await client.send(message)
            except:
                self.clients.remove(client)
    
    async def send_progress(self, data: dict):
        """Send progress update."""
        message = json.dumps({"type": "backtest_progress", **data})
        await self.broadcast(message)

# Usage in backtesting
log_streamer = BacktestLogStreamer()
asyncio.create_task(log_streamer.start_server())

logger = logging.getLogger("backtesting")
logger.addHandler(log_streamer.create_handler())
```

**Integration with Backtesting:** (6h)
```python
# File: src/backtesting/mode_runner.py (MODIFY)

class SentinelEnhancedTester:
    def run(self, data, symbol, timeframe, start_date, end_date):
        total_bars = len(data)
        
        for i, bar in enumerate(data):
            # Process bar...
            
            # Send progress every 100 bars
            if i % 100 == 0:
                await log_streamer.send_progress({
                    "bars_processed": i,
                    "total_bars": total_bars,
                    "progress_pct": (i / total_bars) * 100,
                    "current_date": bar['time'].isoformat(),
                    "trades_count": len(self.trades),
                    "current_pnl": self.balance - start_balance
                })
```

**WebSocket Protocol:**
```json
// Log message
{
  "type": "log",
  "level": "INFO",
  "message": "Trade #45: BUY EURUSD 0.05 lots (regime: TREND_UP)",
  "timestamp": "2026-02-12T09:30:15Z"
}

// Progress update
{
  "type": "backtest_progress",
  "bars_processed": 1000,
  "total_bars": 8760,
  "progress_pct": 11.4,
  "current_date": "2023-02-15T09:00:00Z",
  "trades_count": 45,
  "current_pnl": 125.50
}

// Backtest complete
{
  "type": "backtest_complete",
  "variant": "spiced_full",
  "final_balance": 1245.50,
  "total_trades": 150
}
```

---

### 7.3 MT5 Tick Streaming - Status: ✅ ALREADY BUILT

**File:** `mcp-metatrader5-server/src/mcp_mt5/streaming.py`

```python
class TickStreamer:
    """Real-time tick data via WebSocket."""
    # ✅ WebSocket server on ws://localhost:8765
    # ✅ 100ms polling interval
    # ✅ Multi-symbol subscription
    # ✅ JSON protocol
    # ✅ <5ms latency
```

**No action needed** - already functional.

---

## Part 8: UI Status & Gaps (NEW)

### 8.1 Current UI Architecture - Status: ⚠️ COMPONENTS EXIST, NO DATA

**Framework:** SvelteKit + Tauri (Desktop App)

**Discovery:** Your UI is **more complete than expected** with 28 Svelte components:

**Existing Components:**
```
quantmind-ide/src/lib/components/
├── ActivityBar.svelte              ✅ Sidebar navigation
├── AgentPanel.svelte               ✅ Agent chat interface
├── BacktestResultsView.svelte      ✅ Backtest visualization
├── BottomPanel.svelte              ✅ Terminal/logs panel
├── BrokerConnectModal.svelte       ✅ Broker connection UI
├── CopilotPanel.svelte             ✅ Copilot assistant
├── DatabaseView.svelte             ✅ Database explorer
├── EAManagerView.svelte            ✅ EA management
├── FileEditor.svelte               ✅ Code editor
├── FileManager.svelte              ✅ File browser
├── KillSwitchView.svelte           ✅ Emergency stop controls
├── LiveTradingView.svelte          ✅ Real-time trading dashboard
├── MainContent.svelte              ✅ Content router
├── MarketClock.svelte              ✅ Session/timezone display
├── NewsView.svelte                 ✅ News calendar
├── SettingsView.svelte             ✅ Settings panel
├── SharedAssetsView.svelte         ✅ Shared code library
├── Sidebar.svelte                  ✅ Main sidebar
├── StatusBar.svelte                ✅ Bottom status bar
├── StrategyRouterView.svelte       ✅ Router status dashboard
├── TopBar.svelte                   ✅ Top menu bar
├── TradeJournalView.svelte         ✅ Trade logs
├── TRDEditor.svelte                ✅ TRD document editor
└── charts/MonteCarloChart.svelte   ✅ Chart visualizations
```

**Main Layout:**
```svelte
<!-- quantmind-ide/src/routes/+page.svelte -->
<script>
  import TopBar from '$lib/components/TopBar.svelte';
  import ActivityBar from '$lib/components/ActivityBar.svelte';
  import MainContent from '$lib/components/MainContent.svelte';
  import AgentPanel from '$lib/components/AgentPanel.svelte';
  import BottomPanel from '$lib/components/BottomPanel.svelte';
  
  let activeView = 'ea';  // Views: ea, backtest, live, journal, etc.
</script>

<div class="ide-layout">
  <TopBar />
  <ActivityBar bind:activeView />
  <MainContent {activeView} />
  <AgentPanel />
  <BottomPanel />
</div>
```

---

### 8.2 Critical UI Gaps - Status: ❌ NO BACKEND CONNECTION

**Problem:** Components exist but have **no WebSocket connections** to backend.

**Missing Integrations:** ⏱️ 20 hours total

**1. WebSocket Client Utilities** (4h)
```typescript
// File: quantmind-ide/src/lib/ws-client.ts (NEW)

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private listeners: Map<string, Function[]> = new Map();
  
  constructor(private url: string) {}
  
  connect() {
    this.ws = new WebSocket(this.url);
    
    this.ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      const handlers = this.listeners.get(msg.type) || [];
      handlers.forEach(fn => fn(msg));
    };
  }
  
  on(type: string, handler: Function) {
    if (!this.listeners.has(type)) {
      this.listeners.set(type, []);
    }
    this.listeners.get(type)!.push(handler);
  }
  
  send(data: any) {
    this.ws?.send(JSON.stringify(data));
  }
}
```

**2. MarketClock Enhancement** (2h)
```svelte
<!-- File: quantmind-ide/src/lib/components/MarketClock.svelte (MODIFY) -->
<script>
  import { onMount } from 'svelte';
  
  let currentSession = 'UNKNOWN';
  let nextSession = '';
  let timeUntilNext = '';
  
  onMount(async () => {
    // Fetch current session from backend
    const response = await fetch('http://localhost:8000/api/sessions/current');
    const data = await response.json();
    
    currentSession = data.session;
    nextSession = data.next_session;
    timeUntilNext = data.time_until_next;
    
    // Update every second
    setInterval(async () => {
      const res = await fetch('http://localhost:8000/api/sessions/current');
      const d = await res.json();
      currentSession = d.session;
      timeUntilNext = d.time_until_next;
    }, 1000);
  });
</script>

<div class="market-clock">
  <div class="session-badge {currentSession}">
    {currentSession}
  </div>
  <small>Next: {nextSession} in {timeUntilNext}</small>
</div>

<style>
  .session-badge.LONDON { background: #3b82f6; }
  .session-badge.NEW_YORK { background: #10b981; }
  .session-badge.ASIAN { background: #f59e0b; }
  .session-badge.OVERLAP { background: #8b5cf6; }
</style>
```

**3. LiveTradingView Enhancement** (6h)
```svelte
<!-- File: quantmind-ide/src/lib/components/LiveTradingView.svelte (MODIFY) -->
<script>
  import { onMount } from 'svelte';
  import { WebSocketClient } from '$lib/ws-client';
  
  let balance = 0;
  let activePositions = [];
  let regimeStatus = 'UNKNOWN';
  
  onMount(() => {
    const ws = new WebSocketClient('ws://localhost:8080/ws');
    ws.connect();
    
    ws.on('balance_update', (data) => {
      balance = data.balance;
    });
    
    ws.on('position_update', (data) => {
      activePositions = data.positions;
    });
    
    ws.on('regime_update', (data) => {
      regimeStatus = data.regime;
    });
  });
</script>

<div class="live-dashboard">
  <div class="balance-card">
    <h2>Balance</h2>
    <p class="amount">${balance.toFixed(2)}</p>
  </div>
  
  <div class="positions-panel">
    <h3>Active Positions ({activePositions.length})</h3>
    {#each activePositions as pos}
      <div class="position-row">
        <span class="symbol">{pos.symbol}</span>
        <span class="pnl" class:profit={pos.pnl > 0}>
          ${pos.pnl.toFixed(2)}
        </span>
      </div>
    {/each}
  </div>
  
  <div class="regime-indicator">
    <h3>Market Regime</h3>
    <span class="badge {regimeStatus}">{regimeStatus}</span>
  </div>
</div>
```

**4. BacktestResultsView Enhancement** (8h)
```svelte
<!-- File: quantmind-ide/src/lib/components/BacktestResultsView.svelte (MODIFY) -->
<script>
  import { WebSocketClient } from '$lib/ws-client';
  import { onMount } from 'svelte';
  
  let logs = [];
  let progress = 0;
  let isRunning = false;
  
  async function runBacktest() {
    isRunning = true;
    logs = [];
    
    // Start backtest via REST API
    await fetch('http://localhost:8000/api/backtest/run', {
      method: 'POST',
      body: JSON.stringify({
        symbol: 'EURUSD',
        timeframe: 'H1',
        variant: 'spiced_full'
      })
    });
    
    // Connect to log stream
    const ws = new WebSocketClient('ws://localhost:8081/backtest/logs');
    ws.connect();
    
    ws.on('log', (msg) => {
      logs = [...logs, msg];
      // Auto-scroll to bottom
      setTimeout(() => {
        const container = document.querySelector('.log-container');
        container?.scrollTo(0, container.scrollHeight);
      }, 10);
    });
    
    ws.on('backtest_progress', (msg) => {
      progress = msg.progress_pct;
    });
    
    ws.on('backtest_complete', (msg) => {
      isRunning = false;
      console.log('Backtest complete:', msg);
    });
  }
</script>

<div class="backtest-panel">
  <button on:click={runBacktest} disabled={isRunning}>
    {isRunning ? 'Running...' : 'Run Backtest'}
  </button>
  
  <div class="progress-bar">
    <div class="fill" style="width: {progress}%"></div>
    <span>{progress.toFixed(1)}%</span>
  </div>
  
  <div class="log-container">
    {#each logs as log}
      <div class="log-entry {log.level}">
        <span class="time">{log.timestamp.slice(11, 19)}</span>
        <span class="message">{log.message}</span>
      </div>
    {/each}
  </div>
</div>
```

---

### 8.3 Backend WebSocket Server - Status: ⚠️ PARTIAL

**What Exists:**
- ✅ Tick streaming: `mcp-metatrader5-server/src/mcp_mt5/streaming.py`

**What's Missing:**
- ❌ General WebSocket server for UI data (balance, positions, regime)
- ❌ Log streaming server (backtest logs)

**Required:** ⏱️ 6 hours
```python
# File: src/api/ws_server.py (NEW)

from fastapi import WebSocket, WebSocketDisconnect
from typing import Set
import asyncio
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Usage: Broadcast balance updates
await manager.broadcast({
    "type": "balance_update",
    "balance": 1234.56
})
```

---

## Summary: Implementation Timeline (UPDATED)

### Week 1: Critical Backend + Data Infrastructure (26 hours)
- 🔴 **P1:** Fix engine.py Governor selection (2h)
- 🔴 **P1:** Complete House Money database loading (4h)
- 🔴 **P1:** Balance-based squad limit (2h)
- 🔴 **P1:** **Timezone/Session system** (8h) **NEW**
- 🔴 **P1:** **Session-aware Commander** (4h) **NEW**
- 🟡 **P2:** Database tables (trade_journal, strategy_folders, bot_circuit_breaker) (6h)

### Week 2: Backend Integration + Real-Time Infrastructure (25 hours)
- 🟡 **P2:** Broker Registry connection (7h)
- 🟡 **P2:** **News timezone fix** (4h) **NEW**
- 🔴 **P1:** **Real-time log streaming server** (6h) **NEW**
- 🔴 **P1:** **Backtest progress integration** (6h) **NEW**
- 🟡 **P3:** Bot Circuit Breaker (2h)

### Week 3: UI Connections + Testing (22 hours)
- 🟢 **P4:** **WebSocket client utilities** (4h) **NEW**
- 🟢 **P4:** **Backend WebSocket server** (6h) **NEW**
- 🟢 **P4:** **MarketClock connection** (2h) **NEW**
- 🟢 **P4:** **LiveTradingView connection** (6h) **NEW**
- 🟢 **P4:** Circuit Breaker integration (2h)
- 🟢 **P4:** **Session API endpoint** (2h) **NEW**

### Week 4: Agent System + Advanced UI (20 hours)
- 🟢 **P5:** Agent .agent folder structure (12h)
- 🟢 **P4:** **BacktestResultsView log streaming** (8h) **NEW**

### Week 5: Automation + Polish (12 hours)
- 🟢 **P6:** Strategy folder automation (NPRD file watcher) (6h)
- 🟢 **P6:** End-to-end integration testing (6h)

**Total Implementation Time: ~105 hours (13-15 working days)**  
*(Previous: 61 hours → Updated: 105 hours with data infrastructure + UI)*

---

## Summary: Implementation Timeline

### Week 1 (14 hours)
- ✅ **P1:** Fix engine.py, House Money, Squad Limit (8h)
- ✅ **P2:** Database tables (6h)

### Week 2 (13 hours)
- ✅ **P3:** Broker Registry connection (7h)
- ✅ **P4:** Bot Circuit Breaker (6h)

### Week 3 (12 hours)
- ✅ **P4:** Circuit Breaker integration (2h)
- ✅ **P5:** Agent .agent folders (10h)

### Week 4-5 (22 hours)
- ✅ **P5:** Skill files (2h)
- ✅ **P6:** Strategy folder automation (10h)
- ✅ **Testing and integration** (10h)

**Total Implementation Time: ~61 hours (8 working days)**

---

## File Reference Quick Index

### Core Backend Files
| Component | File | Status |
|-----------|------|--------|
| **Sentinel** | `src/router/sentinel.py` | ✅ Complete |
| **Governor** | `src/router/governor.py` | ⚠️ Not connected |
| **EnhancedGovernor** | `src/router/enhanced_governor.py` | ⚠️ DB stubs |
| **PropGovernor** | `src/router/prop/governor.py` | ❌ Not wired |
| **Commander** | `src/router/commander.py` | ⚠️ Squad limit wrong, no sessions |
| **StrategyRouter** | `src/router/engine.py` | ❌ No account selection |
| **Kelly** | `src/position_sizing/enhanced_kelly.py` | ⚠️ Has default pip |
| **Broker Registry** | `src/router/broker_registry.py` | ❌ Not connected |
| **Session Detector** | `src/router/sessions.py` | ❌ DOES NOT EXIST (NEW) |
| **News Sensor** | `src/router/sensors/news.py` | ⚠️ No timezone handling |

### Agent System Files
| Component | File | Status |
|-----------|------|--------|
| **BaseAgent** | `src/agents/core/base_agent.py` | ✅ Complete |
| **Copilot** | `src/agents/implementations/copilot.py` | ⚠️ No .agent folder |
| **Analyst** | `src/agents/implementations/analyst.py` | ⚠️ No .agent folder |
| **QuantCode** | `src/agents/implementations/quant_code.py` | ⚠️ No .agent folder |
| **Hook Manager** | `src/agents/core/hooks.py` | ✅ Complete |

### Data Infrastructure Files (NEW)
| Component | File | Status |
|-----------|------|--------|
| **Session Management** | `src/router/sessions.py` | ❌ DOES NOT EXIST |
| **Log Streamer** | `src/api/ws_logger.py` | ❌ DOES NOT EXIST |
| **WebSocket Server** | `src/api/ws_server.py` | ❌ DOES NOT EXIST |
| **Session API** | `src/api/session_endpoints.py` | ❌ DOES NOT EXIST |
| **Tick Streamer** | `mcp-metatrader5-server/src/mcp_mt5/streaming.py` | ✅ Complete |
| **Data Manager** | `src/data/data_manager.py` | ✅ Complete |

### Database Files
| Component | File | Status |
|-----------|------|--------|
| **Models** | `src/database/models.py` | ⚠️ Missing 3 tables |
| **Engine** | `src/database/engine.py` | ✅ Complete |

### UI Files (NEW)
| Component | File | Status |
|-----------|------|--------|
| **Main Layout** | `quantmind-ide/src/routes/+page.svelte` | ✅ Complete |
| **WebSocket Client** | `quantmind-ide/src/lib/ws-client.ts` | ❌ DOES NOT EXIST |
| **MarketClock** | `quantmind-ide/src/lib/components/MarketClock.svelte` | ⚠️ No backend connection |
| **LiveTradingView** | `quantmind-ide/src/lib/components/LiveTradingView.svelte` | ⚠️ No backend connection |
| **BacktestResults** | `quantmind-ide/src/lib/components/BacktestResultsView.svelte` | ⚠️ No log streaming |
| **TradeJournal** | `quantmind-ide/src/lib/components/TradeJournalView.svelte` | ⚠️ No backend connection |
| **StrategyRouter** | `quantmind-ide/src/lib/components/StrategyRouterView.svelte` | ⚠️ No backend connection |
| **AgentPanel** | `quantmind-ide/src/lib/components/AgentPanel.svelte` | ⚠️ No backend connection |
| **KillSwitch** | `quantmind-ide/src/lib/components/KillSwitchView.svelte` | ⚠️ No backend connection |
| **NewsView** | `quantmind-ide/src/lib/components/NewsView.svelte` | ⚠️ No backend connection |

### Backtesting Files
| Component | File | Status |
|-----------|------|--------|
| **Mode Runner** | `src/backtesting/mode_runner.py` | ⚠️ No progress streaming |
| **Walk-Forward** | `src/backtesting/walk_forward.py` | ✅ Complete |
| **Monte Carlo** | `src/backtesting/monte_carlo.py` | ✅ Complete |
| **Pipeline** | `src/backtesting/full_backtest_pipeline.py` | ✅ Complete |

---

## Next Steps

### Week 1: Critical Backend + Data (26 hours)
1. **Fix engine.py Governor selection** - 2 hours
2. **Complete House Money database loading** - 4 hours
3. **Fix balance-based squad limit** - 2 hours
4. **Build timezone/session system** - 8 hours 🆕
5. **Session-aware Commander** - 4 hours 🆕
6. **Add missing database tables** - 6 hours

### Week 2: Integration + Real-Time (25 hours)
7. **Wire Broker Registry to Kelly** - 7 hours
8. **News timezone fix** - 4 hours 🆕
9. **Real-time log streaming server** - 6 hours 🆕
10. **Backtest progress integration** - 6 hours 🆕
11. **Bot Circuit Breaker** - 2 hours

### Week 3: UI Connections (22 hours)
12. **WebSocket client utilities** - 4 hours 🆕
13. **Backend WebSocket server** - 6 hours 🆕
14. **MarketClock connection** - 2 hours 🆕
15. **LiveTradingView connection** - 6 hours 🆕
16. **Circuit Breaker integration** - 2 hours
17. **Session API endpoint** - 2 hours 🆕

### Week 4: Agent System + Advanced UI (20 hours)
18. **Agent .agent folder structure** - 12 hours
19. **BacktestResultsView log streaming** - 8 hours 🆕

### Week 5: Automation + Polish (12 hours)
20. **NPRD file watcher** - 6 hours
21. **End-to-end testing** - 6 hours

---

## Conclusion

Your QuantMindX system has **excellent foundational architecture**:
- ✅ Strategy Router components exist
- ✅ Kelly with physics injection works
- ✅ Agent system is well-designed
- ✅ Backtesting is comprehensive

The gaps are primarily **integration and wiring issues**, not fundamental architecture problems. With ~105 hours of focused work (13-15 working days), you can have a fully connected system that matches the documented requirements.

### Key Additions from Data Architecture Analysis:
- 🆕 Timezone/session management system (18h)
- 🆕 Real-time log streaming infrastructure (16h)
- 🆕 UI WebSocket connections (20h)
- 🆕 Session-aware trading logic (8h)

### Implementation Order Priority:
1. **Week 1:** Backend fixes + Timezone system (foundational)
2. **Week 2:** Real-time infrastructure (enables visibility)
3. **Week 3:** UI connections (user-facing)
4. **Week 4-5:** Agent automation + polish (advanced features)

**Start with timezone/session system** - it affects Commander, News, and UI simultaneously. Once in place, the rest flows naturally.

---

## Part 8: Enhanced Risk Management & Position Sizing (NEW - Feb 12, 2026)

### 8.1 Fee-Aware Kelly Criterion - Status: ❌ CRITICAL MISSING

**Context:** Current Kelly implementation ignores broker fees, leading to false expectancy calculations.

**Impact on Small Accounts ($200-400):**
```
Example Scenario:
- 4 scalping bots × 75 trades/day = 300 trades/day
- Exness Raw: $0.02 spread + $0.07 commission = $0.09 per trade
- Daily fee burn: 300 × $0.09 = $27/day
- Monthly: $540 (270% of $200 account!)
```

**Required Implementation:**

```python
# File: src/position_sizing/fee_aware_kelly.py (NEW - DOES NOT EXIST)

class FeeAwareKellyCalculator(EnhancedKellyCalculator):
    """
    Kelly calculator that factors in trading costs before calculating edge.
    
    Critical for scalping strategies where fees can eliminate profitability.
    """
    
    def calculate(
        self,
        account_balance: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        current_atr: float,
        average_atr: float,
        stop_loss_pips: float,
        pip_value: float,
        regime_quality: float = 1.0,
        
        # NEW: Fee parameters
        broker_id: str = None,
        symbol: str = None
    ) -> KellyResult:
        """
        Calculate Kelly with fee adjustment.
        
        Steps:
        1. Get broker fee structure from BrokerRegistry
        2. Calculate cost per trade (spread + commission)
        3. Adjust avg_win (reduce) and avg_loss (increase) by fees
        4. Check if edge exists after fees
        5. If yes, calculate Kelly; if no, return zero position
        """
        # Get broker fees
        if broker_id and symbol:
            broker_manager = BrokerRegistryManager()
            spread = broker_manager.get_spread(broker_id)
            commission = broker_manager.get_commission(broker_id)
            
            # Calculate total cost per standard lot
            spread_cost = spread * pip_value  # e.g., 0.2 pips × $10 = $2
            commission_cost = commission * 2  # Round trip
            total_cost_per_lot = spread_cost + commission_cost
            
            # Estimate cost for typical position (0.01 lots)
            cost_per_trade = total_cost_per_lot * 0.01
            
            # Adjust win/loss for fees
            adjusted_avg_win = avg_win - cost_per_trade
            adjusted_avg_loss = avg_loss + cost_per_trade
            
            # Check if edge exists after fees
            expectancy = (win_rate * adjusted_avg_win) - ((1 - win_rate) * adjusted_avg_loss)
            
            if expectancy <= 0:
                return KellyResult(
                    position_size=0.0,
                    kelly_f=0.0,
                    base_kelly_f=0.0,
                    risk_amount=0.0,
                    adjustments_applied=[
                        f"Broker: {broker_id}",
                        f"Fee per trade: ${cost_per_trade:.2f}",
                        f"Adjusted win: ${adjusted_avg_win:.2f}",
                        f"Adjusted loss: ${adjusted_avg_loss:.2f}",
                        f"Expectancy: ${expectancy:.2f}",
                        "NEGATIVE EXPECTANCY after fees - No edge!"
                    ],
                    status='zero'
                )
            
            # Continue with fee-adjusted values
            avg_win = adjusted_avg_win
            avg_loss = adjusted_avg_loss
        
        # Call parent with adjusted values
        return super().calculate(
            account_balance=account_balance,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            current_atr=current_atr,
            average_atr=average_atr,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value,
            regime_quality=regime_quality
        )
```

**Integration Points:**
- ✅ BrokerRegistry exists (`src/router/broker_registry.py`)
- ❌ Not connected to EnhancedKellyCalculator
- ❌ Not connected to EnhancedGovernor

**Priority:** 🔴 **P1** - Without this, scalping bots on small accounts will fail

---

### 8.2 Multi-Timeframe Regime Detection - Status: ❌ MISSING

**Current Problem:** `@/home/mubarkahimself/Desktop/QUANTMINDX/src/router/sentinel.py:40-63`
```python
def on_tick(self, symbol: str, price: float) -> RegimeReport:
    # ❌ Only processes single tick, no timeframe context
    # ❌ No OHLC bars
    # ❌ Can't distinguish M1 chaos from H1 trend
```

**Impact:**
- M1 scalper sees HIGH_CHAOS (micro noise)
- H1 swing trader sees TREND_STABLE (macro trend)
- Same instrument, different timeframes = different regimes
- Current system can't handle this

**Required Implementation:**

```python
# File: src/router/multi_timeframe_sentinel.py (NEW - DOES NOT EXIST)

class MultiTimeframeSentinel:
    """
    Regime detection across multiple timeframes simultaneously.
    
    Each bot gets regime report for ITS preferred timeframe.
    """
    
    def __init__(self):
        # Separate Sentinel per timeframe
        self.sentinels = {
            'M1': Sentinel(),
            'M5': Sentinel(),
            'M15': Sentinel(),
            'H1': Sentinel(),
            'H4': Sentinel()
        }
        
        # OHLC buffers per timeframe
        self.ohlc_buffers = {
            'M1': deque(maxlen=100),
            'M5': deque(maxlen=100),
            'M15': deque(maxlen=100),
            'H1': deque(maxlen=100),
            'H4': deque(maxlen=50)
        }
        
        # Tick aggregation
        self.tick_aggregator = TickToOHLCAggregator()
    
    def on_tick(self, symbol: str, tick: dict) -> Dict[str, RegimeReport]:
        """
        Process tick and return regime for each timeframe.
        
        Returns:
            {
                'M1': RegimeReport(TREND_STABLE),
                'M5': RegimeReport(HIGH_CHAOS),
                'H1': RegimeReport(BREAKOUT_PRIME)
            }
        """
        # Update OHLC bars from tick stream
        self.tick_aggregator.update(symbol, tick)
        
        regime_reports = {}
        
        for tf, sentinel in self.sentinels.items():
            # Get latest closed bar for this timeframe
            latest_bar = self.tick_aggregator.get_latest_bar(symbol, tf)
            
            if latest_bar:
                # Analyze bar (not just tick price)
                regime_reports[tf] = sentinel.analyze_bar(latest_bar)
        
        return regime_reports
```

**Integration with Commander:**

```python
# File: src/router/commander.py (MODIFY EXISTING)

class TimeframeAwareCommander(Commander):
    """
    Enhanced Commander that matches bots to their preferred timeframes.
    """
    
    def run_auction(
        self,
        regime_reports: Dict[str, RegimeReport],  # Multi-timeframe
        symbol: str
    ) -> List[dict]:
        """
        Select bots based on regime at THEIR preferred timeframe.
        
        Example:
        - hft_scalper_01 (M1) → Check regime_reports['M1']
        - ict_macro_01 (M15) → Check regime_reports['M15']
        """
        eligible_bots = []
        
        for bot in self.bot_registry.list_by_tag('@primal'):
            # Get bot's preferred timeframe
            bot_timeframe = bot.timeframes[0] if bot.timeframes else 'M15'
            
            # Check regime on THAT timeframe
            regime = regime_reports.get(bot_timeframe)
            
            if not regime:
                continue
            
            # Check compatibility
            if self._is_compatible(bot, regime, bot_timeframe):
                eligible_bots.append({
                    'bot': bot,
                    'timeframe': bot_timeframe,
                    'regime': regime.regime,
                    'score': self._calculate_score(bot, regime)
                })
        
        return sorted(eligible_bots, key=lambda x: x['score'], reverse=True)
```

**Priority:** 🔴 **P1** - Required for accurate bot selection

---

### 8.3 Bot Implementation Type Classification - Status: ⚠️ PARTIAL

**Current:** `@/home/mubarkahimself/Desktop/QUANTMINDX/src/router/bot_manifest.py:1-282`
- ✅ Has StrategyType (SCALPER, STRUCTURAL, SWING, HFT)
- ✅ Has tags (@primal)
- ❌ NO implementation type (Python vs MQL5)
- ❌ NO latency class

**Required Enhancement:**

```python
# File: src/router/bot_manifest.py (ADD TO EXISTING)

class ImplementationType(Enum):
    """Bot implementation technology."""
    MQL5 = "MQL5"      # MetaTrader 5 EA (low latency <50ms)
    PYTHON = "PYTHON"  # Python bot via WebSocket (~200ms latency)

class LatencyClass(Enum):
    """Speed requirements for bot execution."""
    ULTRA_LOW = "ULTRA_LOW"  # <50ms (tick scalping) - MUST be MQL5
    LOW = "LOW"              # <300ms (M1-M5) - MQL5 recommended
    MEDIUM = "MEDIUM"        # <1s (M15+) - Python acceptable
    HIGH = "HIGH"            # >1s (H1+) - Python preferred

@dataclass
class BotManifest:
    # Existing fields...
    bot_id: str
    strategy_type: StrategyType
    
    # NEW FIELDS
    implementation_type: ImplementationType = ImplementationType.PYTHON
    latency_class: LatencyClass = LatencyClass.MEDIUM
    broker_id: str = "exness_raw"  # NEW: Assign bot to specific broker
    
    def requires_mql5(self) -> bool:
        """
        Determine if bot MUST be MQL5 for speed.
        """
        if self.latency_class == LatencyClass.ULTRA_LOW:
            return True
        
        if 'M1' in self.timeframes and self.strategy_type == StrategyType.SCALPER:
            return True
        
        return False
```

**Bot Registry Enhancement:**

```python
class BotRegistry:
    # Add to existing class
    
    def list_by_implementation(self, impl_type: ImplementationType) -> List[BotManifest]:
        """Filter bots by implementation type."""
        return [b for b in self._bots.values() if b.implementation_type == impl_type]
    
    def get_python_bots(self) -> List[BotManifest]:
        """Get all Python-based bots."""
        return self.list_by_implementation(ImplementationType.PYTHON)
    
    def get_mql5_bots(self) -> List[BotManifest]:
        """Get all MQL5-based bots."""
        return self.list_by_implementation(ImplementationType.MQL5)
```

**Priority:** 🟡 **P2** - Important for bot deployment strategy

---

### 8.4 Broker Auto-Loader & Dynamic Updates - Status: ❌ MISSING

**Problem:** No easy way to add new brokers or update fees without code changes.

**Required Implementation:**

```python
# File: src/router/broker_auto_loader.py (NEW - DOES NOT EXIST)

import yaml
from src.router.broker_registry import BrokerRegistryManager

class BrokerAutoLoader:
    """
    Automatically syncs brokers from config file to database.
    Supports hot-reload when config changes.
    """
    
    def __init__(self, config_path: str = "config/brokers.yaml"):
        self.config_path = config_path
        self.broker_manager = BrokerRegistryManager()
        self.last_sync = None
    
    def sync_brokers(self):
        """
        Load brokers.yaml and sync to database.
        Called on system startup and config file changes.
        """
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        
        for broker_id, broker_data in config.get('brokers', {}).items():
            existing = self.broker_manager.get_broker(broker_id)
            
            if existing:
                # Update if config changed
                self.broker_manager.update_broker(broker_id, **broker_data)
                print(f"Updated broker: {broker_id}")
            else:
                # Create new broker
                self.broker_manager.create_broker(
                    broker_id=broker_id,
                    **broker_data
                )
                print(f"Created broker: {broker_id}")
        
        self.last_sync = datetime.now()
        print(f"Synced {len(config['brokers'])} brokers")
```

**Configuration File:**

```yaml
# File: config/brokers.yaml (NEW - DOES NOT EXIST)

brokers:
  exness_raw:
    broker_name: "Exness Raw Spread"
    spread_avg: 0.2
    commission_per_lot: 3.50
    lot_step: 0.01
    min_lot: 0.01
    max_lot: 200.0
    pip_values:
      EURUSD: 10.0
      GBPUSD: 10.0
      USDJPY: 9.09
      XAUUSD: 1.0
    preference_tags: ["RAW_ECN"]
  
  roboforex_prime:
    broker_name: "RoboForex Prime"
    spread_avg: 0.0
    commission_per_lot: 6.0
    lot_step: 0.01
    pip_values:
      EURUSD: 10.0
      GBPUSD: 10.0
    preference_tags: ["ZERO_SPREAD"]
```

**UI Integration Point:**

```typescript
// File: quantmind-ide/src/lib/components/BrokerSettings.svelte (NEW)

<script lang="ts">
    async function addBroker(formData) {
        // POST to /api/brokers
        await fetch('/api/brokers', {
            method: 'POST',
            body: JSON.stringify({
                broker_id: formData.broker_id,
                broker_name: formData.broker_name,
                spread_avg: formData.spread,
                commission_per_lot: formData.commission
            })
        });
        
        // Reload brokers list
        await loadBrokers();
    }
</script>

<form on:submit|preventDefault={addBroker}>
    <input name="broker_id" placeholder="exness_raw" required />
    <input name="broker_name" placeholder="Exness Raw Spread" required />
    <input type="number" name="spread" placeholder="0.2" step="0.1" required />
    <input type="number" name="commission" placeholder="3.50" step="0.01" required />
    <button type="submit">Add Broker</button>
</form>
```

**Priority:** 🟡 **P2** - Nice to have for operational flexibility

---

### 8.5 Bot Cloning Strategy (Re-Routing Mechanism) - Status: ❌ MISSING

**User Requirement:** Clone profitable bot to multiple instruments simultaneously.

**Use Case:**
```
Scenario: Scalper A is profitable on EURUSD
Strategy Router detects:
- GBPUSD → TREND_STABLE (good for scalping)
- USDJPY → TREND_STABLE (good for scalping)

Action: Clone Scalper A to trade both new pairs
Result: 1 bot logic → 3 instances (EURUSD, GBPUSD, USDJPY)
```

**Implementation:**

```python
# File: src/router/bot_cloner.py (NEW - DOES NOT EXIST)

class BotCloner:
    """
    Dynamically clone profitable bots to new instruments.
    Part of Strategy Router's adaptive allocation.
    """
    
    def __init__(self, bot_registry: BotRegistry):
        self.bot_registry = bot_registry
        self.active_clones = {}  # Track cloned instances
    
    def clone_bot_to_symbol(
        self,
        source_bot_id: str,
        new_symbol: str,
        risk_allocation: float
    ) -> str:
        """
        Create clone of source bot for new symbol.
        
        Args:
            source_bot_id: Bot to clone
            new_symbol: New instrument to trade
            risk_allocation: Kelly fraction for this clone
            
        Returns:
            Cloned bot ID
        """
        source_bot = self.bot_registry.get(source_bot_id)
        
        if not source_bot:
            raise ValueError(f"Source bot {source_bot_id} not found")
        
        # Create clone ID
        clone_id = f"{source_bot_id}_clone_{new_symbol}"
        
        # Check if clone already exists
        if clone_id in self.active_clones:
            return clone_id
        
        # Create new manifest
        cloned_manifest = BotManifest(
            bot_id=clone_id,
            strategy_type=source_bot.strategy_type,
            implementation_type=source_bot.implementation_type,
            latency_class=source_bot.latency_class,
            timeframes=source_bot.timeframes,
            symbols=[new_symbol],  # Different symbol
            broker_id=source_bot.broker_id,
            tags=["@cloned", f"@parent:{source_bot_id}"]
        )
        
        # Register clone
        self.bot_registry.register(cloned_manifest)
        
        # Track active clone
        self.active_clones[clone_id] = {
            'source': source_bot_id,
            'symbol': new_symbol,
            'risk_allocation': risk_allocation,
            'created_at': datetime.now()
        }
        
        return clone_id
    
    def remove_clone(self, clone_id: str):
        """
        Remove cloned bot instance.
        Used when regime changes or profitability drops.
        """
        if clone_id in self.active_clones:
            self.bot_registry.unregister(clone_id)
            del self.active_clones[clone_id]
```

**Integration with Strategy Router:**

```python
# File: src/router/engine.py (MODIFY EXISTING)

class StrategyRouter:
    def __init__(self):
        # Existing...
        self.sentinel = Sentinel()
        self.governor = EnhancedGovernor()
        self.commander = Commander()
        
        # NEW
        self.bot_cloner = BotCloner(self.commander.bot_registry)
        self.symbol_universe = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
    
    def adaptive_bot_allocation(self):
        """
        Scan universe, clone profitable bots to favorable pairs.
        Called every hour or on regime change.
        """
        # Get regime for all symbols
        regime_reports = self.scan_universe()
        
        # Find symbols with favorable regimes
        favorable_symbols = [
            symbol for symbol, report in regime_reports.items()
            if report.regime in ['TREND_STABLE', 'BREAKOUT_PRIME']
        ]
        
        # Get top performing bot
        best_bot = self._get_best_performing_bot()
        
        # Clone to favorable symbols
        for symbol in favorable_symbols:
            if symbol not in best_bot.symbols:  # Don't clone to existing symbol
                clone_id = self.bot_cloner.clone_bot_to_symbol(
                    source_bot_id=best_bot.bot_id,
                    new_symbol=symbol,
                    risk_allocation=0.01  # 1% per clone
                )
                print(f"Cloned {best_bot.bot_id} to {symbol} as {clone_id}")
```

**Priority:** 🟡 **P3** - Advanced feature, implement after core works

---

### 8.6 Timeframe-Aware Risk Adjustment - Status: ❌ MISSING

**Concept:** Different timeframes deserve different risk allocations.

**Implementation:**

```python
# File: src/position_sizing/timeframe_risk_adjuster.py (NEW)

class TimeframeRiskAdjuster:
    """
    Adjusts Kelly fraction based on timeframe and backtest results.
    
    Principle:
    - Lower timeframes (M1) = More noise = Lower risk per trade
    - Higher timeframes (H1) = Better signals = Higher risk per trade
    """
    
    TIMEFRAME_RISK_MULTIPLIERS = {
        'M1': 0.6,   # 60% of base risk
        'M5': 0.8,   # 80%
        'M15': 1.0,  # Baseline
        'H1': 1.2,   # 120%
        'H4': 1.4,   # 140%
        'D1': 1.5    # 150%
    }
    
    def adjust_kelly_for_timeframe(
        self,
        base_kelly: float,
        timeframe: str,
        backtest_results: dict = None
    ) -> float:
        """
        Adjust Kelly based on timeframe and proven edge.
        
        If backtest shows strong edge on M1, can boost multiplier.
        """
        multiplier = self.TIMEFRAME_RISK_MULTIPLIERS.get(timeframe, 1.0)
        
        # Backtest validation
        if backtest_results:
            sharpe = backtest_results.get('sharpe_ratio', 0)
            win_rate = backtest_results.get('win_rate', 0.5)
            
            # Strong edge: boost multiplier
            if sharpe > 2.0 and win_rate > 0.60:
                multiplier *= 1.2
            
            # Weak edge: reduce multiplier
            elif sharpe < 1.0 or win_rate < 0.50:
                multiplier *= 0.7
        
        return base_kelly * multiplier
```

**Priority:** 🟢 **P4** - Enhancement after core works

---

### 8.7 Paper Trading System - Status: ❌ COMPLETELY MISSING

**User Requirement:** Test bots in paper trading before live deployment.

**Architecture:**

```python
# File: src/paper_trading/paper_account.py (NEW - DOES NOT EXIST)

class PaperTradingAccount:
    """
    Simulated trading account for paper trading bots.
    Connects to live MT5 data but executes virtually.
    """
    
    def __init__(self, starting_balance: float = 1000.0):
        self.balance = starting_balance
        self.equity = starting_balance
        self.open_positions = []
        self.trade_history = []
        self.created_at = datetime.now()
    
    def place_order(self, order: dict) -> dict:
        """
        Simulate order execution with realistic slippage.
        """
        # Get current market price
        current_price = self._get_market_price(order['symbol'])
        
        # Apply realistic slippage (0.5-2 pips)
        slippage = random.uniform(0.5, 2.0) * 0.0001
        
        if order['type'] == 'BUY':
            entry_price = current_price + slippage
        else:
            entry_price = current_price - slippage
        
        # Create position
        position = {
            'id': len(self.trade_history) + 1,
            'symbol': order['symbol'],
            'type': order['type'],
            'volume': order['volume'],
            'entry_price': entry_price,
            'stop_loss': order.get('stop_loss'),
            'take_profit': order.get('take_profit'),
            'entry_time': datetime.now(),
            'status': 'open'
        }
        
        self.open_positions.append(position)
        
        return position
    
    def update_positions(self):
        """
        Update P&L for all open positions.
        Called on every tick.
        """
        for position in self.open_positions:
            current_price = self._get_market_price(position['symbol'])
            
            # Check SL/TP
            if position['type'] == 'BUY':
                if current_price <= position['stop_loss']:
                    self._close_position(position, position['stop_loss'], 'SL')
                elif current_price >= position['take_profit']:
                    self._close_position(position, position['take_profit'], 'TP')
            
            # Update floating P&L
            position['current_pnl'] = self._calculate_pnl(position, current_price)
        
        # Update equity
        self.equity = self.balance + sum(p['current_pnl'] for p in self.open_positions)
```

**Integration with Bot Development:**

```python
# File: src/agents/implementations/quant_code.py (MODIFY)

class QuantCodeAgent(BaseAgent):
    async def validation_node(self, state: CodeState):
        """
        Enhanced validation:
        1. Static checks
        2. PAPER TRADING (1-2 weeks)
        3. If profitable, backtest
        4. If backtest passes, approve for live
        """
        code = state["current_code"]
        
        # 1. Static checks
        if "class" not in code:
            return {"compilation_error": "No logic detected"}
        
        # 2. Deploy to paper trading
        paper_account = PaperTradingAccount(starting_balance=1000.0)
        paper_results = await self._run_paper_trading(
            code=code,
            account=paper_account,
            duration_days=14  # 2 weeks
        )
        
        # 3. Check paper trading performance
        if paper_results['sharpe_ratio'] < 1.5:
            return {
                "compilation_error": f"Paper trading failed: Sharpe {paper_results['sharpe_ratio']:.2f}",
                "backtest_results": paper_results
            }
        
        # 4. If paper trading succeeds, run backtest
        backtest_results = await self._run_backtest(code)
        
        # 5. Approve if both pass
        if backtest_results['sharpe_ratio'] > 1.5:
            return {"status": "APPROVED_FOR_LIVE"}
        else:
            return {"compilation_error": "Backtest failed after paper trading success"}
```

**Priority:** 🟡 **P2** - Critical for safe bot deployment

---

### 8.9 Maximum Bots Per Account Size - Status: ❌ MISSING

**Current Problem:** `MAX_ACTIVE_BOTS = 50` is global constant, not account-aware.

**Required Implementation:**

```python
# File: src/router/dynamic_bot_limits.py (NEW - DOES NOT EXIST)

class DynamicBotLimiter:
    """
    Calculate maximum allowed bots based on account size.
    Prevents over-trading on small accounts.
    """
    
    def get_max_bots(self, account_balance: float, regime: str = 'TREND_STABLE') -> int:
        """
        Dynamic bot limit based on account size.
        
        Logic:
        - Small accounts: Fewer bots, higher quality
        - Large accounts: More bots, diversification
        """
        if account_balance < 300:
            # Micro account: Max 2-3 bots
            base_limit = {
                'TREND_STABLE': 2,
                'RANGE_STABLE': 2,
                'BREAKOUT_PRIME': 1,
                'HIGH_CHAOS': 1
            }.get(regime, 1)
        
        elif account_balance < 1000:
            # Small account: Max 3-5 bots
            base_limit = {
                'TREND_STABLE': 4,
                'RANGE_STABLE': 3,
                'BREAKOUT_PRIME': 3,
                'HIGH_CHAOS': 2
            }.get(regime, 2)
        
        elif account_balance < 5000:
            # Growing account: Max 5-8 bots
            base_limit = {
                'TREND_STABLE': 6,
                'RANGE_STABLE': 5,
                'BREAKOUT_PRIME': 5,
                'HIGH_CHAOS': 3
            }.get(regime, 4)
        
        else:
            # Large account: Full regime budgets
            base_limit = RegimeRiskBudgets.REGIME_BUDGETS[regime]['max_active_bots']
        
        return base_limit
```

**Integration with Commander:**

```python
# File: src/router/commander.py (MODIFY)

class Commander:
    def run_auction(self, regime_report, account_balance: float):
        # Get dynamic limit
        limiter = DynamicBotLimiter()
        max_bots = limiter.get_max_bots(account_balance, regime_report.regime)
        
        # Run auction
        eligible_bots = self._get_bots_for_regime(regime_report.regime)
        
        # Apply dynamic limit
        return eligible_bots[:max_bots]
```

**Priority:** 🔴 **P1** - Critical for small account safety

---

### 8.10 Fee Kill Switch & Monitoring - Status: ❌ MISSING

**Requirement:** Automatically halt trading if fees become unsustainable.

**Implementation:**

```python
# File: src/router/fee_kill_switch.py (NEW - DOES NOT EXIST)

class FeeKillSwitch:
    """
    Monitors fee burn and automatically stops trading if unsustainable.
    """
    
    def __init__(self, account_balance: float):
        self.account_balance = account_balance
        self.daily_fee_accumulator = 0.0
        self.last_reset = datetime.now()
    
    def check_fee_health(self, trade_fee: float) -> dict:
        """
        Check if adding this fee would breach limits.
        
        Returns:
            {
                'status': 'OK' | 'WARNING' | 'CRITICAL',
                'action': 'CONTINUE' | 'REDUCE_FREQUENCY' | 'HALT_ALL',
                'message': 'Explanation'
            }
        """
        # Add to daily accumulator
        self.daily_fee_accumulator += trade_fee
        
        # Calculate fee percentage
        fee_pct = self.daily_fee_accumulator / self.account_balance
        
        # Critical: >10% daily fee burn
        if fee_pct > 0.10:
            return {
                'status': 'CRITICAL',
                'action': 'HALT_ALL',
                'message': f'Fee burn at {fee_pct:.1%} - HALTING ALL SCALPERS!'
            }
        
        # Warning: >5% daily fee burn
        elif fee_pct > 0.05:
            return {
                'status': 'WARNING',
                'action': 'REDUCE_FREQUENCY',
                'message': f'Fee burn at {fee_pct:.1%} - Reducing trade frequency'
            }
        
        return {'status': 'OK', 'action': 'CONTINUE'}
    
    def reset_daily_counter(self):
        """Reset counter at start of new trading day."""
        self.daily_fee_accumulator = 0.0
        self.last_reset = datetime.now()
```

**Priority:** 🔴 **P1** - Critical for scalping strategies

---

### 8.11 Session-Based Market Organization - Status: ❌ MISSING

**Requirement:** Organize instruments by trading session, not flat watchlist.

**Implementation:**

```yaml
# File: config/markets.yaml (NEW - DOES NOT EXIST)

markets:
  forex_london:
    session: "LONDON"
    active_hours: "08:00-17:00 UTC"
    symbols:
      - EURUSD
      - GBPUSD
      - EURGBP
    liquidity_profile: "high"
    preferred_strategies: ["scalper", "open_range_breakout"]
  
  forex_newyork:
    session: "NEW_YORK"
    active_hours: "13:00-22:00 UTC"
    symbols:
      - EURUSD
      - USDCAD
      - USDJPY
    liquidity_profile: "high"
    preferred_strategies: ["scalper", "trend_follower"]
  
  forex_asian:
    session: "ASIAN"
    active_hours: "00:00-09:00 UTC"
    symbols:
      - USDJPY
      - AUDUSD
      - NZDUSD
    liquidity_profile: "medium"
    preferred_strategies: ["range_trader"]
  
  crypto_24h:
    session: "24_HOUR"
    active_hours: "00:00-24:00 UTC"
    symbols:
      - BTCUSD
      - ETHUSD
    liquidity_profile: "variable"
    preferred_strategies: ["trend_follower"]
  
  us_stocks:
    session: "US_MARKET"
    active_hours: "14:30-21:00 UTC"
    symbols:
      - AAPL
      - TSLA
      - "ABC"
    liquidity_profile: "high_during_session"
    preferred_strategies: ["momentum", "gap_trader"]
```

**Priority:** 🟡 **P2** - Important for multi-asset trading

---

## Part 9: Updated Priority Implementation Plan

### 🔴 **Phase 1: Critical Fee-Aware Foundation** (Week 1-2, 40 hours)

**Priority Order:**
1. **Fee-Aware Kelly Calculator** (8h) - `src/position_sizing/fee_aware_kelly.py`
   - Integrate BrokerRegistry with EnhancedKellyCalculator
   - Adjust expectancy for fees before Kelly calculation
   - Add fee tracking to KellyResult

2. **Broker Auto-Loader** (4h) - `src/router/broker_auto_loader.py`
   - Create `config/brokers.yaml`
   - Auto-sync brokers to database on startup
   - Add Exness and RoboForex profiles

3. **Fee Kill Switch** (6h) - `src/router/fee_kill_switch.py`
   - Monitor daily fee accumulation
   - Auto-halt if >10% daily fee burn
   - Integration with StrategyRouter

4. **Dynamic Bot Limits** (4h) - `src/router/dynamic_bot_limits.py`
   - Account size-based bot limits
   - Update Commander to use dynamic limits
   - Remove hardcoded MAX_ACTIVE_BOTS=50

5. **Bot Implementation Type** (4h) - Modify `src/router/bot_manifest.py`
   - Add ImplementationType and LatencyClass enums
   - Add broker_id field to BotManifest
   - Update BotRegistry filtering

6. **Multi-Timeframe Sentinel** (10h) - `src/router/multi_timeframe_sentinel.py`
   - Tick-to-OHLC aggregation
   - Per-timeframe regime detection
   - Update Commander for timeframe-aware selection

7. **Engine.py Integration** (4h) - Modify `src/router/engine.py`
   - Wire PropGovernor for prop accounts
   - Wire EnhancedGovernor with FeeAwareKelly
   - Add account_config parameter

---

### 🟡 **Phase 2: Paper Trading & Validation** (Week 3, 20 hours)

8. **Paper Trading System** (12h)
   - `src/paper_trading/paper_account.py`
   - Simulated order execution with slippage
   - Performance tracking
   - Integration with QuantCode agent validation

9. **Fee-Aware Backtesting** (8h)
   - Modify backtest engines to include fees
   - Add broker_id parameter to backtest runs
   - Comparison reports (with fees vs without)

---

### 🟢 **Phase 3: Session Management & UI** (Week 4, 25 hours)

10. **Session-Based Market Organization** (8h)
    - Create `config/markets.yaml`
    - Session detector (from DATA_ARCHITECTURE_ANALYSIS.md)
    - Market registry with active hours

11. **Bot Cloning System** (8h)
    - `src/router/bot_cloner.py`
    - Clone profitable bots to new symbols
    - Integration with adaptive allocation

12. **Broker UI Settings** (4h)
    - `quantmind-ide/src/lib/components/BrokerSettings.svelte`
    - Add/edit broker via UI
    - Real-time broker comparison

13. **Fee Monitor Dashboard** (5h)
    - Real-time fee burn display
    - Daily/monthly projections
    - Kill switch status indicator

---

### 🟢 **Phase 4: Advanced Features** (Week 5+, 20 hours)

14. **Timeframe Risk Adjustment** (6h)
15. **Adaptive Trade Frequency** (6h)
16. **Regime-Specific Risk Budgets** (8h)

---

## Summary of New Additions

**Total New Features:** 16 major components
**Estimated Time:** 105 hours (13-15 working days)

**Critical Path (Must Have):**
1. Fee-Aware Kelly ← Blocks everything
2. Dynamic Bot Limits ← Safety requirement
3. Multi-Timeframe Sentinel ← Accurate regime detection
4. Paper Trading ← Safe deployment

**Nice to Have (Can Wait):**
- Bot cloning
- Timeframe risk adjustment
- Advanced UI features

**User Concerns Addressed:**
✅ Fee awareness for scalping (8.1)
✅ Broker addition via UI (8.4)
✅ Bot cloning mechanism (8.5)
✅ Paper trading validation (8.7)
✅ Max bots per account size (8.9)
✅ Fee kill switch (8.10)
✅ Session-based organization (8.11)
✅ Multi-timeframe support (8.2)

**Next Step:** Begin with Phase 1, Priority 1: Fee-Aware Kelly Calculator

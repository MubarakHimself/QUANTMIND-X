# QuantMindLib V1 — Evaluation and Workflows

## Evaluation System

### EvaluationOrchestrator (`src/library/evaluation/evaluation_orchestrator.py`)

The canonical evaluation cycle. Wires BotSpec → StrategyCodeGenerator → FullBacktestPipeline → EvaluationBridge → BotEvaluationProfile.

```python
class EvaluationOrchestrator:
    def evaluate(
        self,
        bot_spec: BotSpec,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> BotEvaluationProfile

    def evaluate_with_data(
        self,
        bot_spec: BotSpec,
        bars: List[Bar],
    ) -> BotEvaluationProfile
```

**Full evaluation cycle:**
1. Validate BotSpec
2. Generate strategy code via `StrategyCodeGenerator`
3. Create `FullBacktestPipeline` (from `src.backtesting`)
4. Run all variants: VANILLA, SPICED, VANILLA_FULL, SPICED_FULL
5. Derive `BotEvaluationProfile` with PBO + robustness scores
6. Generate backtest report if `BacktestReportBridge` available

### StrategyCodeGenerator (`src/library/evaluation/strategy_code_generator.py`)

```python
class StrategyCodeGenerator:
    def generate(self, bot_spec: BotSpec) -> str:
        """
        Generates strategy code from BotSpec.
        Outputs Python strategy code that can be executed by FullBacktestPipeline.
        """
```

Converts a `BotSpec` into executable strategy code. This is the critical translation point that enables BotSpec-driven backtesting.

### BacktestReportBridge (`src/library/evaluation/report_bridge.py`)

```python
class BacktestReportBridge:
    def generate_report(
        self,
        evaluation_profile: BotEvaluationProfile,
        bot_spec: BotSpec,
    ) -> str:
        """
        Generates markdown backtest report from evaluation profile.
        """
```

Generates structured markdown reports for human review.

### cTraderBacktestSchema (`src/library/evaluation/ctrader_backtest_schema.py`)

Schema compatibility layer. Ensures cTrader backtest results can be passed to `FullBacktestPipeline` with no changes.

**Status:** Schema defined, not yet tested against real cTrader data.

### EvaluationBridge (`src/library/core/bridges/lifecycle_eval_workflow_bridges.py`)

```python
class EvaluationBridge:
    def from_backtest_result(self, result: Any) -> EvaluationResult
    def to_evaluation_profile(
        self,
        results: Dict[EvaluationMode, EvaluationResult],
    ) -> BotEvaluationProfile:
        """
        Derives BotEvaluationProfile from 4-mode evaluation results.
        Computes PBO and robustness scores.
        """
```

**PBO computation:**
```python
pbo = max(0.0, 1.0 - walk_forward_stability)
```

**Robustness computation:**
```python
robustness = (wf_stability * 0.5) + (mc_quality * 0.3) + (min(1.0, sharpe/2.0) * 0.2)
```

### LifecycleBridge (`src/library/core/bridges/lifecycle_eval_workflow_bridges.py`)

```python
class LifecycleBridge:
    def request_transition(
        self,
        bot_id: str,
        target: str,  # BACKTEST / PAPER / LIVE
    ) -> bool:
        """
        Request a lifecycle transition.
        Enforces BACKTEST → PAPER → LIVE order.
        Enforces minimum paper days before LIVE.
        """
```

State machine:
```
BACKTEST ──(evaluation passes)──► PAPER
PAPER ──────(3-day minimum)──────► LIVE
LIVE ───────(DPR drop / kill)────► QUARANTINE
```

### WorkflowBridge (`src/library/core/bridges/lifecycle_eval_workflow_bridges.py`)

```python
class WorkflowBridge:
    def is_wf1_to_wf2_ready(self, bot_id: str) -> bool:
        """
        Check if bot is ready for WF2 (improvement loop).
        Requires: WF1 evaluation passed, robustness_score threshold met.
        """
```

## Workflow Bridges

### WF1Bridge (`src/library/workflows/wf1_bridge.py`)

Full implementation. Bridges TRD → BotSpec → AlphaForgeFlow.

```python
class WF1Bridge:
    def trd_to_bot_spec(self, trd_document: Dict[str, Any]) -> BotSpec
    def evaluate_candidate(self, bot_spec: BotSpec) -> BotEvaluationProfile
    def promote_to_paper(self, bot_spec: BotSpec) -> bool
    def get_workflow_state(self, bot_id: str) -> WorkflowState
```

### WF2Bridge (`src/library/workflows/wf2_bridge.py`)

Full implementation. Bridges variant lineage → ImprovementLoopFlow.

```python
class WF2Bridge:
    def prepare_mutation(
        self,
        parent_bot_spec: BotSpec,
        mutation_profile: BotMutationProfile,
    ) -> BotSpec
    def evaluate_mutation(self, bot_spec: BotSpec) -> BotEvaluationProfile
    def promote_demote_kill(
        self,
        bot_id: str,
        evaluation: BotEvaluationProfile,
    ) -> str
    def check_paper_lag(self, bot_id: str) -> bool
```

### stub_flows.py (`src/library/workflows/stub_flows.py`)

**EXPLICIT STUBS** for external Prefect flows. These are NOT real implementations.

```python
class AlgoForgeFlowStub:
    """
    Stub for flows/alpha_forge_flow.py.
    This file does not exist in this codebase.
    Real implementation: integrate with Prefect flow.
    """

class ImprovementLoopFlowStub:
    """
    Stub for flows/improvement_loop_flow.py.
    This file does not exist in this codebase.
    Real implementation: integrate with Prefect flow.
    """
```

The Prefect flow files (`flows/alpha_forge_flow.py`, `flows/improvement_loop_flow.py`) do **not exist** in the repository. They are referenced but not implemented. The workflow bridges provide the library-side contract; actual Prefect integration is a Phase 2 task.

## Evaluation Modes

| Mode | What It Tests | Sentinel-Enhanced |
|------|--------------|-----------------|
| VANILLA | Raw strategy without regime filter | No |
| SPICED | Strategy with regime filter | Yes |
| VANILLA_FULL | Raw strategy with walk-forward | No |
| SPICED_FULL | Regime-filtered with walk-forward | Yes |
| MODE_B | Alternative configuration set | No |
| MODE_C | Alternative configuration set | No |

## What Is Complete

| Component | Status |
|-----------|--------|
| EvaluationOrchestrator | ✓ Complete |
| StrategyCodeGenerator | ✓ Complete |
| BacktestReportBridge | ✓ Complete |
| cTraderBacktestSchema | ✓ Complete |
| EvaluationBridge | ✓ Complete |
| LifecycleBridge | ✓ Complete |
| WorkflowBridge | ✓ Complete |
| WF1Bridge | ✓ Complete |
| WF2Bridge | ✓ Complete |
| Prefect AlphaForgeFlow | ✗ STUB (external, not implemented) |
| Prefect ImprovementLoopFlow | ✗ STUB (external, not implemented) |

## What Is Deferred

| Component | Deferred To |
|-----------|-----------|
| Real Prefect flow implementations | Phase 2 |
| cTrader backtest engine | Phase 11 |
| External order flow data → evaluation | Phase 11 |
| EnsembleVoter live wiring to evaluation | Phase 2 |

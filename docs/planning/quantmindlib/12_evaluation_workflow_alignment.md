# 12 — Evaluation and Workflow Alignment

**Version:** 1.0
**Date:** 2026-04-08
**Purpose:** Map QuantMindLib to existing evaluation system and workflows

---

## 1. Evaluation Pipeline Overview

The existing evaluation pipeline runs 4 primary modes plus Monte Carlo and Walk-Forward analysis. The library must align with this pipeline so bots are evaluated consistently.

### Current Evaluation Chain
```
BotSpec (or TRD + strategy code)
    │
    ▼
FullBacktestPipeline.run_all_variants()
    │
    ├─► VANILLA      — default params, in-sample
    ├─► SPICED       — optimized params + regime filter, in-sample
    ├─► VANILLA_FULL — Walk-Forward, default params
    └─► SPICED_FULL  — Walk-Forward + regime filter, optimized params
    │
    ▼
MonteCarloSimulator (on VANILLA and SPICED results)
    │
    ▼
PBOCalculator (probability of backtest overfitting)
    │
    ▼
Robustness score = mean([regime_robustness, WFA_validity, MC_confidence, PBO_score])
    │
    ▼
BacktestReportSubAgent (markdown report generation)
    │
    ▼
StrategyPerformance DB (kelly_score, sharpe_ratio, max_drawdown, win_rate, etc.)
```

### Library Integration Point
The EvaluationBridge is the integration point. It:
1. Translates BotSpec → backtest input format
2. Collects 4-mode results → EvaluationResult (per mode)
3. Collects MC results → BotEvaluationProfile.monte_carlo
4. Collects WFA results → BotEvaluationProfile.walk_forward
5. Reads PBO → BotEvaluationProfile.pbo_score
6. Writes to StrategyPerformance DB

---

## 2. Vanilla vs Spiced Path

### Vanilla Path (Standard Backtest)
- Strategy with default parameters
- In-sample data period
- No regime filtering
- Baseline performance measurement

### Spiced Path (Regime-Filtered)
- Strategy with optimized parameters
- In-sample data period
- Regime filtering: skip trades when `chaos_score > 0.6` or `regime in [HIGH_CHAOS, NEWS_EVENT]`
- SentinelEnhancedTester wraps the base backtest engine
- Additional output: `regime_distribution`, `filtered_trades`, `avg_regime_quality`

### Library's Role
Both paths use the same `EvaluationBridge`:
```python
evaluation_bridge.evaluate(bot_spec, mode=VANILLA)  # → EvaluationResult
evaluation_bridge.evaluate(bot_spec, mode=SPICED)   # → EvaluationResult (with regime analytics)
```

The bridge handles mode-specific translation.

---

## 3. Monte Carlo — EVALUATION, NOT MUTATION

**Critical boundary:** Monte Carlo is part of evaluation, not mutation.

### What Monte Carlo Does
1. Shuffles trade return order (1000 simulations by default)
2. Recalculates cumulative return for each shuffle
3. Reports confidence intervals (5th, 95th, 99th percentiles)
4. Reports probability of profit, probability of target return

### What Monte Carlo Does NOT Do
- Does NOT change strategy parameters
- Does NOT modify the candidate
- Does NOT generate new variants

### Library Alignment
```python
@dataclass
class MonteCarloMetrics:
    confidence_interval_5th: float
    confidence_interval_95th: float
    confidence_interval_99th: float
    value_at_risk_95: float
    expected_shortfall_95: float
    mean_return: float
    std_return: float
    median_return: float
    probability_profitable: float
    probability_target: float
    num_simulations: int = 1000
```

This schema is produced by EvaluationBridge and attached to BotEvaluationProfile.

---

## 4. Walk-Forward Analysis

### What WFA Does
1. Splits data into train/test/gap windows (default 50/20/10%)
2. Optimizes parameters on training window
3. Tests on out-of-sample window (gap excluded)
4. Steps forward and repeats
5. Aggregates results across all windows

### Library Alignment
```python
@dataclass
class WalkForwardMetrics:
    windows: List[WalkForwardWindow]  # train_result, test_result, params per window
    aggregate_sharpe_mean: float
    aggregate_sharpe_std: float
    aggregate_return_mean: float
    aggregate_return_std: float
    aggregate_drawdown_mean: float
    total_windows: int
    profitable_windows: int
    wfa_efficiency: float  # OOS performance / IS performance
    regime_stats: Dict[str, Dict]  # per-regime performance across windows
```

---

## 5. Normal Backtest, Monte Carlo, Walk-Forward Distinction

| Aspect | Normal Backtest | Monte Carlo | Walk-Forward |
|--------|----------------|-------------|---------------|
| Purpose | Measure strategy performance | Measure resilience to sequence | Measure OOS robustness |
| Data | Full or in-sample | Shuffled trade sequence | Rolling train/test windows |
| Parameters | Fixed | Fixed | Optimized per window |
| Output | EvaluationResult | MonteCarloMetrics | WalkForwardMetrics |
| Mutation? | No | No | No |
| Library path | `evaluate(mode=VANILLA)` | `evaluate_mc()` | `evaluate_wfa()` |

---

## 6. WF1 Alignment (AlphaForgeFlow)

### WF1 Input
```
TRDDocument (from ResearchHead)
    │
    ▼
TRD → BotSpec conversion
    │
    ▼
BotSpec (static profile)
    │
    ▼
Composer.load(ArchetypeSpec) + Composer.validate()
    │
    ▼
BotStateManager initialized
    │
    ▼
AlphaForgeFlow.trigger(BotSpec)
    │
    ▼
Development (strategy code generation)
    │
    ▼
Compilation (MQL5 or cTrader)
    │
    ▼
EvaluationBridge.run_evaluation(bot_spec, strategy_code, mode=ALL)
    │
    ▼
SIT Gate: >= 4 of 6 modes pass, OOS degradation <= 15%
    │
    ▼
Deploy to paper trading (via enhanced_paper_trading_deployer)
```

### Library Contract for WF1
```python
# Input contract
class WF1Input:
    trd: TRDDocument
    symbol: str
    timeframe: str

# BotSpec generated from TRD
wf1_input.trd → BotSpecConverter.convert(trd)

# Output contract
class WF1Output:
    bot_spec: BotSpec
    evaluation_results: List[EvaluationResult]  # 4 modes
    monte_carlo: Optional[MonteCarloMetrics]
    walk_forward: Optional[WalkForwardMetrics]
    robustness_score: float
    sit_gate_passed: bool
```

---

## 7. WF2 Alignment (ImprovementLoopFlow)

### WF2 Input
```
Surviving variants from WF1 (BotSpec + evaluation results)
    │
    ▼
Variant lineage loaded (BotMutationProfile)
    │
    ▼
MutationEngine.apply_mutations(BotSpec, allowed_areas, locked_areas)
    │
    ▼
Mutated BotSpec
    │
    ▼
ImprovementLoopFlow.trigger(MutatedBotSpec)
    │
    ▼
Re-backtest on 20% validation split
    │
    ▼
EvaluationBridge.run_evaluation(bot_spec, mode=ALL)
    │
    ▼
Monte Carlo (1000 sims) + WFA
    │
    ▼
EvaluationResult → DPR score
    │
    ▼
Promote: DPR >= 50 → LifecycleManager.promote_to_live()
Demote: DPR drop > 20 → LifecycleManager.quarantine()
Kill: PBO > 0.5 → LifecycleManager.retire()
    │
    ▼
3-day paper lag → live promotion
```

### Library Contract for WF2
```python
# Input contract
class WF2Input:
    parent_bot_spec: BotSpec
    parent_evaluation: BotEvaluationProfile
    mutation_profile: BotMutationProfile  # allowed/locked areas
    mutation_parameters: Dict[str, Any]  # parameter changes

# Output contract
class WF2Output:
    mutated_bot_spec: BotSpec
    evaluation_results: List[EvaluationResult]
    monte_carlo: MonteCarloMetrics
    dpr_score: float
    dpr_tier: str  # T1/T2/T3
    promotion_recommendation: str  # PROMOTE / DEMOTE / KILL
    pbo_score: float
```

---

## 8. Reports and Journaling

### Backtest Report Generation
`BacktestReportSubAgent` generates structured markdown:
- Section 1: Summary (strategy type, symbol, timeframe, SIT Gate)
- Section 2: IS vs OOS table (win rate, profit factor, max drawdown, Sharpe — with degradation)
- Section 3: Monte Carlo (95th/5th percentile returns, probability of profit)
- Section 4: Walk-Forward (WFA efficiency, windows passed/total)
- Section 5: Overfitting (PBO score and flag)
- Section 6: Improvement Suggestions (AI-generated, triggered on >15% degradation)

### Library's Role
EvaluationBridge produces structured data → BacktestReportSubAgent uses data → Report attached to BotEvaluationProfile.

The library does NOT generate reports — it provides the data contract that report generation tools consume.

---

## 9. Paper Trading Handoff

### WF1 → Paper Trading
After SIT gate passes:
1. `enhanced_paper_trading_deployer.py` deploys to paper
2. Paper validation: 5 days with specific criteria
3. Paper validation passed → LifecycleManager promotion

### Library Contract
```python
@dataclass
class PaperTradingHandoff:
    bot_spec: BotSpec
    evaluation_profile: BotEvaluationProfile
    robustness_score: float
    paper_deployment_config: PaperConfig
    validation_criteria: List[str]  # e.g., ["win_rate >= 0.50", "max_dd <= 0.10"]
```

---

## 10. Evaluation Result Mapping

| EvaluationOutput | BacktestResult | LibraryType | DB Model |
|-----------------|----------------|-------------|----------|
| sharpe_ratio | MT5BacktestResult.sharpe | EvaluationResult.sharpe_ratio | StrategyPerformance.sharpe_ratio |
| max_drawdown | MT5BacktestResult.drawdown | EvaluationResult.max_drawdown | StrategyPerformance.max_drawdown |
| win_rate | MT5BacktestResult.win_rate | EvaluationResult.win_rate | StrategyPerformance.win_rate |
| profit_factor | MT5BacktestResult.profit_factor | EvaluationResult.profit_factor | StrategyPerformance.profit_factor |
| kelly_score | computed | EvaluationResult.kelly_score | StrategyPerformance.kelly_score |
| total_trades | MT5BacktestResult.trades | EvaluationResult.total_trades | StrategyPerformance.total_trades |
| return_pct | MT5BacktestResult.return_pct | EvaluationResult.return_pct | — |
| regime_distribution | SpicedBacktestResult | EvaluationResult.regime_distribution | — |
| filtered_trades | SpicedBacktestResult | EvaluationResult.filtered_trades | — |
| monte_carlo | MonteCarloResult | MonteCarloMetrics | Stored in backtest artifact JSON |
| walk_forward | WalkForwardResult | WalkForwardMetrics | Stored in backtest artifact JSON |
| pbo | PBOCalculator | EvaluationResult.pbo_score | Computed from strategy_performance |
| robustness | computed | EvaluationResult.passes_gate | Computed from above |

---

## 11. Results Map Back Into Profiles/DPR/Journaling/Registry

### BotEvaluationProfile ← EvaluationResult
```python
bot_evaluation_profile.backtest = BacktestMetrics(
    vanilla=evaluation_results[0],
    spiced=evaluation_results[1],
    vanilla_full=evaluation_results[2],
    spiced_full=evaluation_results[3],
)
bot_evaluation_profile.monte_carlo = monte_carlo_metrics
bot_evaluation_profile.walk_forward = walk_forward_metrics
bot_evaluation_profile.pbo_score = pbo_score
bot_evaluation_profile.robustness_score = robustness_score
```

### DPR ← BotEvaluationProfile
DPR bridge reads `BotEvaluationProfile.robustness_score` and `BotEvaluationProfile.pbo_score` for scoring:
- High robustness + low PBO → score boost
- Low robustness + high PBO → score penalty

### Registry ← DPR + BotEvaluationProfile
BotRegistry updated with:
- latest DPR score
- evaluation results (kelly_score, sharpe, win_rate, etc.)
- variant lineage

### Journal ← Evaluation
Evaluation results written to WF1 artifact tree:
- `backtests/{variant}/backtest_result.json`
- `reports/backtests/summary.md`
- `reports/backtests/monte_carlo.json`
- `reports/backtests/walk_forward.json`

---

## 12. Indexed References

| Aspect | Codebase Reference | Notes |
|--------|-------------------|-------|
| FullBacktestPipeline | `src/backtesting/full_backtest_pipeline.py` | Orchestrates all modes |
| 6-mode evaluation | `src/backtesting/mode_runner.py` | BacktestMode enum |
| SentinelEnhancedTester | `src/backtesting/mode_runner.py` | Regime filtering |
| MonteCarloSimulator | `src/backtesting/monte_carlo.py` | Trade-order randomization |
| WalkForwardOptimizer | `src/backtesting/walk_forward.py` | Train/test/gap windows |
| PBOCalculator | `src/backtesting/pbo_calculator.py` | CSCV bootstrap |
| BacktestReportSubAgent | `src/agents/departments/subagents/backtest_report_subagent.py` | Report generation |
| AlphaForgeFlow | `flows/alpha_forge_flow.py` | WF1 |
| ImprovementLoopFlow | `flows/improvement_loop_flow.py` | WF2 |
| StrategyPerformance | `src/database/models/performance.py` | Evaluation DB model |
| WF1 artifact tree | `src/api/wf1_artifacts.py` | Report storage |
| Recovery: Monte Carlo vs mutation | R-18 | Critical boundary |
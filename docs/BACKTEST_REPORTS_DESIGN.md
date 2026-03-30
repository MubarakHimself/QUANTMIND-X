# Backtest Reports — Agentic System Design

## Status: PENDING — Awaiting agentic system completion

## Context

User is building an agentic system. Backtest reports should be a first-class citizen in that system.
The Development department owns backtest tooling and reports.

## Data Pipeline (ALREADY EXISTS, NEEDS WIRING)

### Multiple Providers
User has data from multiple providers:
1. **histdata.com** — M1 data, resampled to M5 (via `download_and_train_hmm.py`)
2. **MT5 tick data** — collected daily, used for HMM training
3. **Other providers** — ~12 forex pairs, multi-timeframe

### Wiring Into Backtesting
- DataManager has fallback chain: MT5 → Dukascopy → API → Cache (Parquet)
- Need to extend `DataManager` to use existing downloaded data as a source
- Data likely stored in `data/` dir as Parquet/CSV from `download_and_train_hmm.py`
- After wiring, backtests use real data automatically via `DataManager.fetch_data()`

### Out-of-Sample Strategy
- MT5 tick data (90-day warm window) → out-of-sample backtest data
- HMM models trained on histdata.com data → test on MT5 tick data
- This separates training (histdata) from validation (MT5)

## Report Workflows

### Workflow 1: Initial Backtest Report
- Run backtest with 4 variants: vanilla, spiced, vanilla_full, spiced_full
- Each variant produces a report
- Reports stored in `StrategyPerformance` DB with variant/symbol/timeframe/parent_id
- Report fields: win rate, Sharpe, drawdown, profit factor, Kelly score, total trades

### Workflow 2: Agent Comparative Report
- Agents read Workflow 1 reports
- Compare variants (e.g., vanilla vs spiced)
- Write new analysis report with:
  - Best variant recommendation
  - Regime quality analysis (from Spiced mode)
  - Walk-forward stability (from vanilla_full/spiced_full modes)
  - Parameter sensitivity
  - SL/TP handling validation (do dynamic stops work?)
  - Overfitting check (PBO score)

## Report Template (Scalping Focus)

Reports should emphasize:
- **High win rate** (primary for scalping)
- **SL/TP execution quality** — verify dynamic SL/TP works in backtest
- **Drawdown recovery speed**
- **Profit factor** (gross wins / gross losses)
- **Kelly score** (position sizing fitness)
- **Regime stability** (avg_regime_quality from Spiced mode)

## Parameter Optimization
- Already exists: `BacktestTools.run_optimization()` does parameter sweeping
- Optimization results stored alongside backtest results
- WFA (Walk-Forward Analysis) via `vanilla_full`/`spiced_full` modes

## Database Persistence (DONE)
- `StrategyPerformance` table has: variant, symbol, timeframe, parent_id, backtest_results JSON
- Full metrics stored for query/reporting
- Genealogy chain via `parent_id` for variant lineage

## TODO
- [ ] Wire existing downloaded data (histdata.com + MT5 cache) into DataManager as source
- [ ] Ensure tick data (90-day window) available as OOS backtest data
- [ ] Design report template Svelte component for Development department
- [ ] Agent reads StrategyPerformance, writes comparative analysis
- [ ] Frontend: BacktestRunner.svelte already calls `/api/v1/backtest/run` — results display works
- [ ] Verify all 12 pairs covered in data collection

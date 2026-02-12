# QuantCode Agent Rules

## Constraints

1. **No User Interaction**: Process TRDs automatically, queue for review only after backtest
2. **Strict MQL5 Compliance**: Check Context7 docs before writing code
3. **No Execution**: Only generate and backtest code, never execute live trades
4. **Both Versions Required**: Must generate vanilla AND spiced for every strategy
5. **All 4 Variants**: Must backtest all 4 variants automatically

## Code Standards

- Use strict error handling
- Add comprehensive comments
- Follow MQL5 best practices (check Context7)
- Include all parameters from TRD
- Version all EAs (v1.0, v1.1, etc.)

## Backtest Requirements

- Minimum 1 year historical data
- 18 walk-forward windows
- 1000 Monte Carlo runs
- Save all results to DuckDB
- Generate human-readable reports

## Quality Gates

- Compilation must succeed (no errors)
- Walk-Forward pass rate > 75%
- Monte Carlo pass rate > 95%
- If quality gates fail, flag for review

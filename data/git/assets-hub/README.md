# Assets Hub

This Git repository stores raw source files for the QuantMindX Assets Hub.

## Structure

- `templates/`: Algorithm template files organized by category
  - `trend_following/`: Trend-following strategy templates
  - `mean_reversion/`: Mean-reversion strategy templates
  - `breakout/`: Breakout strategy templates
  - `momentum/`: Momentum-based strategy templates
  - `arbitrage/`: Arbitrage strategy templates

- `skills/`: Agentic skill definitions organized by category
  - `trading_skills/`: Trading-related skills (indicators, signals, risk management)
  - `system_skills/`: System-level skills (file operations, API calls, logging)
  - `data_skills/`: Data processing skills (fetching, cleaning, resampling)

- `generated_bots/`: Temporary storage for generated trading bots
  - Contents are gitignored (see .gitignore)

- `configuration/`: Configuration files for backtesting and deployment

## Usage

Templates and skills are managed through the GitClient class:
- `read_template(category, name)`: Read a template file
- `write_template(category, name, content)`: Write a template file
- `commit_template(category, name, message)`: Commit template changes

## Commit Message Format

Template commits follow this format:
```
Strategy: {name}, Backtest: Sharpe={sharpe}, Drawdown={drawdown}%
```

Example:
```
Strategy: RSI Mean Reversion, Backtest: Sharpe=1.8, Drawdown=12%
```

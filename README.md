# QUANTMIND-X: AI-Powered Trading Platform

> An autonomous trading ecosystem with multi-agent AI system, risk management, and MetaTrader 5 integration.

**Status:** Active Development

---

## What is QuantMind-X?

QuantMind-X is a comprehensive trading platform that combines:
- **Multi-Agent AI System** - Claude-powered agents for strategy analysis and development
- **Trading IDE** - Web-based interface for strategy management
- **Risk Management** - Enhanced Kelly position sizing, chaos detection
- **MT5 Integration** - Expert Advisors and bridge to MetaTrader 5

---

## Quick Start

### Backend (Python/FastAPI)

```bash
# Clone and setup
git clone https://github.com/MubarakHimself/QUANTMIND-X.git
cd QUANTMIND-X

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your API keys to .env

# Start API server
python3 -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

### Frontend (SvelteKit)

```bash
cd quantmind-ide

# Install dependencies
npm install

# Start development server
npm run dev
```

The IDE runs at http://localhost:3001

---

## Project Structure

```
QUANTMIND-X/
├── quantmind-ide/          # Frontend (SvelteKit)
├── src/                    # Backend (Python/FastAPI)
│   ├── agents/             # AI agent system
│   ├── api/               # REST API endpoints
│   ├── database/           # SQLite + DuckDB
│   ├── risk/              # Risk management
│   └── router/            # Strategy router
├── docker/                 # Docker configurations
└── docs/                  # Documentation
```

---

## Features

### AI Agents
- **Analyst Agent** - Market analysis and pattern recognition
- **Copilot Agent** - Trading assistant and orchestration
- **QuantCode Agent** - Strategy code generation

### Trading
- Paper trading with MT5 demo accounts
- Strategy backtesting
- Risk management (Kelly criterion, position sizing)
- Expert Advisor deployment

### IDE
- Strategy editor and management
- Real-time monitoring dashboard
- Backtest results visualization
- MCP server integration

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Backend | Python, FastAPI, SQLAlchemy |
| Frontend | SvelteKit, TypeScript, Tailwind |
| Database | SQLite (data), DuckDB (analytics) |
| Trading | MetaTrader 5, MT5 Bridge |
| AI | Anthropic Claude, Claude Agent SDK |

---

## Environment Variables

Create a `.env` file with:

```env
# Required
ANTHROPIC_API_KEY=your_key_here

# Optional
OPENAI_API_KEY=your_key_here
ZHIPU_API_KEY=your_key_here
MINIMAX_API_KEY=your_key_here

# Database
DATABASE_URL=sqlite:///data/quantmind.db
```

---

## API Endpoints

- `/api/chat` - Agent chat
- `/api/trading` - Trading operations
- `/api/mcp` - MCP server management
- `/api/agents` - Agent management
- `/api/floor-manager` - Floor manager control
- `/health` - Health check

---

## Documentation

See `docs/` directory for detailed documentation.

---

## License

MIT License

---

© 2026 QuantMind Labs

# QuantMindX Enhancement Proposal
**Date:** February 12, 2026  
**Session:** Strategy Router & Risk Management Enhancement Discussion  
**Status:** Ready for Implementation

---

## Table of Contents
1. [Kill Switch Protocol - Progressive Multi-Tier System](#1-kill-switch-protocol)
2. [Paper Trading System - Hybrid 3-Stage Approach](#2-paper-trading-system)
3. [UI Implementation - 9 Priority Components](#3-ui-implementation)
4. [Cloud Deployment Strategy - Contabo VPS](#4-cloud-deployment-strategy)
5. [Implementation Timeline](#5-implementation-timeline)
6. [Success Metrics](#6-success-metrics)

---

## 1. Kill Switch Protocol - Progressive Multi-Tier System

### 1.1 Current State Analysis
**Existing Implementation:**
- ✅ Basic kill switch (`src/router/kill_switch.py`)
  - STOP file monitoring
  - Manual `panic()` function
  - API/Socket command triggers
- ✅ Smart kill switch with regime-aware exits
  - IMMEDIATE, BREAKEVEN, TRAILING, SMART_LOSS strategies
  - Chaos thresholds: <0.3 wait, 0.3-0.6 trail, >0.6 immediate
- ✅ Bot circuit breaker (`src/router/bot_circuit_breaker.py`)
  - 5 consecutive losses → quarantine
  - 20 trades/day limit

### 1.2 Proposed Enhancement: 5-Tier Progressive System

#### Tier 1: Bot-Level Protection
**Triggers:**
- 5 consecutive losses → auto-quarantine
- 20 trades/day limit (configurable by account size)
- 15% daily drawdown per bot

**Response:**
- Quarantine specific bot
- Send notification
- Log reason to database

#### Tier 2: Strategy-Level Protection
**Triggers:**
- 3 bots of same family failing simultaneously
- 20% loss across all bots in strategy family
- Example: If 3 scalpers fail → pause all scalpers

**Response:**
- Halt entire strategy family
- Keep other families active
- Review required before reactivation

#### Tier 3: Account-Level Protection
**Triggers:**
- 3% daily account loss (hard stop)
- 10% weekly account loss (circuit breaker)
- Max 8 concurrent trades exceeded
- 4% max portfolio heat exceeded

**Response:**
- Immediate position closure
- Block new trades
- Email/SMS alert
- Manual review required

#### Tier 4: Session-Level Protection
**Triggers:**
- High-impact news event detected
- End of trading day (21:00 UTC)
- 50 trades/hour threshold exceeded
- Weekend trading attempt

**Response:**
- Smart exit based on current regime
- Scheduled position closure
- Rate limiting
- Trading window enforcement

#### Tier 5: System-Level Protection (Nuclear Option)
**Triggers:**
- Global chaos score > 0.75
- Broker connection loss
- 5 consecutive API errors
- Critical system error

**Response:**
- Full emergency stop
- Close all positions immediately
- System shutdown
- Administrator notification

### 1.3 Alert Level Progression

| Level | Threshold | Color | Action |
|-------|-----------|-------|--------|
| **GREEN** | 0-50% of limits | #10b981 | Continue normal trading |
| **YELLOW** | 50-70% of limits | #f59e0b | Send notifications, monitor closely |
| **ORANGE** | 70-85% of limits | #f97316 | Reduce position sizes by 50% |
| **RED** | 85-95% of limits | #ef4444 | Halt new trades, prepare exit |
| **BLACK** | >95% of limits | #18181b | Full kill switch activation |

### 1.4 Implementation Files

**New Files Required:**
```
src/router/progressive_kill_switch.py
config/kill_switch_config.yaml
```

**Configuration Schema:**
```yaml
kill_switch_tiers:
  bot_level:
    consecutive_loss_trigger: 5
    daily_trade_limit: 20
    max_daily_drawdown_pct: 0.15
    
  strategy_level:
    family_loss_trigger: 3
    family_drawdown_pct: 0.20
    
  account_level:
    daily_loss_limit_pct: 0.03
    weekly_loss_limit_pct: 0.10
    max_open_trades: 8
    max_portfolio_heat_pct: 0.04
    
  session_level:
    news_event_halt: true
    end_of_day_close: "21:00 UTC"
    max_trades_per_hour: 50
    weekend_trading: false
    
  system_level:
    global_chaos_threshold: 0.75
    broker_connection_loss: true
    api_error_threshold: 5
```

---

## 2. Paper Trading System - Hybrid 3-Stage Approach

### 2.1 Architecture Overview

```
Stage 1: Paper Mode (Week 1-2)
    ↓
Stage 2: Demo Mode (Week 3)
    ↓
Stage 3: Live Mode (Week 4+)
```

### 2.2 Stage 1: Paper Mode (Custom Simulation)

**Purpose:** Zero-risk validation of bot logic on live data

**How It Works:**
1. Bot receives REAL-TIME market data from MT5 WebSocket
2. Strategy signals are generated normally
3. Order execution is SIMULATED in Python
4. Realistic fills applied:
   - Current market price from live feed
   - Broker spread (0.2 pips for Exness Raw)
   - Commission ($3.50 per lot per side)
   - Slippage (0.1-0.5 pips random)
5. Results tracked in separate database table

**Key Features:**
- Fee-aware execution
- Separate paper account balance ($1000 default)
- No connection to real broker
- Can run unlimited bots simultaneously

**Implementation:**
```python
# New files:
src/paper_trading/paper_trading_engine.py
src/paper_trading/paper_account.py
src/paper_trading/paper_trade.py

# Trading mode enum:
class TradingMode(Enum):
    PAPER = "PAPER"    # Simulated on live data
    DEMO = "DEMO"      # MT5 demo account
    LIVE = "LIVE"      # Real money
```

**Example Usage:**
```python
# Register bot in paper mode
paper_engine = PaperTradingEngine(data_feed)
paper_engine.register_bot("scalper_01", TradingMode.PAPER, initial_balance=1000.0)

# Execute trade (simulated)
trade = paper_engine.execute_trade(
    bot_id="scalper_01",
    symbol="EURUSD",
    side="BUY",
    volume=0.01,
    stop_loss=1.0850,
    take_profit=1.0890,
    broker_id="exness_raw"
)
```

### 2.3 Stage 2: Demo Mode (MT5 Demo Account)

**Purpose:** Test actual broker execution and infrastructure

**How It Works:**
1. Create free demo account at Exness/RoboForex
2. Bot connects to demo via MT5 API
3. REAL execution on demo account
4. Same spreads/commissions as live
5. Validates WebSocket integration

**Demo Account Setup:**
- Provider: Exness or RoboForex
- Account type: Raw Spread or ECN
- Balance: $1000-$5000
- Leverage: 1:100
- Expiration: 30-90 days (easy to recreate)

**Implementation:**
```python
# New file:
src/paper_trading/mt5_demo_manager.py

class MT5DemoManager:
    def create_demo_account(self, config: DemoAccountConfig) -> int:
        # Create demo via broker API or MT5
        pass
    
    def connect_paper_bot(self, bot_id: str, demo_login: int, 
                          demo_password: str, demo_server: str) -> bool:
        # Connect bot to demo account
        pass
    
    def get_paper_results(self, bot_id: str) -> Dict:
        # Query demo account history
        pass
```

### 2.4 Stage 3: Live Mode (Small Capital)

**Purpose:** Graduated deployment with real money

**Promotion Criteria:**
- ✅ Minimum 50 trades in paper mode
- ✅ Win rate > 55%
- ✅ Profit factor > 1.5
- ✅ Max drawdown < 15%
- ✅ At least 2 weeks of trading
- ✅ Pass same criteria in demo mode

**Graduated Capital Deployment:**
```
Week 1-2: Paper Mode ($1000 simulated)
Week 3: Demo Mode ($1000 demo account)
Week 4: Live Mode ($50-100 real)
Week 5+: Scale up to $200-400 after proven results
```

**Risk Management:**
- Start with minimum capital
- Monitor for 1 week at each capital level
- Automatic downgrade if performance degrades
- Manual approval required for scaling

### 2.5 Bot Promotion Workflow

```python
def check_promotion_eligibility(bot_id: str) -> bool:
    """
    Check if bot ready for promotion.
    
    Paper → Demo criteria:
    - 50+ trades
    - Win rate > 55%
    - Profit factor > 1.5
    - Max DD < 15%
    - 14+ days active
    
    Demo → Live criteria:
    - Same as above, validated on demo
    - Manual human approval
    """
    account = paper_accounts.get(bot_id)
    stats = account.get_statistics()
    
    return (
        stats['total_trades'] >= 50
        and stats['win_rate'] > 0.55
        and stats['profit_factor'] > 1.5
        and stats['max_drawdown'] < 0.15
        and stats['days_active'] >= 14
    )

def promote_bot(bot_id: str, from_mode: str, to_mode: str):
    """Promote bot through stages."""
    if not check_promotion_eligibility(bot_id):
        raise ValueError("Bot does not meet promotion criteria")
    
    # Update bot configuration
    bot_config.mode = to_mode
    
    # Send notification
    notify_admin(f"Bot {bot_id} promoted from {from_mode} to {to_mode}")
```

---

## 3. UI Implementation - 9 Priority Components

### 3.1 Priority 1: Fee Monitoring System (Week 1)

#### Component 1: Real-Time Fee Monitor

**Location:** Dashboard (top card)

**Features:**
- Today's fee burn ($7.83 / $27.00)
- Projected daily fees
- Monthly projection
- Breakeven return required
- Color-coded alert level
- Progress bar visualization

**Data Display:**
```
┌─────────────────────────────────────┐
│ Fee Burn Monitor              ⚠️    │
├─────────────────────────────────────┤
│ Today's Fees:           $7.83       │
│ Projected Daily:        $27.00      │
│ Monthly Projection:     $540.00     │
│                                     │
│ [████████████░░░░░░░░] 29%         │
│                                     │
│ ⚠️ Requires 13.5% daily return     │
│    to break even                   │
└─────────────────────────────────────┘
```

**Alert Thresholds:**
- Green: 0-50% of daily limit
- Yellow: 50-70% of daily limit
- Orange: 70-85% of daily limit
- Red: >85% of daily limit

**File:** `src/lib/components/FeeMonitor.svelte` (NEW)

#### Component 2: Broker Settings Panel

**Location:** Settings → Live Trading section

**Features:**
- List all configured brokers
- Add new broker via form
- Edit existing broker details
- Delete broker
- View monthly volume & rebate tier
- Fee burn warning per broker

**Broker Card Display:**
```
┌─────────────────────────────────────┐
│ Exness Raw Spread            [Edit] │
├─────────────────────────────────────┤
│ Spread:          0.2 pips           │
│ Commission:      $3.50/lot          │
│ Monthly Volume:  45.2 lots          │
│ Rebate Tier:     20%                │
│                                     │
│ ⚠️ Daily fee burn: $27 at 300      │
│    trades/day                       │
└─────────────────────────────────────┘
```

**API Endpoints:**
- `POST /api/brokers/add` - Add new broker
- `PUT /api/brokers/{id}` - Update broker
- `DELETE /api/brokers/{id}` - Remove broker
- `GET /api/brokers/stats` - Get fee statistics

**File:** `src/lib/components/BrokerSettingsPanel.svelte` (NEW)

### 3.2 Priority 2: Paper Trading UI (Week 2)

#### Component 3: Trading Mode Badges

**Location:** Bot cards everywhere

**Visual Design:**
```
┌─────────────────────────────────────┐
│ [PAPER] ICT Scalper     @EURUSD    │  ← Blue dashed border
│ Status: Ready                       │
│ 📊 87 trades  ✅ 68% WR  💰 $234   │
│                                     │
│ [Promote to Demo →]                │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ [DEMO] SMC Reversal     @GBPUSD    │  ← Orange border
│ Status: Active                      │
│ Demo Account: #12345678             │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ [LIVE] AMD Hunter       @GOLD      │  ← Green solid border
│ Status: Trading                     │
│ Live Balance: $250                  │
└─────────────────────────────────────┘
```

**Color Scheme:**
- **PAPER**: Blue (#3b82f6) with dashed border
- **DEMO**: Orange (#f59e0b) with solid border
- **LIVE**: Green (#10b981) with solid border

**File:** Enhancement to `src/lib/components/LiveTradingView.svelte`

#### Component 4: Paper Trading Statistics Dashboard

**Location:** Live Trading → Paper Bots tab

**Displays:**
- All bots in paper/demo mode
- Performance metrics
- Promotion eligibility status
- Time until promotion eligible
- Comparison chart (paper vs demo results)

**Metrics Table:**
```
┌──────────────────────────────────────────────────────────────┐
│ Bot ID        │ Mode  │ Trades │ Win% │ PF   │ DD   │ Days  │
├──────────────────────────────────────────────────────────────┤
│ scalper_01    │ PAPER │ 87     │ 68%  │ 2.1  │ 8%   │ 14    │ ✅
│ smc_reversal  │ PAPER │ 34     │ 58%  │ 1.4  │ 12%  │ 7     │ ⏳
│ amd_hunter    │ DEMO  │ 156    │ 71%  │ 2.8  │ 6%   │ 21    │ ✅
└──────────────────────────────────────────────────────────────┘
```

**File:** `src/lib/components/PaperTradingDashboard.svelte` (NEW)

#### Component 5: Bot Promotion Workflow

**Location:** Bot card action button

**Flow:**
1. User clicks "Promote to Demo" button
2. Modal shows promotion criteria checklist
3. Confirms bot meets all requirements
4. User approves promotion
5. System creates demo account (if needed)
6. Bot switched to demo mode
7. Notification sent

**Promotion Modal:**
```
┌─────────────────────────────────────┐
│ Promote scalper_01 to Demo Mode?   │
├─────────────────────────────────────┤
│ ✅ 87 trades (min: 50)              │
│ ✅ 68% win rate (min: 55%)          │
│ ✅ 2.1 profit factor (min: 1.5)     │
│ ✅ 8% max DD (max: 15%)             │
│ ✅ 14 days active (min: 14)         │
│                                     │
│ [Cancel]  [Promote to Demo →]      │
└─────────────────────────────────────┘
```

**File:** `src/lib/components/BotPromotionModal.svelte` (NEW)

### 3.3 Priority 3: Multi-Timeframe Display (Week 3)

#### Component 6: Multi-Timeframe Regime Grid

**Location:** Strategy Router → Regime tab

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│ Multi-Timeframe Regime: EURUSD                             │
├─────────────────────────────────────────────────────────────┤
│ ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐          │
│ │  M1  │  │  M5  │  │  M15 │  │  H1  │  │  H4  │          │
│ │ HIGH │  │RANGE │  │TREND │  │TREND │  │BREAK │          │
│ │CHAOS │  │STABLE│  │STABLE│  │STABLE│  │ OUT  │          │
│ │ 🔴   │  │ 🔵   │  │ 🟢   │  │ 🟢   │  │ 🟡   │          │
│ │ 0.72 │  │ 0.35 │  │ 0.25 │  │ 0.18 │  │ 0.42 │          │
│ └──────┘  └──────┘  └──────┘  └──────┘  └──────┘          │
└─────────────────────────────────────────────────────────────┘
```

**Color Coding:**
- 🟢 TREND_STABLE: Green
- 🔵 RANGE_STABLE: Blue
- 🟡 BREAKOUT_PRIME: Yellow/Orange
- 🔴 HIGH_CHAOS: Red
- 🟣 NEWS_EVENT: Purple

**Features:**
- Real-time updates every 5 seconds
- Chaos score displayed
- Quality metric displayed
- Click to see detailed regime analysis
- Symbol selector dropdown

**File:** `src/lib/components/MultiTimeframeRegime.svelte` (NEW)

#### Component 7: Timeframe Strategy Matching

**Location:** Strategy Router → Bot Matching tab

**Purpose:** Show which bots are suitable for each timeframe's current regime

**Display:**
```
┌─────────────────────────────────────────────────────────────┐
│ EURUSD Regime-Bot Matching                                 │
├─────────────────────────────────────────────────────────────┤
│ M1 (HIGH_CHAOS 0.72) → ❌ No bots authorized               │
│                                                             │
│ M5 (RANGE_STABLE 0.35) → ✅ Range Scalper                  │
│                            (Score: 7.8)                     │
│                                                             │
│ M15 (TREND_STABLE 0.25) → ✅ ICT Scalper (8.5)             │
│                            ✅ SMC Reversal (7.2)            │
│                                                             │
│ H1 (TREND_STABLE 0.18) → ✅ ICT Macro (9.1)                │
│                            ✅ Structural (8.3)              │
└─────────────────────────────────────────────────────────────┘
```

**File:** `src/lib/components/TimeframeStrategyMatcher.svelte` (NEW)

### 3.4 Priority 4: Progressive Kill Switch UI (Week 1)

#### Component 8: Alert Level Status Display

**Location:** Dashboard header + Kill Switch view

**Alert Levels:**
```
🟢 GREEN ALERT
✅ Continue normal trading
All systems operational

⚡ YELLOW ALERT
⚠️ CAUTION - Monitor closely
• Bot scalper_01: 3/5 consecutive losses
• Daily fee burn: 67% of limit

🟠 ORANGE ALERT
⚠️ REDUCE RISK - Cut position sizes by 50%
• Daily loss at 2.1% (limit: 3%)
• Strategy family: 2 scalpers failing

🔴 RED ALERT
⛔ HALT NEW TRADES - Prepare smart exit
• Daily loss at 2.8% (limit: 3%)
• Weekly loss at 8.5% (limit: 10%)
• 7 active bots (limit: 8)

⚫ BLACK ALERT
🚨 TRIGGER KILL SWITCH
• Global chaos: 0.78 (limit: 0.75)
• CRITICAL: Execute emergency stop immediately
```

**Features:**
- Large, prominent alert banner
- Color-coded background
- Recommended action in bold
- Expandable warning details
- Quick kill switch button (RED/BLACK only)

**File:** Enhancement to `src/lib/components/KillSwitchView.svelte`

#### Component 9: Warning Breakdown Panel

**Location:** Kill Switch view

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│ Warning Breakdown                                           │
├─────────────────────────────────────────────────────────────┤
│ 🤖 Bot Warnings (2)                                         │
│   • scalper_01: 3/5 consecutive losses                     │
│   • smc_reversal: 18/20 daily trades                       │
│                                                             │
│ 📊 Strategy Warnings (1)                                    │
│   • Scalper family: 2/3 bots failing                       │
│                                                             │
│ 💰 Account Warnings (2)                                     │
│   • Daily loss: 2.1% / 3.0% limit                          │
│   • Weekly loss: 8.5% / 10.0% limit                        │
│                                                             │
│ ⏰ Session Warnings (1)                                     │
│   • Trade frequency: 47/50 trades per hour                 │
│                                                             │
│ 🖥️ System Warnings (0)                                      │
│   • All systems operational                                │
└─────────────────────────────────────────────────────────────┘
```

**File:** `src/lib/components/ProgressiveAlertStatus.svelte` (NEW)

---

## 4. Cloud Deployment Strategy - Contabo VPS

### 4.1 Infrastructure Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ GITHUB REPOSITORY                                            │
│ • Main branch (production)                                   │
│ • Dev branch (testing)                                       │
│ • GitHub Actions workflows                                   │
└────────────────────────┬─────────────────────────────────────┘
                         │ Push triggers webhook
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ GITHUB ACTIONS CI/CD                                         │
│ 1. Run tests (pytest)                                        │
│ 2. Deploy via SSH                                            │
│ 3. Restart services                                          │
│ 4. Health check                                              │
└────────────────────────┬─────────────────────────────────────┘
                         │ Deploy to VPS
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ CONTABO VPS (Ubuntu 22.04)                                   │
│                                                              │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ Nginx (Port 80/443)                                    │  │
│ │ • SSL termination (Let's Encrypt)                      │  │
│ │ • Reverse proxy                                        │  │
│ │ • Load balancing                                       │  │
│ └───────┬────────────────────────────────────────────────┘  │
│         │                                                    │
│ ┌───────┴──────────────┐  ┌───────────────────────────────┐ │
│ │ Backend (Port 8000)  │  │ Frontend (Port 3000)          │ │
│ │ • FastAPI            │  │ • SvelteKit                   │ │
│ │ • WebSocket server   │  │ • Static files                │ │
│ │ • Strategy Router    │  │ • UI components               │ │
│ └──────────────────────┘  └───────────────────────────────┘ │
│                                                              │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ PostgreSQL Database                                    │  │
│ │ • Bot registry                                         │  │
│ │ • Trade history                                        │  │
│ │ • Paper trading results                                │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ MT5 Bridge (Wine or Windows VM)                        │  │
│ │ • MetaTrader 5                                         │  │
│ │ • WebSocket bridge to Python                           │  │
│ └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 VPS Setup Checklist

#### Phase 1: Initial Server Configuration (Day 1)

**1.1 SSH Access**
```bash
# From local machine
ssh root@your-contabo-ip

# Create deployment user
adduser quantmind
usermod -aG sudo quantmind
```

**1.2 System Updates**
```bash
apt update && apt upgrade -y
apt install -y \
  python3.11 python3-pip python3-venv \
  nodejs npm \
  git nginx \
  postgresql postgresql-contrib \
  redis-server \
  supervisor \
  htop curl wget \
  ufw fail2ban
```

**1.3 Firewall Configuration**
```bash
ufw allow 22     # SSH
ufw allow 80     # HTTP
ufw allow 443    # HTTPS
ufw allow 8765   # MT5 WebSocket (if needed)
ufw enable
```

**1.4 PostgreSQL Setup**
```bash
sudo -u postgres psql

CREATE DATABASE quantmindx;
CREATE USER quantmind WITH PASSWORD 'secure_password_here';
GRANT ALL PRIVILEGES ON DATABASE quantmindx TO quantmind;
\q
```

**1.5 Project Directory**
```bash
su - quantmind
mkdir -p /home/quantmind/quantmindx
cd /home/quantmind/quantmindx
git clone https://github.com/yourusername/QUANTMINDX.git .
```

**1.6 Python Environment**
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**1.7 Frontend Build**
```bash
cd quantmind-ide
npm install
npm run build
cd ..
```

**1.8 Environment Configuration**
```bash
cp .env.example .env
nano .env  # Edit with production values
```

**Production `.env`:**
```bash
# Database
DATABASE_URL=postgresql://quantmind:secure_password@localhost/quantmindx

# MT5 Connection
MT5_BRIDGE_HOST=localhost  # or windows-vps-ip
MT5_BRIDGE_PORT=8765

# Broker Credentials
EXNESS_LOGIN=your_mt5_login
EXNESS_PASSWORD=your_mt5_password
EXNESS_SERVER=Exness-MT5Real

# API Keys
OPENROUTER_API_KEY=your_openrouter_key
ZHIPU_API_KEY=your_zhipu_key

# Security
SECRET_KEY=generate_random_64_character_string_here
ALLOWED_HOSTS=your-domain.com,your-vps-ip

# Features
PAPER_TRADING_ENABLED=true
PROGRESSIVE_KILL_SWITCH_ENABLED=true

# Environment
ENVIRONMENT=production
DEBUG=false
```

#### Phase 2: Service Management (Day 2)

**2.1 Backend Systemd Service**

Create: `/etc/systemd/system/quantmindx-backend.service`
```ini
[Unit]
Description=QuantMindX Backend API
After=network.target postgresql.service

[Service]
Type=simple
User=quantmind
WorkingDirectory=/home/quantmind/quantmindx
Environment="PATH=/home/quantmind/quantmindx/venv/bin"
ExecStart=/home/quantmind/quantmindx/venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**2.2 Frontend Systemd Service**

Create: `/etc/systemd/system/quantmindx-frontend.service`
```ini
[Unit]
Description=QuantMindX Frontend
After=network.target

[Service]
Type=simple
User=quantmind
WorkingDirectory=/home/quantmind/quantmindx/quantmind-ide
Environment="PORT=3000"
Environment="NODE_ENV=production"
ExecStart=/usr/bin/node build/index.js
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**2.3 Enable Services**
```bash
sudo systemctl daemon-reload
sudo systemctl enable quantmindx-backend
sudo systemctl enable quantmindx-frontend
sudo systemctl start quantmindx-backend
sudo systemctl start quantmindx-frontend

# Check status
sudo systemctl status quantmindx-backend
sudo systemctl status quantmindx-frontend
```

#### Phase 3: Nginx Configuration (Day 3)

**3.1 Nginx Site Configuration**

Create: `/etc/nginx/sites-available/quantmindx`
```nginx
# HTTP → HTTPS redirect
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    return 301 https://$server_name$request_uri;
}

# HTTPS server
server {
    listen 443 ssl http2;
    server_name your-domain.com www.your-domain.com;

    # SSL certificates (Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;

    # Frontend (SvelteKit)
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket (real-time updates)
    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript 
               application/x-javascript application/xml+rss 
               application/json application/javascript;
}
```

**3.2 Enable Site**
```bash
sudo ln -s /etc/nginx/sites-available/quantmindx /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

**3.3 SSL Certificate (Let's Encrypt)**
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Auto-renewal test
sudo certbot renew --dry-run
```

#### Phase 4: CI/CD Setup (Day 4)

**4.1 GitHub Actions Workflow**

Create: `.github/workflows/deploy.yml`
```yaml
name: Deploy to Contabo VPS

on:
  push:
    branches:
      - main  # Production
      - dev   # Testing

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          pytest tests/ -v --cov=src --cov-report=term
      
      - name: Run linting
        run: |
          flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - name: Deploy to VPS
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.VPS_HOST }}
          username: ${{ secrets.VPS_USER }}
          key: ${{ secrets.SSH_PRIVATE_KEY }}
          script: |
            # Navigate to project
            cd /home/quantmind/quantmindx
            
            # Pull latest code
            git pull origin main
            
            # Activate virtual environment
            source venv/bin/activate
            
            # Update Python dependencies
            pip install -r requirements.txt --upgrade
            
            # Run database migrations
            alembic upgrade head
            
            # Rebuild frontend
            cd quantmind-ide
            npm install
            npm run build
            cd ..
            
            # Restart services
            sudo systemctl restart quantmindx-backend
            sudo systemctl restart quantmindx-frontend
            
            # Wait for services to start
            sleep 5
            
            # Health check
            curl -f http://localhost:8000/health || exit 1
            curl -f http://localhost:3000/ || exit 1
            
            echo "✅ Deployment successful"
      
      - name: Notify deployment success
        if: success()
        run: |
          echo "::notice::Deployment to production successful"
      
      - name: Notify deployment failure
        if: failure()
        run: |
          echo "::error::Deployment failed - check logs"
```

**4.2 GitHub Secrets Configuration**

Add these secrets to your repository (Settings → Secrets → Actions):

```
VPS_HOST=your-contabo-ip-address
VPS_USER=quantmind
SSH_PRIVATE_KEY=<paste your SSH private key>
```

**Generate SSH key pair:**
```bash
# On local machine
ssh-keygen -t ed25519 -C "github-actions@quantmindx"

# Copy public key to VPS
ssh-copy-id -i ~/.ssh/id_ed25519.pub quantmind@your-vps-ip

# Copy private key to GitHub secrets
cat ~/.ssh/id_ed25519  # Paste this into SSH_PRIVATE_KEY
```

#### Phase 5: Monitoring & Maintenance (Day 5)

**5.1 Health Check Script**

Create: `/home/quantmind/scripts/health_check.sh`
```bash
#!/bin/bash

LOG_FILE="/var/log/quantmindx/health_check.log"
EMAIL="your@email.com"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> $LOG_FILE
}

# Check backend service
if ! systemctl is-active --quiet quantmindx-backend; then
    log "ERROR: Backend service down, restarting..."
    sudo systemctl restart quantmindx-backend
    echo "QuantMindX Backend was down and restarted" | mail -s "Service Alert" $EMAIL
fi

# Check frontend service
if ! systemctl is-active --quiet quantmindx-frontend; then
    log "ERROR: Frontend service down, restarting..."
    sudo systemctl restart quantmindx-frontend
    echo "QuantMindX Frontend was down and restarted" | mail -s "Service Alert" $EMAIL
fi

# Check API health endpoint
if ! curl -f -s http://localhost:8000/health > /dev/null; then
    log "ERROR: API health check failed"
    sudo systemctl restart quantmindx-backend
    echo "QuantMindX API health check failed" | mail -s "API Alert" $EMAIL
fi

# Check disk space
df -h | grep -vE '^Filesystem|tmpfs|cdrom' | awk '{ print $5 " " $1 }' | while read output;
do
    usage=$(echo $output | awk '{ print $1}' | cut -d'%' -f1)
    partition=$(echo $output | awk '{ print $2 }')
    if [ $usage -ge 90 ]; then
        log "WARNING: Disk space critical on $partition ($usage%)"
        echo "Disk space warning: $partition at $usage%" | mail -s "Disk Alert" $EMAIL
    fi
done

# Check memory usage
mem_usage=$(free | grep Mem | awk '{print ($3/$2) * 100.0}' | cut -d'.' -f1)
if [ $mem_usage -ge 90 ]; then
    log "WARNING: Memory usage at $mem_usage%"
fi

log "Health check completed"
```

**Make executable:**
```bash
chmod +x /home/quantmind/scripts/health_check.sh
```

**5.2 Cron Job Setup**
```bash
crontab -e

# Add these lines:
*/5 * * * * /home/quantmind/scripts/health_check.sh
0 2 * * * /home/quantmind/scripts/backup_database.sh
0 0 * * 0 /usr/bin/certbot renew --quiet
```

**5.3 Database Backup Script**

Create: `/home/quantmind/scripts/backup_database.sh`
```bash
#!/bin/bash

BACKUP_DIR="/home/quantmind/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/quantmindx_$DATE.sql"

# Create backup directory if not exists
mkdir -p $BACKUP_DIR

# Backup database
pg_dump -U quantmind quantmindx > $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE

# Keep only last 30 days of backups
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete

echo "Database backup completed: ${BACKUP_FILE}.gz"
```

**5.4 Log Rotation**

Create: `/etc/logrotate.d/quantmindx`
```
/var/log/quantmindx/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 quantmind quantmind
    sharedscripts
    postrotate
        systemctl reload quantmindx-backend > /dev/null 2>&1 || true
    endscript
}
```

### 4.3 MT5 Bridge Setup

**Option A: Wine (Run MT5 on Linux)**
```bash
# Install Wine
sudo dpkg --add-architecture i386
sudo apt update
sudo apt install wine64 wine32 winetricks

# Configure Wine
winecfg  # Set to Windows 10

# Download MT5
wget https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe

# Install MT5
wine mt5setup.exe

# Run MT5 WebSocket bridge
cd /home/quantmind/quantmindx/mt5-bridge
source ../venv/bin/activate
python server.py &
```

**Option B: Separate Windows VPS (Recommended)**
- More reliable MT5 execution
- Better compatibility
- Easier troubleshooting

**Setup:**
1. Get cheap Windows VPS (separate from Linux VPS)
2. Install MetaTrader 5
3. Run MT5 WebSocket bridge server
4. Connect from Linux VPS via WebSocket
5. Update `.env`: `MT5_BRIDGE_HOST=windows-vps-ip`

### 4.4 Deployment Workflow

**Daily Development Cycle:**
```bash
# Local development
cd /home/mubarkahimself/Desktop/QUANTMINDX

# Create feature branch
git checkout -b feature/fee-aware-kelly
# Make changes...
git add .
git commit -m "feat: implement fee-aware Kelly calculator"
git push origin feature/fee-aware-kelly

# Create pull request on GitHub
# After review, merge to dev branch

# Dev branch auto-deploys to testing environment
git checkout dev
git merge feature/fee-aware-kelly
git push origin dev

# GitHub Actions runs:
# 1. Tests
# 2. Deploys to dev environment
# 3. Sends notification

# After validation in dev, merge to main
git checkout main
git merge dev
git push origin main

# GitHub Actions deploys to production
```

**Zero-Downtime Deployment:**
- Health checks before traffic switch
- Automatic rollback on failure
- Service restarts in sequence
- Database migrations run first

---

## 5. Implementation Timeline

### Week 1: Critical Foundation (Feb 12-18)

**Day 1-2: Fee-Aware System**
- [ ] Create `src/position_sizing/fee_aware_kelly.py`
- [ ] Integrate with BrokerRegistry
- [ ] Update EnhancedGovernor to use fee-aware calculator
- [ ] Add unit tests for fee calculations

**Day 3-4: Progressive Kill Switch**
- [ ] Create `src/router/progressive_kill_switch.py`
- [ ] Create `config/kill_switch_config.yaml`
- [ ] Integrate with existing kill switch
- [ ] Add alert level calculations

**Day 5-7: UI Components (Priority 1)**
- [ ] Build Fee Monitor component
- [ ] Build Broker Settings Panel
- [ ] Build Progressive Alert Status display
- [ ] Add API endpoints for broker management

**Deliverables:**
- ✅ Fee-aware Kelly working
- ✅ Broker settings editable via UI
- ✅ Progressive kill switch operational
- ✅ Real-time fee monitoring

### Week 2: Paper Trading & Deployment (Feb 19-25)

**Day 1-3: Paper Trading System**
- [ ] Create `src/paper_trading/paper_trading_engine.py`
- [ ] Create `src/paper_trading/paper_account.py`
- [ ] Create `src/paper_trading/mt5_demo_manager.py`
- [ ] Add database tables for paper trades
- [ ] Implement promotion eligibility checks

**Day 4-5: VPS Setup**
- [ ] Setup Contabo VPS
- [ ] Install dependencies
- [ ] Configure services
- [ ] Setup Nginx
- [ ] SSL certificates

**Day 6-7: CI/CD & UI**
- [ ] Create GitHub Actions workflow
- [ ] Test automated deployment
- [ ] Build Paper Trading UI components
- [ ] Build mode badges

**Deliverables:**
- ✅ Paper trading system functional
- ✅ VPS deployed and accessible
- ✅ CI/CD pipeline working
- ✅ Bots can trade in paper mode

### Week 3: Advanced Features (Feb 26-Mar 4)

**Day 1-2: Multi-Timeframe Regime**
- [ ] Enhance Sentinel for multi-TF detection
- [ ] Create multi-timeframe data feed
- [ ] Build Multi-Timeframe Regime component
- [ ] Build Timeframe Strategy Matcher

**Day 3-4: Bot Cloning & Session Management**
- [ ] Implement bot cloning system
- [ ] Create session-based market scanner
- [ ] Build market registry (London/NY/Asian)
- [ ] Add session awareness to Strategy Router

**Day 5-7: Testing & Refinement**
- [ ] Deploy first bot in paper mode
- [ ] Monitor for 1 week
- [ ] Collect performance data
- [ ] Refine fee calculations
- [ ] Optimize kill switch thresholds

**Deliverables:**
- ✅ Multi-timeframe regime detection
- ✅ Session-based trading
- ✅ Bot cloning operational
- ✅ First bot promoted to demo

### Week 4+: Live Trading Preparation (Mar 5+)

**Day 1-7: Validation Period**
- [ ] Run 3+ bots in paper mode
- [ ] Promote best performers to demo
- [ ] Monitor demo results daily
- [ ] Collect 2 weeks of data
- [ ] Validate promotion criteria

**Day 8-14: Live Deployment**
- [ ] Promote 1 bot to live mode ($50 capital)
- [ ] Monitor for 1 week
- [ ] Scale to $200-400 if successful
- [ ] Deploy additional bots
- [ ] Iterate and optimize

**Long-term Goals:**
- Run 4-8 bots simultaneously
- Maintain <3% daily loss limit
- Achieve consistent profitability
- Scale capital gradually
- Automate everything

---

## 6. Success Metrics

### Technical Metrics

**System Reliability:**
- [ ] 99.9% uptime on VPS
- [ ] <100ms API response time
- [ ] Zero data loss
- [ ] Automated recovery from failures

**Code Quality:**
- [ ] 80%+ test coverage
- [ ] All tests passing
- [ ] Zero critical bugs
- [ ] Clean linting (flake8/pylint)

**Deployment:**
- [ ] <5 minute deployment time
- [ ] Zero-downtime deployments
- [ ] Automated rollback on failure
- [ ] Health checks pass

### Trading Performance Metrics

**Paper Trading (Week 1-2):**
- [ ] 50+ trades executed
- [ ] Realistic execution simulation
- [ ] Accurate fee calculations
- [ ] Complete data logging

**Demo Trading (Week 3):**
- [ ] 50+ trades on demo account
- [ ] Win rate >55%
- [ ] Profit factor >1.5
- [ ] Max drawdown <15%

**Live Trading (Week 4+):**
- [ ] First bot profitable in live
- [ ] Daily loss <3%
- [ ] Weekly loss <10%
- [ ] Positive expectancy after fees

### Risk Management Metrics

**Kill Switch Effectiveness:**
- [ ] <10 seconds to close all positions
- [ ] No trades executed during RED/BLACK alert
- [ ] Automatic quarantine working
- [ ] Alert progression functional

**Fee Management:**
- [ ] Real-time fee tracking accurate
- [ ] Breakeven calculations correct
- [ ] Fee kill switch triggers appropriately
- [ ] Monthly fee burn <20% of capital

**Account Protection:**
- [ ] No single-day loss >3%
- [ ] No weekly loss >10%
- [ ] Position limits respected
- [ ] Portfolio heat <4%

---

## 7. Next Steps

### Immediate Actions (This Week)

1. **Setup GitHub repository** (if not already)
   - Create `main` and `dev` branches
   - Add `.github/workflows/deploy.yml`
   - Configure secrets

2. **Order Contabo VPS**
   - Choose data center (EU for Exness/RoboForex)
   - Ubuntu 22.04 LTS
   - Minimum 4GB RAM, 2 CPU cores

3. **Create demo accounts**
   - Exness Raw Spread demo
   - RoboForex Prime demo

4. **Start implementing**
   - Begin with fee-aware Kelly calculator
   - Most critical component
   - Blocks all scalping strategies

### Contact & Support

**Development Questions:**
- Check `IMPLEMENTATION_REPORT.md` for technical details
- Review `DATA_ARCHITECTURE_ANALYSIS.md` for system context

**Deployment Issues:**
- Consult GitHub Actions logs
- Check VPS system logs: `/var/log/quantmindx/`
- Review Nginx logs: `/var/log/nginx/`

**Trading Questions:**
- Monitor paper trading results daily
- Review kill switch alert history
- Analyze fee burn reports

---

## Document Version Control

**Version:** 1.0  
**Created:** February 12, 2026  
**Last Updated:** February 12, 2026  
**Status:** Ready for Implementation  
**Author:** QuantMindX Development Team  

**Change Log:**
- v1.0 (2026-02-12): Initial comprehensive proposal based on strategy router enhancement discussion

---

**END OF DOCUMENT**

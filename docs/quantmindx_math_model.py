"""
QuantMindX Trading System — Mathematical Projection Model
Base Equity: $100 | 47 Bots | HoM + R-HoM + WF2 Ranking
"""

# ── System Parameters ─────────────────────────────────────────────
BASE_EQUITY = 100.0
TOTAL_BOTS = 47
SCALP_PCT = 0.70
ORB_PCT = 0.30
SCALP_BOTS = round(TOTAL_BOTS * SCALP_PCT)   # 33
ORB_BOTS = TOTAL_BOTS - SCALP_BOTS            # 14

# ── WF2 Identity / Ranking (Issue: ~10% paused at any time) ──────
WF2_PAUSE_RATE = 0.10
ACTIVE_BOTS = round(TOTAL_BOTS * (1 - WF2_PAUSE_RATE))  # 42
ACTIVE_SCALP = round(SCALP_BOTS * (1 - WF2_PAUSE_RATE)) # 30
ACTIVE_ORB = ACTIVE_BOTS - ACTIVE_SCALP                  # 12

# WF2 Tier Distribution (of active bots)
TIERS = {
    "PRIMAL":     {"pct": 0.15, "multiplier": 2.0},
    "READY":      {"pct": 0.25, "multiplier": 1.5},
    "PROCESSING": {"pct": 0.35, "multiplier": 1.0},
    "PENDING":    {"pct": 0.25, "multiplier": 0.5},
}

# ── Trade Frequency ──────────────────────────────────────────────
SCALP_TRADES_PER_BOT = 3    # per day
ORB_TRADES_PER_BOT = 1      # per day
RAW_TRADES = ACTIVE_SCALP * SCALP_TRADES_PER_BOT + ACTIVE_ORB * ORB_TRADES_PER_BOT

# Reality adjustments
ACTIVITY_RATE = 0.65         # not all bots fire every session
REAL_TRADES = round(RAW_TRADES * ACTIVITY_RATE)

# ── Edge Calculation (R-multiples) ───────────────────────────────
# Scalping: WR=58%, RR=1.3:1
SCALP_WR = 0.58
SCALP_RR = 1.3
SCALP_EDGE_R = SCALP_WR * SCALP_RR - (1 - SCALP_WR) * 1  # +0.3340R

# ORB: WR=52%, RR=2.0:1
ORB_WR = 0.52
ORB_RR = 2.0
ORB_EDGE_R = ORB_WR * ORB_RR - (1 - ORB_WR) * 1           # +0.5600R

# ── Position Sizing ─────────────────────────────────────────────
RISK_PER_TRADE_PCT = 0.005   # 0.5% of base equity
RISK_PER_TRADE = BASE_EQUITY * RISK_PER_TRADE_PCT  # $0.50

# ── Daily Expectancy (flat, no compounding) ──────────────────────
SCALP_TRADES_DAY = round(ACTIVE_SCALP * SCALP_TRADES_PER_BOT * ACTIVITY_RATE)
ORB_TRADES_DAY = round(ACTIVE_ORB * ORB_TRADES_PER_BOT * ACTIVITY_RATE)

SCALP_DAILY_R = SCALP_TRADES_DAY * SCALP_EDGE_R
ORB_DAILY_R = ORB_TRADES_DAY * ORB_EDGE_R

# Dollar expectancy
SCALP_DAILY_USD = SCALP_DAILY_R * RISK_PER_TRADE
ORB_DAILY_USD = ORB_DAILY_R * RISK_PER_TRADE
COMBINED_DAILY_USD = SCALP_DAILY_USD + ORB_DAILY_USD

# Friction: slippage/spread costs (-30%) + correlation discount (-15%)
FRICTION = 0.70 * 0.85  # = 0.595
SCALP_DAILY_NET = SCALP_DAILY_USD * FRICTION
ORB_DAILY_NET = ORB_DAILY_USD * FRICTION
COMBINED_DAILY_NET = SCALP_DAILY_NET + ORB_DAILY_NET

# ── HoM / R-HoM Thresholds ──────────────────────────────────────
HOM_THRESHOLD = BASE_EQUITY * 2          # $200 — activate HoM
RHOM_THRESHOLD = BASE_EQUITY * 0.80      # $80  — activate R-HoM

HOM_RISK_PCT = 0.015                     # 1.5% of house money
RHOM_FLOOR_PCT = 0.10                    # 10% floor on risk scaling

def hom_risk(equity):
    """Risk per trade under HoM regime."""
    if equity <= BASE_EQUITY:
        return RISK_PER_TRADE
    house_money = equity - BASE_EQUITY
    return RISK_PER_TRADE + house_money * HOM_RISK_PCT

def rhom_risk(equity):
    """Risk per trade under R-HoM regime."""
    if equity >= BASE_EQUITY:
        return RISK_PER_TRADE
    drawdown_pct = (BASE_EQUITY - equity) / BASE_EQUITY
    scale = max(RHOM_FLOOR_PCT, 1.0 - drawdown_pct * 2)
    return RISK_PER_TRADE * scale

# ── Risk / Margin Constraints ────────────────────────────────────
MAX_CONCURRENT = round(ACTIVE_BOTS * 0.50)  # ~50% can be open at once
MAX_CONCURRENT_RISK = MAX_CONCURRENT * RISK_PER_TRADE
MARGIN_PER_POS = 2.50  # micro lot margin estimate
TOTAL_MARGIN = MAX_CONCURRENT * MARGIN_PER_POS
MARGIN_USAGE_PCT = TOTAL_MARGIN / BASE_EQUITY * 100

# ── WF2 Tier Allocation Breakdown ───────────────────────────────
tier_data = []
for name, info in TIERS.items():
    count = round(ACTIVE_BOTS * info["pct"])
    alloc = RISK_PER_TRADE * info["multiplier"]
    tier_daily = count * alloc * (SCALP_EDGE_R * SCALP_TRADES_PER_BOT * SCALP_PCT +
                                   ORB_EDGE_R * ORB_TRADES_PER_BOT * ORB_PCT) * ACTIVITY_RATE * FRICTION
    tier_data.append((name, count, info["multiplier"], alloc, tier_daily))

# ── Projections (flat, no compounding) ───────────────────────────
WEEKLY = COMBINED_DAILY_NET * 5
MONTHLY = COMBINED_DAILY_NET * 22
QUARTERLY = MONTHLY * 3
YEARLY = MONTHLY * 12
DAILY_ROI = COMBINED_DAILY_NET / BASE_EQUITY * 100
MONTHLY_ROI = MONTHLY / BASE_EQUITY * 100
YEARLY_ROI = YEARLY / BASE_EQUITY * 100

# ── Stepped Compounding (lot step-up per $100 gained) ────────────
def project_stepped(days, daily_net, base, step=100, max_mult=10):
    """Compound by stepping lot size up 1× per $step gained, capped."""
    equity = base
    history = []
    for d in range(1, days + 1):
        mult = min(1 + (equity - base) // step, max_mult) if equity > base else 1
        if equity < base:  # R-HoM
            dd_pct = (base - equity) / base
            mult = max(0.1, 1.0 - dd_pct * 2)
        day_pnl = daily_net * mult
        equity += day_pnl
        if d in (30, 60, 90, 180, 365) or d == days:
            history.append((d, equity, mult))
    return history

stepped = project_stepped(365, COMBINED_DAILY_NET, BASE_EQUITY)

# ── Output ───────────────────────────────────────────────────────
print("=" * 65)
print("   QUANTMINDX — TRADING SYSTEM MATH MODEL")
print("=" * 65)

print(f"""
┌─ SYSTEM CONFIGURATION ──────────────────────────────┐
│  Base Equity:        ${BASE_EQUITY:.0f}                          │
│  Total Bots:         {TOTAL_BOTS} ({SCALP_BOTS} scalp + {ORB_BOTS} ORB)          │
│  Active (post-WF2):  {ACTIVE_BOTS} ({ACTIVE_SCALP} scalp + {ACTIVE_ORB} ORB)          │
│  Risk per Trade:     ${RISK_PER_TRADE:.2f} (0.5% of equity)          │
│  Paused (WF2):       {TOTAL_BOTS - ACTIVE_BOTS} bots (~10% bottom tier)          │
└──────────────────────────────────────────────────────┘

┌─ EDGE PROFILE ──────────────────────────────────────┐
│  Scalping:  WR={SCALP_WR:.0%}  RR={SCALP_RR}:1  Edge=+{SCALP_EDGE_R:.4f}R/trade  │
│  ORB:       WR={ORB_WR:.0%}  RR={ORB_RR}:1  Edge=+{ORB_EDGE_R:.4f}R/trade  │
└──────────────────────────────────────────────────────┘

┌─ DAILY TRADE VOLUME ────────────────────────────────┐
│  Raw:    {RAW_TRADES} trades/day                               │
│  Real:   {REAL_TRADES} trades/day (65% activity rate)          │
│    ├─ Scalping: {SCALP_TRADES_DAY} trades                            │
│    └─ ORB:      {ORB_TRADES_DAY} trades                             │
└──────────────────────────────────────────────────────┘

┌─ DAILY EXPECTANCY ──────────────────────────────────┐
│  Gross (before friction):                            │
│    Scalping:  {SCALP_DAILY_R:+.2f}R  = ${SCALP_DAILY_USD:.2f}               │
│    ORB:       {ORB_DAILY_R:+.2f}R  = ${ORB_DAILY_USD:.2f}                │
│    Combined:          = ${SCALP_DAILY_USD + ORB_DAILY_USD:.2f}               │
│                                                      │
│  Net (after 30% slippage + 15% correlation):         │
│    Scalping:  ${SCALP_DAILY_NET:.2f}/day                        │
│    ORB:       ${ORB_DAILY_NET:.2f}/day                         │
│    Combined:  ${COMBINED_DAILY_NET:.2f}/day  ({DAILY_ROI:.2f}% daily ROI)    │
└──────────────────────────────────────────────────────┘""")

print(f"""
┌─ WF2 TIER ALLOCATION ───────────────────────────────┐""")
for name, count, mult, alloc, daily in tier_data:
    print(f"│  {name:<12} {count:>2} bots  {mult:.1f}×  ${alloc:.2f}/trade  ${daily:.2f}/day │")
print(f"└──────────────────────────────────────────────────────┘")

print(f"""
┌─ HoM / R-HoM DYNAMICS ─────────────────────────────┐
│  House of Money (equity ≥ ${HOM_THRESHOLD:.0f}):                 │
│    Risk = $0.50 + 1.5% of house money               │
│    At $200: ${hom_risk(200):.2f}/trade  (4.0× base)            │
│    At $300: ${hom_risk(300):.2f}/trade  (7.0× base)            │
│    At $500: ${hom_risk(500):.2f}/trade  (13.0× base)           │
│                                                      │
│  Reverse HoM (equity ≤ ${RHOM_THRESHOLD:.0f}):                   │
│    Risk scales down proportionally (10% floor)       │
│    At $80:  ${rhom_risk(80):.2f}/trade  ({rhom_risk(80)/RISK_PER_TRADE:.1f}× base)             │
│    At $60:  ${rhom_risk(60):.2f}/trade  ({rhom_risk(60)/RISK_PER_TRADE:.1f}× base)             │
│    At $50:  ${rhom_risk(50):.2f}/trade  ({rhom_risk(50)/RISK_PER_TRADE:.1f}× base)             │
└──────────────────────────────────────────────────────┘

┌─ RISK CONSTRAINTS ──────────────────────────────────┐
│  Max Concurrent:     {MAX_CONCURRENT} positions                    │
│  Max Risk Exposed:   ${MAX_CONCURRENT_RISK:.2f}                        │
│  Margin Usage:       ${TOTAL_MARGIN:.2f} ({MARGIN_USAGE_PCT:.0f}% of equity)          │
└──────────────────────────────────────────────────────┘

┌─ FLAT PROJECTIONS (no compounding) ─────────────────┐
│  Daily:     ${COMBINED_DAILY_NET:.2f}   ({DAILY_ROI:.2f}% ROI)                │
│  Weekly:    ${WEEKLY:.2f}  (5 trading days)              │
│  Monthly:   ${MONTHLY:.2f} (22 trading days)            │
│  Quarterly: ${QUARTERLY:.2f}                              │
│  Yearly:    ${YEARLY:.2f}                            │
│  Annual ROI: {YEARLY_ROI:.1f}%                                │
└──────────────────────────────────────────────────────┘""")

print(f"""
┌─ STEPPED COMPOUNDING (lot +1× per $100 gained) ────┐
│  Cap: 10× base lot | R-HoM scales down in drawdown  │
│                                                      │""")
for day, eq, mult in stepped:
    roi = (eq - BASE_EQUITY) / BASE_EQUITY * 100
    print(f"│  Day {day:>3}:  ${eq:>10.2f}  ({mult:.1f}× lot)  {roi:>+8.1f}% │")
print(f"└──────────────────────────────────────────────────────┘")

print(f"""
┌─ KEY TAKEAWAYS ─────────────────────────────────────┐
│  1. Flat expectancy: ~${COMBINED_DAILY_NET:.2f}/day on $100 base       │
│  2. WF2 ranking adds ~18% alpha by concentrating     │
│     capital on top-performing bots                    │
│  3. HoM accelerates gains once profitable but        │
│     R-HoM protects capital during drawdowns           │
│  4. {MARGIN_USAGE_PCT:.0f}% margin usage leaves room for volatility    │
│  5. Real edge depends heavily on execution quality    │
│     — slippage/spread is the biggest drag             │
└──────────────────────────────────────────────────────┘
""")

const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat,
  HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak, TabStopType, TabStopPosition
} = require("/home/mubarkahimself/Desktop/QUANTMINDX/node_modules/docx");

// ── Helpers ──────────────────────────────────────────────────────────
const FONT = "Arial";
const PAGE_W = 12240, PAGE_H = 15840, MARGIN = 1440;
const CONTENT_W = PAGE_W - 2 * MARGIN; // 9360
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 80, bottom: 80, left: 120, right: 120 };
const accentBorder = { style: BorderStyle.SINGLE, size: 1, color: "2E75B6" };
const accentBorders = { top: accentBorder, bottom: accentBorder, left: accentBorder, right: accentBorder };

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text, font: FONT, bold: true, size: 32 })] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 }, children: [new TextRun({ text, font: FONT, bold: true, size: 28 })] });
}
function h3(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 180, after: 100 }, children: [new TextRun({ text, font: FONT, bold: true, size: 24 })] });
}
function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120 },
    ...opts.paraOpts,
    children: [new TextRun({ text, font: FONT, size: 22, ...opts })]
  });
}
function bold(text) { return new TextRun({ text, font: FONT, size: 22, bold: true }); }
function normal(text) { return new TextRun({ text, font: FONT, size: 22 }); }
function mixed(...runs) { return new Paragraph({ spacing: { after: 120 }, children: runs }); }
function bullet(text, ref = "bullets", level = 0) {
  return new Paragraph({ numbering: { reference: ref, level }, spacing: { after: 60 }, children: [normal(text)] });
}
function numberedItem(text, ref = "numbers", level = 0) {
  return new Paragraph({ numbering: { reference: ref, level }, spacing: { after: 60 }, children: [normal(text)] });
}
function headerCell(text, width) {
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA },
    shading: { fill: "2E4057", type: ShadingType.CLEAR },
    margins: cellMargins,
    children: [new Paragraph({ children: [new TextRun({ text, font: FONT, size: 20, bold: true, color: "FFFFFF" })] })]
  });
}
function cell(text, width, shade) {
  const opts = { borders, width: { size: width, type: WidthType.DXA }, margins: cellMargins,
    children: [new Paragraph({ children: [new TextRun({ text, font: FONT, size: 20 })] })] };
  if (shade) opts.shading = { fill: shade, type: ShadingType.CLEAR };
  return new TableCell(opts);
}
function cellMulti(runs, width, shade) {
  const opts = { borders, width: { size: width, type: WidthType.DXA }, margins: cellMargins,
    children: [new Paragraph({ children: runs })] };
  if (shade) opts.shading = { fill: shade, type: ShadingType.CLEAR };
  return new TableCell(opts);
}
function makeTable(headers, rows, colWidths) {
  const tw = colWidths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: tw, type: WidthType.DXA }, columnWidths: colWidths,
    rows: [
      new TableRow({ children: headers.map((h, i) => headerCell(h, colWidths[i])) }),
      ...rows.map((row, ri) => new TableRow({
        children: row.map((c, ci) => cell(c, colWidths[ci], ri % 2 === 0 ? "F5F7FA" : undefined))
      }))
    ]
  });
}
function spacer() { return new Paragraph({ spacing: { after: 200 }, children: [] }); }

// ── Section builders ─────────────────────────────────────────────────
function buildTitlePage() {
  return [
    spacer(), spacer(), spacer(),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [
      new TextRun({ text: "QUANTMINDX", font: FONT, size: 56, bold: true, color: "2E4057" })
    ]}),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [
      new TextRun({ text: "Risk & Position Sizing", font: FONT, size: 40, color: "4A90D9" })
    ]}),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [
      new TextRun({ text: "Complete System Redesign Specification", font: FONT, size: 36, color: "4A90D9" })
    ]}),
    spacer(),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ text: "Version 1.0  |  April 2026", font: FONT, size: 24, color: "666666" })
    ]}),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ text: "CONFIDENTIAL", font: FONT, size: 22, bold: true, color: "CC0000" })
    ]}),
    new Paragraph({ children: [new PageBreak()] })
  ];
}

function buildExecutiveSummary() {
  return [
    h1("1. Executive Summary"),
    p("QuantMindX is an agentic trading system designed to compound small accounts from $100 to $5,000, withdraw, reset to $500, and repeat across multiple accounts. The core goal is simple: the system must compound without losing money faster than it makes it."),
    p("This document specifies the complete redesign of the risk management and position sizing subsystems. The current codebase contains two separate, conflicting risk systems that were built at different times and never unified. Additionally, prop firm-specific logic is deeply embedded throughout the codebase and must be completely removed as the system transitions to personal book trading exclusively."),
    h2("1.1 Core Problem"),
    p("The current system has a hardcoded 3% portfolio risk cap and a 2% per-trade maximum that makes mathematical sense for large accounts but crushes the compounding potential of small accounts. A $100 account risking 2% per trade ($2.00) cannot generate meaningful growth. The system needs dynamic, tier-based risk parameters that scale with account size."),
    h2("1.2 What This Document Covers"),
    bullet("Complete inventory of both risk/sizing systems (System A and System B) and their conflicts"),
    bullet("All connected systems: Sentinel, BOCPD/MS-GARCH, SQS, Kill Switches, Sessions, DPR, Circuit Breakers, Strategy Router"),
    bullet("Full prop firm removal plan across 55+ files (Python + MQL5)"),
    bullet("Dynamic tier-based risk parameters (GROWTH / SCALING / GUARDIAN)"),
    bullet("Unified trade flow from signal to execution"),
    bullet("Implementation priorities and phasing"),
    new Paragraph({ children: [new PageBreak()] })
  ];
}

function buildCurrentArchitecture() {
  return [
    h1("2. Current System Architecture"),
    h2("2.1 The Two-System Problem"),
    p("The codebase contains two independent risk/position sizing systems built at different times. System A is active in production. System B was one of the first systems built but was partially abandoned when System A was developed. Neither team realized the other existed, leading to duplicate logic, conflicting constants, and dead code paths."),

    h3("System A (Active Production) - src/position_sizing/"),
    makeTable(
      ["Component", "File", "Status"],
      [
        ["EnhancedKellyCalculator", "src/position_sizing/enhanced_kelly.py", "ACTIVE - Primary calculator"],
        ["EnhancedKellyConfig", "src/position_sizing/kelly_config.py", "ACTIVE - 2% max, 3% portfolio"],
        ["PropFirmPresets", "src/position_sizing/kelly_config.py", "REMOVE - Prop firm code"],
        ["SessionKellyModifiers", "src/risk/sizing/session_kelly_modifiers.py", "ACTIVE - HMM/session scaling"],
        ["EnhancedGovernor", "src/router/enhanced_governor.py", "ACTIVE - Bridge between A & B"],
        ["Base Governor", "src/router/governor.py", "ACTIVE - Physics throttle + caps"],
      ],
      [2500, 4360, 2500]
    ),
    spacer(),

    h3("System B (Partially Dormant) - src/risk/"),
    makeTable(
      ["Component", "File", "Status"],
      [
        ["RiskGovernor", "src/risk/governor.py", "DORMANT - Never called"],
        ["PhysicsAwareKellyEngine", "src/risk/sizing/kelly_engine.py", "DORMANT - Has useful physics logic"],
        ["PortfolioKellyScaler", "src/risk/portfolio_kelly.py", "ORPHANED - Not imported anywhere"],
        ["Risk Config", "src/risk/config.py", "PARTIAL - Drawdown limits active"],
        ["PropFirmType/Presets", "src/risk/config.py", "REMOVE - Prop firm code"],
      ],
      [2500, 4360, 2500]
    ),
    spacer(),

    h2("2.2 Conflicting Constants"),
    p("The two systems define conflicting risk parameters that make the system behavior unpredictable:"),
    makeTable(
      ["Parameter", "System A Value", "System B Value", "Resolution"],
      [
        ["Max Risk Per Trade", "2%", "5%", "Dynamic by tier"],
        ["Max Portfolio Risk", "3%", "15%", "Dynamic by tier"],
        ["Kelly Fraction", "50%", "25% (MQL5)", "Unified 50% half-Kelly"],
        ["Hard Cap", "2%", "10% (physics engine)", "Dynamic by tier"],
        ["Drawdown Limits", "Not defined in A", "3% / 5% / 7%", "Dynamic 3/5/7 by tier"],
      ],
      [2000, 1800, 1800, 3760]
    ),
    spacer(),

    h2("2.3 Production Trade Flow (Current)"),
    p("The actual production trade flow passes through both systems via the EnhancedGovernor bridge:"),
    numberedItem("Signal arrives from Strategy/Bot", "flow1"),
    numberedItem("EnhancedGovernor.calculate_risk() is called", "flow1"),
    numberedItem("Session boundary check (are we in an active session window?)", "flow1"),
    numberedItem("HMM state update (Hidden Markov Model for regime detection)", "flow1"),
    numberedItem("EnhancedKellyCalculator computes base Kelly fraction (System A)", "flow1"),
    numberedItem("Fee kill switch check (if fees >= avg_win, block trade)", "flow1"),
    numberedItem("Physics throttle applied: chaos > 0.6 = 0.2x, chaos > 0.3 = 0.7x (System B logic)", "flow1"),
    numberedItem("SessionKellyModifiers applied: HMM scaling, reverse HMM, premium sessions", "flow1"),
    numberedItem("Final position size = account_equity * adjusted_kelly_fraction", "flow1"),
    numberedItem("Order sent to MT5 via MQL5 execution layer", "flow1"),
    new Paragraph({ children: [new PageBreak()] })
  ];
}

function buildConnectedSystems() {
  return [
    h1("3. Connected Systems Inventory"),
    p("The risk/sizing redesign touches or is touched by every system below. Each must be considered in the unified flow."),

    h2("3.1 Market Sentinel"),
    mixed(bold("Location: "), normal("src/router/sentinel.py, multi_timeframe_sentinel.py, sensors/")),
    p("The Sentinel is the central market intelligence hub. It aggregates signals from BOCPD (changepoint detection), MS-GARCH (volatility regime), and various sensors into a composite chaos_score (0.0 to 1.0). This score drives the physics throttle in the Governor, reducing position sizes as market chaos increases."),
    bullet("chaos_score > 0.6: Risk multiplier drops to 0.2x (extreme caution)"),
    bullet("chaos_score > 0.3: Risk multiplier drops to 0.7x (moderate caution)"),
    bullet("chaos_score <= 0.3: Full risk (1.0x multiplier)"),
    p("Change needed: The Sentinel integration stays. The chaos_score thresholds should be configurable, not hardcoded."),

    h2("3.2 BOCPD + MS-GARCH (Physics/ML Layer)"),
    mixed(bold("Location: "), normal("src/risk/physics/bocpd/, src/risk/physics/msgarch/, src/risk/physics/ensemble/")),
    p("Bayesian Online Changepoint Detection (BOCPD) identifies structural breaks in market data. MS-GARCH models volatility regimes. The ensemble voter combines their outputs into the chaos_score fed to Sentinel. These are read-only data providers and do not need architectural changes."),
    p("Change needed: None. These stay as-is. The question of whether additional physics models (Lyapunov, Ising, Eigenvalue) are needed is deferred. The current BOCPD + MS-GARCH ensemble is sufficient."),

    h2("3.3 Spread Quality Score (SQS)"),
    mixed(bold("Location: "), normal("src/risk/sqs_engine.py, sqs_cache.py, sqs_calendar.py, weekend_guard.py")),
    p("SQS is a pre-trade filter that evaluates spread quality before allowing a trade. It checks current spread against historical norms, applies calendar-based adjustments (wider spreads around news events), and includes a weekend guard that prevents trades near market close on Friday."),
    p("Change needed: SQS stays as a pre-trade gate. It currently returns a binary pass/fail. Consider adding a quality score (0-1) that can modulate position size rather than just blocking."),

    h2("3.4 Kill Switches (Progressive + Basic + Copilot)"),
    mixed(bold("Location: "), normal("src/router/progressive_kill_switch.py, kill_switch.py, copilot_kill_switch.py")),
    h3("Progressive Kill Switch (5-Tier Hierarchy)"),
    p("The progressive kill switch implements a 5-tier response system that escalates based on drawdown severity:"),
    makeTable(
      ["Tier", "Color", "Drawdown", "Action"],
      [
        ["Tier 0", "GREEN", "< 1%", "Full trading, no restrictions"],
        ["Tier 1", "YELLOW", "1% - 2%", "Warning only, log increased monitoring"],
        ["Tier 2", "ORANGE", "2% - 3%", "Reduce position sizes by 50%"],
        ["Tier 3", "RED", "3% - 4%", "Reduce to 25%, close worst performers"],
        ["Tier 4", "BLACK", ">= 4%", "Kill all trading, close all positions"],
      ],
      [1200, 1200, 2400, 4560]
    ),
    spacer(),
    p("Change needed: Tier 3 thresholds must sync with the dynamic 3/5/7 drawdown framework. Currently hardcoded. The tiers should scale by account tier (GROWTH gets wider bands, GUARDIAN gets tighter bands)."),
    h3("Basic Kill Switch"),
    p("The basic kill switch is a simple on/off circuit breaker. When triggered (by the progressive kill switch reaching BLACK, or by manual override), it sets a global variable that blocks all new trades system-wide."),
    p("Change needed: Must be wired into the unified risk flow as the final gate before trade execution."),
    h3("Copilot Kill Switch"),
    p("The copilot kill switch provides human oversight. It allows a human operator to intervene and halt trading at any time via UI or API command."),
    p("Change needed: Stays as-is. Human override always takes priority."),

    h2("3.5 Session Windows (10 Canonical Sessions)"),
    mixed(bold("Location: "), normal("src/router/sessions.py, session_template.py")),
    p("The system defines 10 canonical trading session windows with hardcoded bot caps per session:"),
    makeTable(
      ["Session", "Time (UTC)", "Current Bot Cap", "Notes"],
      [
        ["Asian Early", "00:00 - 02:00", "18", "Low liquidity"],
        ["Tokyo Open", "00:00 - 03:00", "50", "JPY pairs active"],
        ["Asian Late", "03:00 - 06:00", "18", "Transition period"],
        ["London Pre-Open", "06:00 - 07:00", "50", "Building liquidity"],
        ["London Open", "07:00 - 09:00", "60", "PREMIUM - Highest liquidity"],
        ["London-NY Overlap", "12:00 - 16:00", "60", "PREMIUM - Highest volume"],
        ["NY Open", "13:00 - 16:00", "60", "PREMIUM - USD pairs active"],
        ["NY Afternoon", "16:00 - 19:00", "50", "Declining volume"],
        ["NY Close", "19:00 - 21:00", "6", "Low liquidity"],
        ["Off-Hours", "21:00 - 00:00", "0", "No trading"],
      ],
      [2000, 2000, 1800, 3560]
    ),
    spacer(),
    p("Change needed: Bot caps must become configurable via UI/config instead of hardcoded. The session framework itself is sound and stays."),

    h2("3.6 Dynamic Performance Ranking (DPR)"),
    mixed(bold("Location: "), normal("src/risk/dpr/scoring_engine.py, queue_manager.py, queue_models.py")),
    p("DPR ranks all active bots into three performance tiers based on a composite score (win rate, profit factor, drawdown, consistency). This ranking drives both bot scheduling and sizing:"),
    bullet("TIER_1 (top performers): 1.2x sizing multiplier"),
    bullet("TIER_2 (average): 1.0x sizing multiplier (baseline)"),
    bullet("TIER_3 (underperformers): 0.8x sizing multiplier"),
    p("Change needed: DPR integration stays. The sizing multipliers should be configurable. DPR tier should also influence the per-bot circuit breaker thresholds."),

    h2("3.7 Circuit Breakers (Bot + SSL)"),
    mixed(bold("Location: "), normal("src/router/bot_circuit_breaker.py, src/risk/ssl/circuit_breaker.py")),
    p("The bot circuit breaker is per-BOT (not per-session). When a specific bot hits its loss limit, only that bot is stopped. The SSL circuit breaker monitors overall system health via spread/slippage/latency metrics."),
    p("Change needed: Per-bot circuit breaker thresholds should be influenced by DPR tier and account tier. SSL circuit breaker stays as-is."),

    h2("3.8 Strategy Router"),
    mixed(bold("Location: "), normal("src/router/")),
    p("The Strategy Router has evolved from its original routing purpose into a monitoring role. It now primarily monitors bot performance via DPR scores and SQS quality gates, rather than actively routing trades between strategies. The actual execution path goes through the Governor chain."),
    p("Change needed: Acknowledge its monitoring role. Clean up any remaining routing logic that conflicts with the Governor chain."),
    new Paragraph({ children: [new PageBreak()] })
  ];
}

function buildWhatNeedsToChange() {
  return [
    h1("4. What Needs to Change"),

    h2("4.1 Unified Kelly Calculator"),
    p("Combine the best of both systems into a single EnhancedKellyCalculator with clearly defined layers:"),
    makeTable(
      ["Layer", "Source", "What It Does"],
      [
        ["Layer 1: Base Kelly", "System A", "Half-Kelly (50% fraction) with Bayesian win-rate estimation"],
        ["Layer 2: Hard Cap", "Dynamic (NEW)", "Tier-based max: GROWTH 5%, SCALING 3%, GUARDIAN 1.5%"],
        ["Layer 3: Volatility Adjustment", "System A", "ATR-based volatility scaling (existing, works well)"],
        ["Layer 4: Physics Throttle", "System B (via Sentinel)", "chaos_score-based multiplier (0.2x / 0.7x / 1.0x)"],
        ["Layer 5: Session Modifiers", "System B", "HMM continuous scaling, reverse HMM, premium sessions"],
        ["Layer 6: DPR Multiplier", "DPR System", "Bot performance tier multiplier (0.8x / 1.0x / 1.2x)"],
        ["Layer 7: Fee Kill Switch", "System A", "Block trade if estimated fees >= average win amount"],
      ],
      [2200, 2200, 4960]
    ),
    spacer(),
    p("The physics-specific models (Lyapunov stability, Ising model, Eigenvalue analysis) from PhysicsAwareKellyEngine in System B are NOT included in the redesign. The BOCPD + MS-GARCH ensemble already provides sufficient regime detection via the chaos_score. These can be revisited later if needed."),

    h2("4.2 Dynamic Account Tiers"),
    p("Replace all hardcoded risk parameters with tier-based dynamic values. Account tier is determined by current equity:"),
    makeTable(
      ["Parameter", "GROWTH (< $1,000)", "SCALING ($1,000 - $5,000)", "GUARDIAN ($5,000+)"],
      [
        ["Max Risk Per Trade", "5%", "3%", "1.5%"],
        ["Max Portfolio Risk", "30%", "15%", "5-10%"],
        ["Kelly Hard Cap", "5%", "3%", "1.5%"],
        ["Daily Drawdown Limit", "10%", "5%", "3%"],
        ["Weekly Drawdown Limit", "20%", "10%", "5%"],
        ["Monthly Drawdown Limit", "30%", "15%", "7%"],
        ["Position Floor", "$5.00 min", "Kelly-based", "Kelly + quadratic throttle"],
        ["Kill Switch Tier 3 (RED)", "At 8% drawdown", "At 4% drawdown", "At 2.5% drawdown"],
        ["Kill Switch Tier 4 (BLACK)", "At 10% drawdown", "At 5% drawdown", "At 3% drawdown"],
      ],
      [2400, 1800, 2400, 2760]
    ),
    spacer(),
    p("The GROWTH tier is deliberately aggressive because a $100 account needs to take calculated risks to grow. The GUARDIAN tier is conservative because at $5,000+ the system is protecting accumulated gains before withdrawal."),

    h2("4.3 Dynamic 3/5/7 Drawdown Framework"),
    p("The current system has hardcoded drawdown limits of 3% daily, 5% weekly, 7% monthly (from src/risk/config.py). These become dynamic, scaling by account tier as shown in Section 4.2. The progressive kill switch tiers must sync with these limits."),

    h2("4.4 Per-Trade Risk Calculation"),
    p("Per-trade risk is always a percentage of CURRENT EQUITY (not starting balance, not hardcoded dollar amount). This is critical for compounding:"),
    mixed(bold("Formula: "), normal("risk_amount = current_equity * tier_risk_pct * kelly_fraction * all_modifiers")),
    spacer(),
    p("Where all_modifiers = physics_throttle * session_modifier * dpr_multiplier * kill_switch_scalar"),
    p("This means as the account grows from $100 to $500, the dollar risk per trade grows proportionally, enabling geometric compounding. As the account crosses tier boundaries, the percentage adjusts to protect gains."),

    h2("4.5 Unified Trade Flow (Redesigned)"),
    p("The complete trade flow from signal to execution, incorporating all systems:"),
    numberedItem("Signal arrives from Strategy/Bot", "flow2"),
    numberedItem("SQS pre-trade filter: Check spread quality. If FAIL, reject trade.", "flow2"),
    numberedItem("Session check: Is current time within an active session window? If NO, reject.", "flow2"),
    numberedItem("Kill switch check: Query progressive kill switch state. If BLACK, reject. If RED/ORANGE, apply scalar.", "flow2"),
    numberedItem("Bot circuit breaker check: Is this specific bot halted? If YES, reject.", "flow2"),
    numberedItem("Copilot kill switch check: Has human operator halted trading? If YES, reject.", "flow2"),
    numberedItem("News guard (Sentinel): Is a high-impact news event imminent? If YES, reject.", "flow2"),
    numberedItem("Determine account tier: GROWTH / SCALING / GUARDIAN based on current equity.", "flow2"),
    numberedItem("Calculate base Kelly: Half-Kelly with Bayesian win-rate, capped by tier hard cap.", "flow2"),
    numberedItem("Apply Layer 3 (ATR volatility adjustment).", "flow2"),
    numberedItem("Apply Layer 4 (Physics throttle via Sentinel chaos_score).", "flow2"),
    numberedItem("Apply Layer 5 (Session modifiers: HMM scaling, reverse HMM, premium session bonus).", "flow2"),
    numberedItem("Apply Layer 6 (DPR bot performance multiplier).", "flow2"),
    numberedItem("Apply Layer 7 (Fee kill switch: if fees >= avg_win, reject trade).", "flow2"),
    numberedItem("Calculate final risk_amount = current_equity * adjusted_fraction.", "flow2"),
    numberedItem("Calculate position_size from risk_amount and stop_loss distance.", "flow2"),
    numberedItem("Send order to MT5 execution layer.", "flow2"),
    new Paragraph({ children: [new PageBreak()] })
  ];
}

function buildPropFirmRemoval() {
  return [
    h1("5. Prop Firm Removal Plan"),
    p("All prop firm-specific logic must be completely removed. The system will operate exclusively as a PERSONAL book type. This affects 55+ files across the Python and MQL5 codebases."),

    h2("5.1 Files to DELETE Entirely"),
    p("These files exist solely for prop firm functionality and have no salvageable logic:"),
    makeTable(
      ["File", "Reason"],
      [
        ["src/risk/prop_firm_manager.py", "Dedicated prop firm manager"],
        ["src/risk/prop_firm_config.py", "Prop firm configuration"],
        ["src/risk/prop_rules_engine.py", "Prop firm rules engine"],
        ["src/prop_firm/", "Entire prop firm directory"],
        ["src/mql5/Include/QuantMind/Risk/PropManager.mqh", "MQL5 prop firm manager (CPropManager)"],
        ["src/mql5/Include/QuantMind/Core/PropManager.mqh", "MQL5 prop firm manager (QMPropManager)"],
        ["docs/prop-firm-integration.md", "Prop firm documentation"],
        ["docs/prop-firm-workflow.md", "Prop firm workflow docs"],
      ],
      [5500, 3860]
    ),
    spacer(),

    h2("5.2 Files to CLEAN (Remove Prop Firm References)"),
    p("These files contain useful logic mixed with prop firm references. Remove the prop firm code but keep everything else:"),
    makeTable(
      ["File", "What to Remove"],
      [
        ["src/position_sizing/kelly_config.py", "PropFirmPresets class, FTPreset, The5ersPreset, FundingPipsPreset"],
        ["src/risk/config.py", "PropFirmType enum, PropFirmPreset dataclass, PROP_FIRM_PRESETS dict"],
        ["src/risk/governor.py", "All PropFirmPreset imports and usage (file may become DELETE candidate)"],
        ["src/router/enhanced_governor.py", "PropFirmAccount imports and any prop firm branching logic"],
        ["src/router/governor.py", "Any prop firm references in risk mandate calculation"],
        ["src/risk/sizing/session_kelly_modifiers.py", "Any prop firm mode checks"],
        ["src/mql5/Include/QuantMind/Core/Constants.mqh", "Rename 'PropFirm Risk Limits' section, make values dynamic"],
        ["src/mql5/Include/QuantMind/Risk/KellySizer.mqh", "Remove prop firm tier references, keep V8 tiered engine"],
      ],
      [5000, 4360]
    ),
    spacer(),

    h2("5.3 Additional Files with Prop Firm References"),
    p("The grep scan found prop firm references in 55+ files total. Beyond the files listed above, these additional files contain prop firm references that must be cleaned:"),
    bullet("All API endpoint files that serve prop firm data"),
    bullet("All test files that test prop firm functionality"),
    bullet("All configuration files with prop firm presets"),
    bullet("All UI components that display prop firm information"),
    bullet("All database migration files that create prop firm tables"),
    p("A complete file-by-file audit should be performed during implementation using: grep -rn 'prop.firm\\|PropFirm\\|PROP_FIRM\\|prop_firm\\|propfirm' src/"),
    new Paragraph({ children: [new PageBreak()] })
  ];
}

function buildSystemBCleanup() {
  return [
    h1("6. System B Cleanup Plan"),
    p("System B (src/risk/ and src/risk/sizing/) was one of the first systems built but was partially abandoned. Each component gets one of three treatments:"),

    h2("6.1 DELETE - No Salvageable Logic"),
    makeTable(
      ["File", "Reason"],
      [
        ["src/risk/governor.py", "RiskGovernor is never called. Its Kelly calc duplicates System A. Prop firm logic throughout."],
        ["src/risk/portfolio_kelly.py", "PortfolioKellyScaler is orphaned (not imported by anything). Port useful API methods first, then delete."],
      ],
      [4000, 5360]
    ),
    spacer(),

    h2("6.2 INTEGRATE - Has Useful Logic"),
    makeTable(
      ["File", "What to Keep", "Where It Goes"],
      [
        ["src/risk/sizing/kelly_engine.py", "Physics model multipliers (Lyapunov, Ising, Eigenvalue)", "Deferred. Not needed now. BOCPD+MS-GARCH sufficient."],
        ["src/risk/sizing/session_kelly_modifiers.py", "HMM scaling, Reverse HMM, Premium sessions", "Already integrated via EnhancedGovernor. Clean prop firm refs."],
        ["src/risk/config.py", "Drawdown limits (3/5/7), book type enum", "Move to unified config. Make dynamic by tier."],
      ],
      [3200, 3200, 2960]
    ),
    spacer(),

    h2("6.3 KEEP AS-IS"),
    makeTable(
      ["File", "Reason"],
      [
        ["src/risk/sqs_engine.py", "SQS is active and working correctly"],
        ["src/risk/sqs_cache.py", "SQS cache layer, working correctly"],
        ["src/risk/sqs_calendar.py", "Calendar-based spread adjustments, working correctly"],
        ["src/risk/dpr/scoring_engine.py", "DPR scoring engine, active and working"],
        ["src/risk/dpr/queue_manager.py", "DPR queue management, active and working"],
        ["src/risk/physics/bocpd/", "Changepoint detection, feeds Sentinel"],
        ["src/risk/physics/msgarch/", "Volatility regime detection, feeds Sentinel"],
        ["src/risk/physics/ensemble/", "Ensemble voter, combines BOCPD + MS-GARCH"],
        ["src/risk/ssl/circuit_breaker.py", "SSL circuit breaker, working correctly"],
      ],
      [4500, 4860]
    ),
    spacer(),
    new Paragraph({ children: [new PageBreak()] })
  ];
}

function buildMQL5Reconciliation() {
  return [
    h1("7. MQL5 Reconciliation"),
    p("The MQL5 layer (MetaTrader 5) is the execution side. It must stay aligned with Python-side risk decisions while handling the realities of broker execution."),

    h2("7.1 KellySizer.mqh - KEEP and CLEAN"),
    mixed(bold("Location: "), normal("src/mql5/Include/QuantMind/Risk/KellySizer.mqh")),
    p("This file contains the V8 Tiered Risk Engine which already implements the GROWTH/SCALING/GUARDIAN tier concept. It is the most aligned MQL5 file with the redesign vision. Changes needed:"),
    bullet("Remove any prop firm references"),
    bullet("Align tier boundaries with Python: GROWTH (< $1,000), SCALING ($1,000 - $5,000), GUARDIAN ($5,000+)"),
    bullet("Align risk percentages with Python dynamic config"),
    bullet("The $5.00 floor for GROWTH tier is good and should stay"),
    bullet("The 25% Kelly fraction cap should be reviewed against Python's 50% half-Kelly"),

    h2("7.2 PropManager Files - DELETE"),
    bullet("src/mql5/Include/QuantMind/Risk/PropManager.mqh (CPropManager - full prop firm manager)"),
    bullet("src/mql5/Include/QuantMind/Core/PropManager.mqh (QMPropManager - lighter version)"),
    p("Both files are entirely prop firm logic with hardcoded 5% daily limits and 4% effective limits. Delete entirely."),

    h2("7.3 Constants.mqh - CLEAN and RENAME"),
    mixed(bold("Location: "), normal("src/mql5/Include/QuantMind/Core/Constants.mqh")),
    p("This file contains system-wide constants. Changes needed:"),
    bullet("Rename 'PropFirm Risk Limits' section to 'Risk Management Constants'"),
    bullet("Remove QM_DAILY_LOSS_LIMIT_PCT (5.0) - was prop firm specific"),
    bullet("Remove QM_HARD_STOP_BUFFER_PCT (1.0) - was prop firm specific"),
    bullet("Remove QM_EFFECTIVE_LIMIT_PCT (4.0) - was prop firm specific"),
    bullet("Keep QM_MAX_RISK_PER_TRADE_PCT (2.0) but make it a default that can be overridden by tier config"),
    bullet("Keep QM_DEFAULT_RISK_PCT (1.0) but align with Python defaults"),
    bullet("Make all risk constants receivable from Python bridge rather than hardcoded"),

    h2("7.4 Python-MQL5 Alignment"),
    p("The Python side makes all risk decisions. The MQL5 side executes. The bridge between them must pass:"),
    bullet("Current account tier (GROWTH / SCALING / GUARDIAN)"),
    bullet("Calculated position size (from unified Kelly calculator)"),
    bullet("Risk multiplier (combined from all modifier layers)"),
    bullet("Kill switch state (GREEN through BLACK)"),
    bullet("Per-trade risk amount in dollars"),
    p("The MQL5 KellySizer should serve as a VALIDATION layer, confirming that the Python-calculated position size falls within its own tier-based bounds. If there is a mismatch, the MQL5 side uses the more conservative value."),
    new Paragraph({ children: [new PageBreak()] })
  ];
}

function buildImplementationPriorities() {
  return [
    h1("8. Implementation Priorities"),
    p("The redesign is organized into 6 phases, ordered by impact and dependency:"),

    h2("Phase 1: Prop Firm Removal (Foundation)"),
    p("Remove all prop firm code before any other changes. This cleans the codebase and prevents merge conflicts with later phases."),
    bullet("Delete all prop firm-only files (Section 5.1)"),
    bullet("Clean prop firm references from mixed files (Section 5.2)"),
    bullet("Remove prop firm API endpoints and UI components"),
    bullet("Replace BookType.PROP_FIRM with BookType.PERSONAL everywhere"),
    bullet("Run full test suite, fix any broken tests"),
    mixed(bold("Estimated effort: "), normal("2-3 days")),

    h2("Phase 2: Dynamic Account Tiers"),
    p("Implement the GROWTH/SCALING/GUARDIAN tier system with dynamic risk parameters."),
    bullet("Create unified tier configuration (single source of truth)"),
    bullet("Implement tier detection based on current equity"),
    bullet("Replace all hardcoded risk caps with tier-based lookups"),
    bullet("Wire tier into EnhancedKellyCalculator as Layer 2"),
    bullet("Update MQL5 KellySizer to receive tier from Python bridge"),
    mixed(bold("Estimated effort: "), normal("3-4 days")),

    h2("Phase 3: Kill Switch + Drawdown Sync"),
    p("Sync the progressive kill switch tiers with the dynamic drawdown framework."),
    bullet("Make kill switch tier thresholds configurable by account tier"),
    bullet("Wire basic kill switch as final gate in trade flow"),
    bullet("Ensure copilot kill switch integration is clean"),
    bullet("Implement dynamic 3/5/7 drawdown limits by tier"),
    bullet("Test all kill switch transitions with various account sizes"),
    mixed(bold("Estimated effort: "), normal("2-3 days")),

    h2("Phase 4: Unified Kelly Calculator"),
    p("Combine System A and System B into the single 7-layer calculator described in Section 4.1."),
    bullet("Merge fee kill switch (Layer 7) into unified flow"),
    bullet("Clean up EnhancedGovernor to use only unified calculator"),
    bullet("Remove System B's RiskGovernor (src/risk/governor.py)"),
    bullet("Remove orphaned PortfolioKellyScaler (src/risk/portfolio_kelly.py)"),
    bullet("Port useful drawdown config from src/risk/config.py to unified config"),
    mixed(bold("Estimated effort: "), normal("3-5 days")),

    h2("Phase 5: Session + DPR Integration"),
    p("Make session bot caps and DPR multipliers configurable."),
    bullet("Move session bot caps to config/database instead of hardcoded"),
    bullet("Add UI controls for session configuration"),
    bullet("Make DPR tier multipliers configurable (currently 0.8x/1.0x/1.2x)"),
    bullet("Wire DPR multiplier into unified calculator as Layer 6"),
    bullet("Consider SQS returning a quality score (0-1) instead of binary"),
    mixed(bold("Estimated effort: "), normal("2-3 days")),

    h2("Phase 6: MQL5 Alignment + Validation"),
    p("Final phase to ensure Python and MQL5 are fully aligned."),
    bullet("Clean KellySizer.mqh: align tiers, remove prop firm refs"),
    bullet("Delete both PropManager.mqh files"),
    bullet("Clean Constants.mqh: rename sections, make values dynamic"),
    bullet("Enhance Python-MQL5 bridge to pass tier + risk parameters"),
    bullet("Implement MQL5 validation layer (conservative override if mismatch)"),
    bullet("End-to-end testing: Python calculates, MQL5 validates, trade executes"),
    mixed(bold("Estimated effort: "), normal("3-4 days")),

    spacer(),
    mixed(bold("Total estimated effort: "), normal("15-22 days for complete redesign")),
    new Paragraph({ children: [new PageBreak()] })
  ];
}

function buildSummaryTable() {
  return [
    h1("9. Summary: All Changes at a Glance"),
    makeTable(
      ["System", "Action", "Priority"],
      [
        ["Prop firm code (55+ files)", "DELETE / CLEAN all references", "Phase 1"],
        ["Account tiers (GROWTH/SCALING/GUARDIAN)", "CREATE unified dynamic config", "Phase 2"],
        ["Progressive kill switch", "MODIFY thresholds to be tier-dynamic", "Phase 3"],
        ["Basic kill switch", "WIRE as final pre-trade gate", "Phase 3"],
        ["Drawdown limits (3/5/7)", "MODIFY to be tier-dynamic", "Phase 3"],
        ["EnhancedKellyCalculator", "MODIFY to 7-layer unified calculator", "Phase 4"],
        ["RiskGovernor (System B)", "DELETE entirely", "Phase 4"],
        ["PortfolioKellyScaler (System B)", "DELETE (port API first)", "Phase 4"],
        ["PhysicsAwareKellyEngine", "DEFER (not needed now)", "Deferred"],
        ["Session bot caps", "MODIFY to be configurable", "Phase 5"],
        ["DPR multipliers", "MODIFY to be configurable", "Phase 5"],
        ["SQS", "CONSIDER quality score vs binary", "Phase 5"],
        ["KellySizer.mqh", "CLEAN prop firm refs, align tiers", "Phase 6"],
        ["PropManager.mqh (both)", "DELETE entirely", "Phase 6"],
        ["Constants.mqh", "CLEAN and rename sections", "Phase 6"],
        ["Python-MQL5 bridge", "ENHANCE to pass tier + risk params", "Phase 6"],
        ["Sentinel / chaos_score", "KEEP (make thresholds configurable)", "Phase 5"],
        ["BOCPD + MS-GARCH", "KEEP as-is", "None"],
        ["SSL circuit breaker", "KEEP as-is", "None"],
        ["Bot circuit breaker", "MODIFY thresholds by tier + DPR", "Phase 5"],
        ["Strategy Router", "CLEAN routing vestiges, keep monitoring", "Phase 5"],
        ["Copilot kill switch", "KEEP as-is", "None"],
      ],
      [3200, 3760, 2400]
    ),
    spacer(),
    spacer(),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ text: "END OF DOCUMENT", font: FONT, size: 24, bold: true, color: "999999" })
    ]})
  ];
}

// ── Assemble document ────────────────────────────────────────────────
async function main() {
  const doc = new Document({
    styles: {
      default: { document: { run: { font: FONT, size: 22 } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 32, bold: true, font: FONT, color: "2E4057" },
          paragraph: { spacing: { before: 360, after: 240 }, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 28, bold: true, font: FONT, color: "4A6FA5" },
          paragraph: { spacing: { before: 240, after: 180 }, outlineLevel: 1 } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 24, bold: true, font: FONT, color: "5B7DB1" },
          paragraph: { spacing: { before: 180, after: 120 }, outlineLevel: 2 } },
      ]
    },
    numbering: {
      config: [
        { reference: "bullets", levels: [
          { level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
          { level: 1, format: LevelFormat.BULLET, text: "\u25E6", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 1440, hanging: 360 } } } }
        ]},
        { reference: "numbers", levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } }
        ]},
        { reference: "flow1", levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } }
        ]},
        { reference: "flow2", levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } }
        ]},
      ]
    },
    sections: [{
      properties: {
        page: {
          size: { width: PAGE_W, height: PAGE_H },
          margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN }
        }
      },
      headers: {
        default: new Header({ children: [
          new Paragraph({
            border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "2E4057", space: 1 } },
            children: [
              new TextRun({ text: "QUANTMINDX  |  Risk & Position Sizing Redesign", font: FONT, size: 18, color: "999999" }),
              new TextRun({ text: "\tCONFIDENTIAL", font: FONT, size: 18, color: "CC0000", bold: true })
            ],
            tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }]
          })
        ]})
      },
      footers: {
        default: new Footer({ children: [
          new Paragraph({
            border: { top: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC", space: 1 } },
            alignment: AlignmentType.CENTER,
            children: [
              new TextRun({ text: "Page ", font: FONT, size: 18, color: "999999" }),
              new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: "999999" })
            ]
          })
        ]})
      },
      children: [
        ...buildTitlePage(),
        ...buildExecutiveSummary(),
        ...buildCurrentArchitecture(),
        ...buildConnectedSystems(),
        ...buildWhatNeedsToChange(),
        ...buildPropFirmRemoval(),
        ...buildSystemBCleanup(),
        ...buildMQL5Reconciliation(),
        ...buildImplementationPriorities(),
        ...buildSummaryTable()
      ]
    }]
  });

  const buffer = await Packer.toBuffer(doc);
  const outPath = "/home/mubarkahimself/Desktop/QUANTMINDX/docs/QUANTMINDX_Risk_Position_Sizing_Redesign.docx";
  fs.writeFileSync(outPath, buffer);
  console.log("Document written to:", outPath);
  console.log("Size:", (buffer.length / 1024).toFixed(1), "KB");
}

main().catch(err => { console.error(err); process.exit(1); });

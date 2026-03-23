<script lang="ts">
  /**
   * Portfolio Canvas
   *
   * Portfolio Department Head workspace.
   * Dense CRM-style tile grid, tab navigation, canvas-local sub-page routing,
   *
   * Tabs: Overview | Accounts | Attribution | Journal | Risk Exposure
   * Accent: --color-accent-cyan (#00d4ff) — portfolio dept
   * Aesthetic: Frosted Terminal glass
   *
   * Svelte 5 runes only — no `export let`.
   */
  import { onMount } from 'svelte';
  import { canvasContextService } from '$lib/services/canvasContextService';
  import { apiFetch } from '$lib/api';
  import {
    portfolioStore,
    accounts,
    portfolioSummary,
    portfolioLoading
  } from '$lib/stores/portfolio';
  import TradingJournal from '$lib/components/portfolio/TradingJournal.svelte';
  import AttributionPanel from '$lib/components/portfolio/AttributionPanel.svelte';
  import AgentTilePanel from '$lib/components/AgentTilePanel.svelte';
  import {
    Briefcase,
    Wallet,
    TrendingUp,
    TrendingDown,
    BarChart3,
    ShieldAlert,
    BookOpen,
    Activity,
    Users,
    LineChart,
    ArrowLeft,
    DollarSign,
    Percent,
    Target,
    AlertTriangle,
    RefreshCw,
    CircleDot,
    Lightbulb,
    ChevronDown
  } from 'lucide-svelte';

  // =============================================================================
  // Types
  // =============================================================================

  type PortfolioTab = 'overview' | 'accounts' | 'attribution' | 'journal' | 'risk-exposure';
  type SubPage = 'grid' | 'journal-detail';

  interface AccountRow {
    account_id: string;
    broker_name: string;
    balance: number;
    equity: number;
    open_trades: number;
    pnl_today: number;
    currency: string;
    connected: boolean;
    account_type: string;
    server: string;
  }

  // =============================================================================
  // State
  // =============================================================================

  let activeTab = $state<PortfolioTab>('overview');
  let currentSubPage = $state<SubPage>('grid');

  // Agent insights strip
  let insightsExpanded = $state(false);
  let insightsUnread = $state(0);

  // Accounts tab state
  let accountsData = $state<AccountRow[]>([]);
  let accountsLoading = $state(false);

  // =============================================================================
  // Tab config
  // =============================================================================

  const tabs: { id: PortfolioTab; label: string; icon: typeof Briefcase }[] = [
    { id: 'overview',      label: 'Overview',      icon: Briefcase  },
    { id: 'accounts',      label: 'Accounts',      icon: Wallet     },
    { id: 'attribution',   label: 'Attribution',   icon: BarChart3  },
    { id: 'journal',       label: 'Journal',       icon: BookOpen   },
    { id: 'risk-exposure', label: 'Risk Exposure', icon: ShieldAlert },
  ];

  // =============================================================================
  // Lifecycle
  // =============================================================================

  onMount(async () => {
    try {
      await canvasContextService.loadCanvasContext('portfolio');
    } catch {
      // canvas context is optional
    }
    await portfolioStore.initialize();
  });

  // =============================================================================
  // Tab switching
  // =============================================================================

  async function handleTabChange(tab: PortfolioTab) {
    activeTab = tab;
    currentSubPage = 'grid';
    if (tab === 'accounts') {
      await loadAccounts();
    }
  }

  // =============================================================================
  // Accounts data
  // =============================================================================

  async function loadAccounts() {
    if (accountsLoading) return;
    accountsLoading = true;
    try {
      const data = await apiFetch<AccountRow[]>('/portfolio/accounts');
      accountsData = data;
    } catch {
      // Fall back to the portfolio store's accounts as demo data
      accountsData = ($accounts as AccountRow[]);
    } finally {
      accountsLoading = false;
    }
  }

  // =============================================================================
  // Formatting helpers
  // =============================================================================

  function fmtCurrency(n: number, currency = 'USD'): string {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency,
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(n);
  }

  function fmtPct(n: number): string {
    return `${n >= 0 ? '+' : ''}${n.toFixed(2)}%`;
  }

  function pnlClass(n: number): string {
    return n > 0 ? 'positive' : n < 0 ? 'negative' : 'neutral';
  }

  // =============================================================================
  // Derived overview metrics (from store)
  // =============================================================================

  const overviewMetrics = $derived([
    {
      id: 'aum',
      label: 'Total AUM',
      value: $portfolioSummary ? fmtCurrency($portfolioSummary.totalEquity) : '—',
      icon: DollarSign,
      color: 'cyan'
    },
    {
      id: 'pnl',
      label: 'Daily P&L',
      value: $portfolioSummary ? fmtCurrency($portfolioSummary.dailyPnL) : '—',
      icon: TrendingUp,
      color: $portfolioSummary && $portfolioSummary.dailyPnL >= 0 ? 'green' : 'red'
    },
    {
      id: 'win-rate',
      label: 'Win Rate',
      value: '62.5%',
      icon: Target,
      color: 'cyan'
    },
    {
      id: 'max-dd',
      label: 'Max Drawdown',
      value: $portfolioSummary ? fmtPct(-$portfolioSummary.drawdownPercent) : '—',
      icon: TrendingDown,
      color: 'red'
    },
    {
      id: 'sharpe',
      label: 'Sharpe Ratio',
      value: '1.82',
      icon: LineChart,
      color: 'cyan'
    },
    {
      id: 'open-pos',
      label: 'Open Positions',
      value: '—',
      icon: Activity,
      color: 'amber'
    },
    {
      id: 'active-accts',
      label: 'Active Accounts',
      value: String($accounts.filter(a => a.connected).length),
      icon: Users,
      color: 'cyan'
    },
    {
      id: 'recent',
      label: 'Recent Activity',
      value: 'See Journal',
      icon: CircleDot,
      color: 'amber',
      clickable: true,
      onclick: () => handleTabChange('journal')
    }
  ]);
</script>

<div class="portfolio-canvas" data-dept="portfolio">

  <!-- ==========================================================================
       Canvas Header
       ========================================================================== -->
  <header class="canvas-header">
    <div class="header-left">
      {#if currentSubPage !== 'grid'}
        <button class="back-btn" onclick={() => (currentSubPage = 'grid')}>
          <ArrowLeft size={13} />
          <span>Back</span>
        </button>
      {/if}
      <Briefcase size={17} class="dept-icon" />
      <h1 class="canvas-title">Portfolio</h1>
      <span class="dept-badge">Portfolio Dept</span>
    </div>

    {#if currentSubPage === 'grid'}
      <nav class="tab-nav">
        {#each tabs as tab}
          <button
            class="tab-btn"
            class:active={activeTab === tab.id}
            onclick={() => handleTabChange(tab.id)}
          >
            <svelte:component this={tab.icon} size={12} />
            <span>{tab.label}</span>
          </button>
        {/each}
      </nav>
    {/if}
  </header>

  <!-- ==========================================================================
       Canvas Body
       ========================================================================== -->
  <div class="canvas-body">

    <!-- Journal sub-page -->
    {#if currentSubPage === 'journal-detail'}
      <div class="subpage-wrapper">
        <TradingJournal onClose={() => (currentSubPage = 'grid')} />
      </div>

    {:else if activeTab === 'journal'}
      <!-- Journal tab renders TradingJournal inline -->
      <div class="journal-wrapper">
        <TradingJournal onClose={() => handleTabChange('overview')} />
      </div>

    {:else if activeTab === 'overview'}
      <!-- ---- Overview: 8-tile grid ---- -->
      <div class="tile-grid">
        {#if $portfolioLoading}
          {#each Array(8) as _, i (i)}
            <div class="tile skeleton">
              <div class="skeleton-line short"></div>
              <div class="skeleton-line long"></div>
            </div>
          {/each}
        {:else}
          {#each overviewMetrics as metric}
            <button
              class="tile metric-tile accent-{metric.color}"
              class:clickable={metric.clickable}
              onclick={metric.onclick ?? undefined}
              disabled={!metric.clickable}
            >
              <div class="tile-header">
                <svelte:component this={metric.icon} size={14} class="tile-icon" />
                <span class="tile-label">{metric.label}</span>
              </div>
              <span class="tile-value accent-{metric.color}">{metric.value}</span>
            </button>
          {/each}
        {/if}
      </div>

    {:else if activeTab === 'accounts'}
      <!-- ---- Accounts: per-broker tiles ---- -->
      <div class="section-toolbar">
        <span class="section-label">Broker Accounts</span>
        <button class="refresh-btn" onclick={loadAccounts} disabled={accountsLoading}>
          <span class:spin={accountsLoading} style="display:inline-flex"><RefreshCw size={13} /></span>
          <span>Refresh</span>
        </button>
      </div>

      {#if accountsLoading}
        <div class="tile-grid">
          {#each Array(4) as _, i (i)}
            <div class="tile skeleton">
              <div class="skeleton-line short"></div>
              <div class="skeleton-line long"></div>
              <div class="skeleton-line medium"></div>
            </div>
          {/each}
        </div>
      {:else if accountsData.length === 0 && $accounts.length === 0}
        <div class="empty-state">
          <Wallet size={28} />
          <span>No broker accounts registered yet</span>
        </div>
      {:else}
        <div class="tile-grid">
          {#each (accountsData.length ? accountsData : ($accounts as AccountRow[])) as acct}
            <div class="tile account-tile">
              <div class="account-header">
                <span class="broker-name">{acct.broker_name}</span>
                <span class="conn-dot" class:connected={acct.connected}></span>
              </div>
              <div class="account-type-label">{acct.account_type ?? ''}</div>
              <div class="account-metrics">
                <div class="acct-metric">
                  <span class="acct-label">Balance</span>
                  <span class="acct-value cyan">{fmtCurrency(acct.balance, acct.currency)}</span>
                </div>
                <div class="acct-metric">
                  <span class="acct-label">Equity</span>
                  <span class="acct-value cyan">{fmtCurrency(acct.equity, acct.currency)}</span>
                </div>
                <div class="acct-metric">
                  <span class="acct-label">Trades</span>
                  <span class="acct-value">{acct.open_trades ?? 0}</span>
                </div>
                <div class="acct-metric">
                  <span class="acct-label">P&L Today</span>
                  <span class="acct-value {pnlClass(acct.pnl_today ?? acct.equity - acct.balance)}">
                    {fmtCurrency((acct.pnl_today ?? acct.equity - acct.balance), acct.currency)}
                  </span>
                </div>
              </div>
              <div class="account-footer">
                <span class="acct-id">{acct.account_id}</span>
                <span class="acct-server">{acct.server}</span>
              </div>
            </div>
          {/each}
        </div>
      {/if}

    {:else if activeTab === 'attribution'}
      <!-- ---- Attribution: strategy/symbol/session tiles + panel ---- -->
      <div class="section-toolbar">
        <span class="section-label">P&L Attribution</span>
      </div>
      <div class="attribution-layout">
        <!-- Top row: quick stat tiles -->
        <div class="tile-grid attribution-tiles">
          <div class="tile stat-tile">
            <div class="tile-header">
              <BarChart3 size={13} class="tile-icon" />
              <span class="tile-label">Strategy Attribution</span>
            </div>
            <span class="tile-sub">P&L per EA · see table below</span>
          </div>
          <div class="tile stat-tile">
            <div class="tile-header">
              <LineChart size={13} class="tile-icon" />
              <span class="tile-label">Symbol Attribution</span>
            </div>
            <span class="tile-sub">XAUUSD, EURUSD, …</span>
          </div>
          <div class="tile stat-tile">
            <div class="tile-header">
              <Activity size={13} class="tile-icon" />
              <span class="tile-label">Session Attribution</span>
            </div>
            <span class="tile-sub">London, NY, Asian</span>
          </div>
          <div class="tile stat-tile">
            <div class="tile-header">
              <Percent size={13} class="tile-icon" />
              <span class="tile-label">Correlation Matrix</span>
            </div>
            <span class="tile-sub">Strategy cross-correlations</span>
          </div>
        </div>
        <!-- Full attribution table -->
        <div class="attribution-panel-wrapper">
          <AttributionPanel />
        </div>
      </div>

    {:else if activeTab === 'risk-exposure'}
      <!-- ---- Risk Exposure: stub tiles ---- -->
      <div class="section-toolbar">
        <span class="section-label">Risk Exposure</span>
        <span class="stub-badge">Wiring Pending</span>
      </div>
      <div class="tile-grid">
        <div class="tile risk-tile">
          <div class="tile-header">
            <AlertTriangle size={13} class="tile-icon amber" />
            <span class="tile-label">Value at Risk (VaR)</span>
          </div>
          <span class="tile-value stub">—</span>
          <span class="tile-sub">95% confidence · 1-day</span>
        </div>
        <div class="tile risk-tile">
          <div class="tile-header">
            <LineChart size={13} class="tile-icon amber" />
            <span class="tile-label">Portfolio Beta</span>
          </div>
          <span class="tile-value stub">—</span>
          <span class="tile-sub">vs. benchmark</span>
        </div>
        <div class="tile risk-tile">
          <div class="tile-header">
            <Activity size={13} class="tile-icon amber" />
            <span class="tile-label">Net Exposure</span>
          </div>
          <span class="tile-value stub">—</span>
          <span class="tile-sub">Long / Short delta</span>
        </div>
        <div class="tile risk-tile">
          <div class="tile-header">
            <DollarSign size={13} class="tile-icon amber" />
            <span class="tile-label">Currency Risk</span>
          </div>
          <span class="tile-value stub">—</span>
          <span class="tile-sub">USD / EUR / GBP exposure</span>
        </div>
      </div>
    {/if}

  </div>

  <!-- =========================================================
       Agent Insights strip (always visible, collapsible)
       ========================================================= -->
  <div class="agent-insights-strip" class:expanded={insightsExpanded}>
    <button class="insights-toggle" onclick={() => { insightsExpanded = !insightsExpanded; }}>
      <Lightbulb size={12} />
      <span>Agent Insights</span>
      {#if insightsUnread > 0}
        <span class="unread-badge">{insightsUnread}</span>
      {/if}
      <span class:rotated={insightsExpanded} style="display:inline-flex;align-items:center;transition:transform 0.2s;"><ChevronDown size={12} /></span>
    </button>
    {#if insightsExpanded}
      <AgentTilePanel
        canvas="portfolio"
        maxHeight="200px"
        showHeader={false}
        onUnreadCount={(n: number) => { insightsUnread = n; }}
      />
    {/if}
  </div>

</div>

<style>
  /* ==========================================================================
     Shell layout
     ========================================================================== */

  .portfolio-canvas {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: rgba(10, 15, 26, 0.92);
    backdrop-filter: blur(16px) saturate(160%);
    -webkit-backdrop-filter: blur(16px) saturate(160%);
    color: #e0e0e0;
    font-family: 'JetBrains Mono', monospace;
    overflow: hidden;
  }

  /* ==========================================================================
     Canvas Header
     ========================================================================== */

  .canvas-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 20px 10px;
    border-bottom: 1px solid rgba(0, 212, 255, 0.15);
    flex-shrink: 0;
    flex-wrap: wrap;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
  }

  :global(.dept-icon) {
    color: #00d4ff;
  }

  .canvas-title {
    font-size: 18px;
    font-weight: 700;
    color: #00d4ff;
    margin: 0;
    letter-spacing: 0.02em;
  }

  .dept-badge {
    padding: 2px 8px;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.3);
    border-radius: 4px;
    font-size: 10px;
    color: #00d4ff;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .back-btn {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 5px 10px;
    background: rgba(0, 212, 255, 0.08);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 5px;
    color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
  }

  .back-btn:hover {
    background: rgba(0, 212, 255, 0.15);
    border-color: rgba(0, 212, 255, 0.4);
  }

  /* ==========================================================================
     Tab Navigation
     ========================================================================== */

  .tab-nav {
    display: flex;
    gap: 4px;
    align-items: center;
    flex-wrap: wrap;
    margin-left: auto;
  }

  .tab-btn {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 5px 11px;
    background: rgba(0, 212, 255, 0.05);
    border: 1px solid rgba(0, 212, 255, 0.12);
    border-radius: 5px;
    color: rgba(224, 224, 224, 0.6);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }

  .tab-btn:hover {
    background: rgba(0, 212, 255, 0.1);
    border-color: rgba(0, 212, 255, 0.3);
    color: #e0e0e0;
  }

  .tab-btn.active {
    background: rgba(0, 212, 255, 0.16);
    border-color: rgba(0, 212, 255, 0.5);
    color: #00d4ff;
  }

  /* ==========================================================================
     Canvas Body
     ========================================================================== */

  .canvas-body {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    transition: padding-bottom 0.25s ease;
    min-height: 0;
  }


  /* ==========================================================================
     Tile Grid
     ========================================================================== */

  .tile-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 14px;
  }

  /* Base tile — Frosted Terminal glass */
  .tile {
    background: rgba(8, 13, 20, 0.35);
    backdrop-filter: blur(12px) saturate(160%);
    -webkit-backdrop-filter: blur(12px) saturate(160%);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 8px;
    padding: 16px;
    cursor: default;
    text-align: left;
    font-family: 'JetBrains Mono', monospace;
    color: #e0e0e0;
    transition: border-color 0.15s, transform 0.12s;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .tile.clickable {
    cursor: pointer;
  }

  .tile.clickable:hover,
  .metric-tile:not([disabled]):hover {
    border-color: rgba(0, 212, 255, 0.4);
    transform: translateY(-1px);
  }

  button.tile {
    background: rgba(8, 13, 20, 0.35);
    cursor: pointer;
  }

  button.tile:disabled {
    cursor: default;
  }

  button.tile:not([disabled]):hover {
    border-color: rgba(0, 212, 255, 0.4);
    transform: translateY(-1px);
  }

  /* ==========================================================================
     Tile internals — header, label, value
     ========================================================================== */

  .tile-header {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  :global(.tile-icon) {
    color: rgba(0, 212, 255, 0.7);
    flex-shrink: 0;
  }

  :global(.tile-icon.amber) {
    color: rgba(245, 158, 11, 0.8);
  }

  .tile-label {
    font-size: 11px;
    color: rgba(224, 224, 224, 0.55);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .tile-value {
    font-size: 22px;
    font-weight: 700;
    color: #e0e0e0;
    line-height: 1;
  }

  .tile-value.accent-cyan  { color: #00d4ff; }
  .tile-value.accent-green { color: #00c896; }
  .tile-value.accent-red   { color: #ff4d4d; }
  .tile-value.accent-amber { color: #f59e0b; }
  .tile-value.stub         { color: rgba(224, 224, 224, 0.25); }

  .tile-sub {
    font-size: 10px;
    color: rgba(224, 224, 224, 0.35);
  }

  /* ==========================================================================
     Skeleton loader tiles
     ========================================================================== */

  .tile.skeleton {
    animation: pulse 1.4s ease-in-out infinite;
    cursor: default;
  }

  .skeleton-line {
    height: 10px;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.06);
  }

  .skeleton-line.short  { width: 45%; }
  .skeleton-line.medium { width: 65%; }
  .skeleton-line.long   { width: 85%; height: 20px; }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.45; }
  }

  /* ==========================================================================
     Account tiles
     ========================================================================== */

  .account-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
  }

  .broker-name {
    font-size: 14px;
    font-weight: 600;
    color: #00d4ff;
  }

  .conn-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #555;
    flex-shrink: 0;
    margin-top: 3px;
  }

  .conn-dot.connected {
    background: #00c896;
    box-shadow: 0 0 6px rgba(0, 200, 150, 0.5);
  }

  .account-type-label {
    font-size: 10px;
    color: rgba(224, 224, 224, 0.35);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .account-metrics {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px 16px;
  }

  .acct-metric {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .acct-label {
    font-size: 9px;
    color: rgba(224, 224, 224, 0.4);
    text-transform: uppercase;
  }

  .acct-value {
    font-size: 13px;
    color: #e0e0e0;
    font-weight: 600;
  }

  .acct-value.cyan     { color: #00d4ff; }
  .acct-value.positive { color: #00c896; }
  .acct-value.negative { color: #ff4d4d; }
  .acct-value.neutral  { color: rgba(224, 224, 224, 0.5); }

  .account-footer {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: #444;
    margin-top: 4px;
  }

  .acct-id     { color: #555; }
  .acct-server { color: #444; }

  /* ==========================================================================
     Section toolbar
     ========================================================================== */

  .section-toolbar {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
  }

  .section-label {
    font-size: 11px;
    color: rgba(224, 224, 224, 0.45);
    text-transform: uppercase;
    letter-spacing: 0.07em;
  }

  .stub-badge {
    padding: 2px 7px;
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.2);
    border-radius: 4px;
    font-size: 10px;
    color: rgba(245, 158, 11, 0.6);
  }

  .refresh-btn {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    background: rgba(0, 212, 255, 0.06);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 5px;
    color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
  }

  .refresh-btn:hover:not([disabled]) {
    background: rgba(0, 212, 255, 0.12);
    border-color: rgba(0, 212, 255, 0.35);
  }

  .refresh-btn:disabled {
    opacity: 0.45;
    cursor: default;
  }

  /* ==========================================================================
     Attribution layout
     ========================================================================== */

  .attribution-layout {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  .attribution-tiles {
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  }

  .attribution-panel-wrapper {
    background: rgba(8, 13, 20, 0.35);
    backdrop-filter: blur(12px) saturate(160%);
    border: 1px solid rgba(0, 212, 255, 0.12);
    border-radius: 8px;
    overflow: hidden;
  }

  /* ==========================================================================
     Risk Exposure tiles
     ========================================================================== */

  .risk-tile .tile-value.stub {
    font-size: 28px;
  }

  /* ==========================================================================
     Empty state
     ========================================================================== */

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 60px 20px;
    color: rgba(224, 224, 224, 0.3);
    font-size: 13px;
  }

  /* ==========================================================================
     Journal wrapper
     ========================================================================== */

  .journal-wrapper,
  .subpage-wrapper {
    height: 100%;
    display: flex;
    flex-direction: column;
  }


  /* Spin animation for loaders */
  :global(.spin) {
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  /* ==========================================================================
     Agent Insights strip
     ========================================================================== */

  .agent-insights-strip {
    flex-shrink: 0;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    background: rgba(8, 13, 20, 0.85);
  }

  .insights-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    width: 100%;
    padding: 6px 16px;
    background: transparent;
    border: none;
    color: rgba(224, 224, 224, 0.45);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    transition: color 0.15s, background 0.15s;
  }

  .insights-toggle:hover {
    color: rgba(224, 224, 224, 0.75);
    background: rgba(255, 255, 255, 0.03);
  }

  .agent-insights-strip.expanded .insights-toggle {
    color: #00d4ff;
  }

  .unread-badge {
    padding: 1px 6px;
    background: rgba(0, 212, 255, 0.15);
    border: 1px solid rgba(0, 212, 255, 0.35);
    border-radius: 10px;
    font-size: 10px;
    color: #00d4ff;
    font-weight: 700;
    line-height: 1.4;
  }

  .insights-toggle .rotated {
    transform: rotate(180deg);
  }
</style>

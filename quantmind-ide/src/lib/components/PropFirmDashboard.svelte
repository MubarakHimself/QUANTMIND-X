<!-- @migration-task Error while migrating Svelte code: This type of directive is not valid on components
https://svelte.dev/e/component_invalid_directive -->
<!-- @migration-task Error while migrating Svelte code: This type of directive is not valid on components
https://svelte.dev/e/component_invalid_directive -->
<script lang="ts">
  import { onMount } from 'svelte';
  import {
    Trophy, DollarSign, TrendingUp, TrendingDown, Clock,
    Target, AlertTriangle, CheckCircle, RefreshCw, Wallet,
    BarChart3, Calendar, Activity
  } from 'lucide-svelte';
  import { API_BASE } from '$lib/constants';

  const apiBase = API_BASE || '';

  // Types
  interface Challenge {
    id: string;
    name: string;
    status: 'active' | 'passed' | 'failed';
    account_size: number;
    current_balance: number;
    drawdown_pct: number;
    profit_target_pct: number;
    current_profit_pct: number;
    start_date: string;
    end_date: string;
    days_remaining: number;
  }

  interface FundedAccount {
    id: string;
    prop_firm: string;
    account_id: string;
    balance: number;
    equity: number;
    drawdown_pct: number;
    monthly_target_pct: number;
    monthly_profit_pct: number;
    last_payout: number;
    last_payout_date: string;
    status: 'active' | 'paused' | 'closed';
  }

  interface PropFirmMetrics {
    total_challenges: number;
    active_challenges: number;
    passed_challenges: number;
    failed_challenges: number;
    total_funded_accounts: number;
    active_funded_accounts: number;
    total_payouts: number;
    average_drawdown: number;
    best_performance: number;
    worst_performance: number;
  }

  // State
  let challenges: Challenge[] = [];
  let fundedAccounts: FundedAccount[] = [];
  let metrics: PropFirmMetrics = {
    total_challenges: 0,
    active_challenges: 0,
    passed_challenges: 0,
    failed_challenges: 0,
    total_funded_accounts: 0,
    active_funded_accounts: 0,
    total_payouts: 0,
    average_drawdown: 0,
    best_performance: 0,
    worst_performance: 0
  };
  let loading = true;
  let error = '';

  onMount(async () => {
    await fetchData();
  });

  async function fetchData() {
    loading = true;
    error = '';

    try {
      const baseUrl = apiBase || window.location.origin;

      // Fetch challenges
      const challengesRes = await fetch(`${baseUrl}/api/propfirm/challenges`);
      if (challengesRes.ok) {
        challenges = await challengesRes.json();
      }

      // Fetch funded accounts
      const accountsRes = await fetch(`${baseUrl}/api/propfirm/accounts`);
      if (accountsRes.ok) {
        fundedAccounts = await accountsRes.json();
      }

      // Calculate metrics
      calculateMetrics();
    } catch (err) {
      console.error('Failed to fetch prop firm data:', err);
      error = 'Failed to load data. Using demo data.';
      loadDemoData();
    } finally {
      loading = false;
    }
  }

  function calculateMetrics() {
    const activeChallenges = challenges.filter(c => c.status === 'active');
    const passedChallenges = challenges.filter(c => c.status === 'passed');
    const failedChallenges = challenges.filter(c => c.status === 'failed');
    const activeAccounts = fundedAccounts.filter(a => a.status === 'active');

    const allDrawdowns = [
      ...challenges.map(c => c.drawdown_pct),
      ...fundedAccounts.map(a => a.drawdown_pct)
    ].filter(d => d > 0);

    const allProfits = fundedAccounts.map(a => a.monthly_profit_pct);

    metrics = {
      total_challenges: challenges.length,
      active_challenges: activeChallenges.length,
      passed_challenges: passedChallenges.length,
      failed_challenges: failedChallenges.length,
      total_funded_accounts: fundedAccounts.length,
      active_funded_accounts: activeAccounts.length,
      total_payouts: fundedAccounts.reduce((sum, a) => sum + a.last_payout, 0),
      average_drawdown: allDrawdowns.length > 0
        ? allDrawdowns.reduce((a, b) => a + b, 0) / allDrawdowns.length
        : 0,
      best_performance: allProfits.length > 0 ? Math.max(...allProfits) : 0,
      worst_performance: allProfits.length > 0 ? Math.min(...allProfits) : 0
    };
  }

  function loadDemoData() {
    challenges = [
      {
        id: '1',
        name: 'Topstep - $100K',
        status: 'active',
        account_size: 100000,
        current_balance: 105000,
        drawdown_pct: 2.3,
        profit_target_pct: 10,
        current_profit_pct: 5,
        start_date: '2026-01-15',
        end_date: '2026-04-15',
        days_remaining: 42
      },
      {
        id: '2',
        name: 'FTMO - $50K',
        status: 'active',
        account_size: 50000,
        current_balance: 52000,
        drawdown_pct: 4.1,
        profit_target_pct: 10,
        current_profit_pct: 4,
        start_date: '2026-02-01',
        end_date: '2026-05-01',
        days_remaining: 58
      },
      {
        id: '3',
        name: 'ApexTrader - $25K',
        status: 'passed',
        account_size: 25000,
        current_balance: 28000,
        drawdown_pct: 3.5,
        profit_target_pct: 8,
        current_profit_pct: 12,
        start_date: '2025-11-01',
        end_date: '2026-02-01',
        days_remaining: 0
      }
    ];

    fundedAccounts = [
      {
        id: '1',
        prop_firm: 'FTMO',
        account_id: 'FT-12345',
        balance: 50000,
        equity: 52500,
        drawdown_pct: 2.1,
        monthly_target_pct: 5,
        monthly_profit_pct: 5.2,
        last_payout: 2500,
        last_payout_date: '2026-02-15',
        status: 'active'
      },
      {
        id: '2',
        prop_firm: 'ApexTrader',
        account_id: 'AP-98765',
        balance: 25000,
        equity: 24200,
        drawdown_pct: 4.8,
        monthly_target_pct: 5,
        monthly_profit_pct: -1.2,
        last_payout: 0,
        last_payout_date: '',
        status: 'active'
      }
    ];

    calculateMetrics();
  }

  function getStatusColor(status: string): string {
    const colors: Record<string, string> = {
      active: '#10b981',
      passed: '#3b82f6',
      failed: '#ef4444',
      paused: '#f59e0b',
      closed: '#64748b'
    };
    return colors[status] || '#64748b';
  }

  function getDrawdownColor(pct: number): string {
    if (pct >= 10) return '#ef4444';
    if (pct >= 5) return '#f59e0b';
    return '#10b981';
  }

  function getProfitColor(pct: number): string {
    if (pct < 0) return '#ef4444';
    if (pct === 0) return '#64748b';
    return '#10b981';
  }

  function formatCurrency(amount: number): string {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);
  }

  function formatPercent(pct: number): string {
    return `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`;
  }
</script>

<div class="propfirm-dashboard">
  <!-- Header -->
  <div class="dashboard-header">
    <div class="header-left">
      <Trophy size={24} />
      <h1>PropFirm Dashboard</h1>
    </div>
    <div class="header-right">
      <button class="refresh-btn" on:click={fetchData} disabled={loading}>
        <span class:spinning={loading}><RefreshCw size={16} /></span>
        Refresh
      </button>
    </div>
  </div>

  {#if error}
    <div class="error-banner">
      <AlertTriangle size={16} />
      <span>{error}</span>
    </div>
  {/if}

  <!-- Stats Grid -->
  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-icon challenges">
        <Target size={20} />
      </div>
      <div class="stat-content">
        <span class="stat-value">{metrics.active_challenges}</span>
        <span class="stat-label">Active Challenges</span>
      </div>
    </div>

    <div class="stat-card">
      <div class="stat-icon funded">
        <Wallet size={20} />
      </div>
      <div class="stat-content">
        <span class="stat-value">{metrics.active_funded_accounts}</span>
        <span class="stat-label">Funded Accounts</span>
      </div>
    </div>

    <div class="stat-card">
      <div class="stat-icon payouts">
        <DollarSign size={20} />
      </div>
      <div class="stat-content">
        <span class="stat-value">{formatCurrency(metrics.total_payouts)}</span>
        <span class="stat-label">Total Payouts</span>
      </div>
    </div>

    <div class="stat-card">
      <div class="stat-icon drawdown">
        <TrendingDown size={20} />
      </div>
      <div class="stat-content">
        <span class="stat-value" style="color: {getDrawdownColor(metrics.average_drawdown)}">
          {metrics.average_drawdown.toFixed(1)}%
        </span>
        <span class="stat-label">Avg Drawdown</span>
      </div>
    </div>
  </div>

  <!-- Main Content -->
  <div class="content-grid">
    <!-- Challenges Section -->
    <div class="section-card">
      <div class="section-header">
        <Target size={18} />
        <h3>Active Challenges</h3>
        <span class="badge">{metrics.active_challenges}</span>
      </div>

      <div class="challenges-list">
        {#each challenges.filter(c => c.status === 'active') as challenge (challenge.id)}
          <div class="challenge-item">
            <div class="challenge-header">
              <span class="challenge-name">{challenge.name}</span>
              <span class="status-badge" style="background: {getStatusColor(challenge.status)}20; color: {getStatusColor(challenge.status)}">
                {challenge.status}
              </span>
            </div>

            <div class="challenge-metrics">
              <div class="metric-row">
                <span class="metric-label">Balance</span>
                <span class="metric-value">{formatCurrency(challenge.current_balance)}</span>
              </div>
              <div class="metric-row">
                <span class="metric-label">Drawdown</span>
                <span class="metric-value" style="color: {getDrawdownColor(challenge.drawdown_pct)}">
                  {challenge.drawdown_pct.toFixed(1)}%
                </span>
              </div>
              <div class="metric-row">
                <span class="metric-label">Profit Target</span>
                <span class="metric-value">
                  <span style="color: {getProfitColor(challenge.current_profit_pct)}">
                    {formatPercent(challenge.current_profit_pct)}
                  </span>
                  / {challenge.profit_target_pct}%
                </span>
              </div>
              <div class="metric-row">
                <span class="metric-label">Time Left</span>
                <span class="metric-value">
                  <Calendar size={12} />
                  {challenge.days_remaining} days
                </span>
              </div>
            </div>

            <div class="progress-bar">
              <div
                class="progress-fill"
                style="width: {(challenge.current_profit_pct / challenge.profit_target_pct) * 100}%"
              ></div>
            </div>
          </div>
        {/each}

        {#if challenges.filter(c => c.status === 'active').length === 0}
          <div class="empty-state">
            <Trophy size={32} />
            <p>No active challenges</p>
          </div>
        {/if}
      </div>
    </div>

    <!-- Funded Accounts Section -->
    <div class="section-card">
      <div class="section-header">
        <Wallet size={18} />
        <h3>Funded Accounts</h3>
        <span class="badge">{metrics.active_funded_accounts}</span>
      </div>

      <div class="accounts-list">
        {#each fundedAccounts as account (account.id)}
          <div class="account-item">
            <div class="account-header">
              <div class="firm-info">
                <span class="firm-name">{account.prop_firm}</span>
                <span class="account-id">{account.account_id}</span>
              </div>
              <span class="status-badge" style="background: {getStatusColor(account.status)}20; color: {getStatusColor(account.status)}">
                {account.status}
              </span>
            </div>

            <div class="account-metrics">
              <div class="metric-group">
                <div class="metric">
                  <span class="metric-label">Equity</span>
                  <span class="metric-value">{formatCurrency(account.equity)}</span>
                </div>
                <div class="metric">
                  <span class="metric-label">Drawdown</span>
                  <span class="metric-value" style="color: {getDrawdownColor(account.drawdown_pct)}">
                    {account.drawdown_pct.toFixed(1)}%
                  </span>
                </div>
              </div>
              <div class="metric-group">
                <div class="metric">
                  <span class="metric-label">Monthly Target</span>
                  <span class="metric-value">{account.monthly_target_pct}%</span>
                </div>
                <div class="metric">
                  <span class="metric-label">This Month</span>
                  <span class="metric-value" style="color: {getProfitColor(account.monthly_profit_pct)}">
                    {formatPercent(account.monthly_profit_pct)}
                  </span>
                </div>
              </div>
            </div>

            {#if account.last_payout > 0}
              <div class="payout-info">
                <DollarSign size={14} />
                <span>Last payout: {formatCurrency(account.last_payout)} on {account.last_payout_date}</span>
              </div>
            {/if}
          </div>
        {/each}

        {#if fundedAccounts.length === 0}
          <div class="empty-state">
            <Wallet size={32} />
            <p>No funded accounts</p>
          </div>
        {/if}
      </div>
    </div>

    <!-- Performance Summary -->
    <div class="section-card wide">
      <div class="section-header">
        <BarChart3 size={18} />
        <h3>Performance Summary</h3>
      </div>

      <div class="summary-grid">
        <div class="summary-item">
          <div class="summary-icon">
            <CheckCircle size={20} />
          </div>
          <div class="summary-content">
            <span class="summary-value">{metrics.passed_challenges}</span>
            <span class="summary-label">Passed Challenges</span>
          </div>
        </div>

        <div class="summary-item">
          <div class="summary-icon failed">
            <AlertTriangle size={20} />
          </div>
          <div class="summary-content">
            <span class="summary-value">{metrics.failed_challenges}</span>
            <span class="summary-label">Failed Challenges</span>
          </div>
        </div>

        <div class="summary-item">
          <div class="summary-icon best">
            <TrendingUp size={20} />
          </div>
          <div class="summary-content">
            <span class="summary-value" style="color: #10b981">
              {formatPercent(metrics.best_performance)}
            </span>
            <span class="summary-label">Best Month</span>
          </div>
        </div>

        <div class="summary-item">
          <div class="summary-icon worst">
            <TrendingDown size={20} />
          </div>
          <div class="summary-content">
            <span class="summary-value" style="color: {getProfitColor(metrics.worst_performance)}">
              {formatPercent(metrics.worst_performance)}
            </span>
            <span class="summary-label">Worst Month</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Historical Challenges -->
    <div class="section-card wide">
      <div class="section-header">
        <Activity size={18} />
        <h3>Challenge History</h3>
      </div>

      <div class="history-table">
        <div class="table-header">
          <span>Name</span>
          <span>Account Size</span>
          <span>Final Balance</span>
          <span>Profit</span>
          <span>Status</span>
        </div>
        {#each challenges as challenge (challenge.id)}
          <div class="table-row">
            <span class="name">{challenge.name}</span>
            <span>{formatCurrency(challenge.account_size)}</span>
            <span>{formatCurrency(challenge.current_balance)}</span>
            <span style="color: {getProfitColor(challenge.current_profit_pct)}">
              {formatPercent(challenge.current_profit_pct)}
            </span>
            <span class="status-badge small" style="background: {getStatusColor(challenge.status)}20; color: {getStatusColor(challenge.status)}">
              {challenge.status}
            </span>
          </div>
        {/each}
      </div>
    </div>
  </div>
</div>

<style>
  .propfirm-dashboard {
    padding: 24px;
    max-width: 1400px;
    margin: 0 auto;
  }

  .dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
    color: #e2e8f0;
  }

  .header-left h1 {
    margin: 0;
    font-size: 20px;
    font-weight: 600;
  }

  .refresh-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    color: #e2e8f0;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .refresh-btn:hover:not(:disabled) {
    background: #334155;
  }

  .refresh-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 6px;
    color: #fca5a5;
    margin-bottom: 16px;
    font-size: 13px;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
  }

  .stat-card {
    background: #1e293b;
    border-radius: 8px;
    padding: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .stat-icon {
    width: 40px;
    height: 40px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .stat-icon.challenges {
    background: rgba(59, 130, 246, 0.1);
    color: #3b82f6;
  }

  .stat-icon.funded {
    background: rgba(16, 185, 129, 0.1);
    color: #10b981;
  }

  .stat-icon.payouts {
    background: rgba(234, 179, 8, 0.1);
    color: #eab308;
  }

  .stat-icon.drawdown {
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
  }

  .stat-content {
    display: flex;
    flex-direction: column;
  }

  .stat-value {
    font-size: 18px;
    font-weight: 600;
    color: #e2e8f0;
  }

  .stat-label {
    font-size: 11px;
    color: #64748b;
  }

  .content-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }

  .section-card {
    background: #1e293b;
    border-radius: 8px;
    padding: 16px;
  }

  .section-card.wide {
    grid-column: span 2;
  }

  .section-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
    color: #e2e8f0;
  }

  .section-header h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    flex: 1;
  }

  .badge {
    padding: 2px 8px;
    background: #334155;
    border-radius: 10px;
    font-size: 11px;
    color: #94a3b8;
  }

  .challenges-list, .accounts-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .challenge-item, .account-item {
    background: #0f172a;
    border-radius: 6px;
    padding: 12px;
  }

  .challenge-header, .account-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .challenge-name, .firm-name {
    font-size: 13px;
    font-weight: 600;
    color: #e2e8f0;
  }

  .account-id {
    font-size: 11px;
    color: #64748b;
    margin-left: 8px;
  }

  .firm-info {
    display: flex;
    align-items: center;
  }

  .status-badge {
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 500;
    text-transform: capitalize;
  }

  .status-badge.small {
    padding: 2px 6px;
    font-size: 9px;
  }

  .challenge-metrics, .account-metrics {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 12px;
  }

  .metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
  }

  .metric-label {
    color: #64748b;
  }

  .metric-value {
    color: #e2e8f0;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .metric-group {
    display: flex;
    justify-content: space-between;
  }

  .metric-group .metric {
    flex: 1;
  }

  .progress-bar {
    height: 4px;
    background: #1e293b;
    border-radius: 2px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: #3b82f6;
    transition: width 0.3s ease;
  }

  .payout-info {
    display: flex;
    align-items: center;
    gap: 6px;
    padding-top: 8px;
    border-top: 1px solid #1e293b;
    font-size: 11px;
    color: #10b981;
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 32px;
    color: #64748b;
    gap: 8px;
  }

  .empty-state p {
    margin: 0;
    font-size: 13px;
  }

  .summary-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
  }

  .summary-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    background: #0f172a;
    border-radius: 6px;
  }

  .summary-icon {
    width: 36px;
    height: 36px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(16, 185, 129, 0.1);
    color: #10b981;
  }

  .summary-icon.failed {
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
  }

  .summary-icon.best {
    background: rgba(16, 185, 129, 0.1);
    color: #10b981;
  }

  .summary-icon.worst {
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
  }

  .summary-content {
    display: flex;
    flex-direction: column;
  }

  .summary-value {
    font-size: 18px;
    font-weight: 600;
    color: #e2e8f0;
  }

  .summary-label {
    font-size: 11px;
    color: #64748b;
  }

  .history-table {
    display: flex;
    flex-direction: column;
  }

  .table-header {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr 1fr 1fr;
    gap: 16px;
    padding: 8px 12px;
    font-size: 11px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    border-bottom: 1px solid #334155;
  }

  .table-row {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr 1fr 1fr;
    gap: 16px;
    padding: 10px 12px;
    font-size: 12px;
    color: #e2e8f0;
    border-bottom: 1px solid #0f172a;
  }

  .table-row:last-child {
    border-bottom: none;
  }

  .table-row .name {
    font-weight: 500;
  }

  @media (max-width: 1024px) {
    .stats-grid {
      grid-template-columns: repeat(2, 1fr);
    }

    .content-grid {
      grid-template-columns: 1fr;
    }

    .section-card.wide {
      grid-column: span 1;
    }

    .summary-grid {
      grid-template-columns: repeat(2, 1fr);
    }

    .table-header, .table-row {
      grid-template-columns: 1.5fr 1fr 1fr 1fr 0.8fr;
      font-size: 10px;
    }
  }

  @media (max-width: 640px) {
    .stats-grid {
      grid-template-columns: 1fr;
    }

    .summary-grid {
      grid-template-columns: 1fr;
    }

    .table-header, .table-row {
      grid-template-columns: 1fr 1fr 1fr;
    }

    .table-header span:nth-child(2),
    .table-row span:nth-child(2) {
      display: none;
    }
  }
</style>

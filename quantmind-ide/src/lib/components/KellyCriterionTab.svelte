<script lang="ts">
  import { Calculator } from 'lucide-svelte';

  export let kellyData: Record<string, {
    kellyFraction: number;
    halfKelly: number;
    winRate: number;
    avgWin: number;
    avgLoss: number;
    expectedValue: number;
    suggestedFraction: number;
  }>;

  export let bots: Array<{
    id: string;
    name: string;
  }>;

  export let kellyRankings: Array<{
    botId: string;
    name: string;
    kellyFraction: number;
    halfKelly: number;
    winRate: number;
    expectedValue: number;
    suggestedFraction: number;
    kellyScore: string;
  }>;

  export let kellyHistory: Array<{
    date: string;
    botId: string;
    fraction: number;
    result: number;
  }>;

  function formatCurrency(value: number) {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value);
  }

  $: avgKelly = Object.values(kellyData).reduce((a, b) => a + b.kellyFraction, 0) / Object.values(kellyData).length * 100;
  $: avgHalfKelly = Object.values(kellyData).reduce((a, b) => a + b.halfKelly, 0) / Object.values(kellyData).length * 100;
  $: maxKelly = Math.max(...Object.values(kellyData).map(k => k.kellyFraction * 100));
</script>

<div class="kelly-section">
  <div class="kelly-header">
    <div class="kelly-info">
      <Calculator size={20} />
      <div>
        <h3>Kelly Criterion Analysis</h3>
        <p>Optimal position sizing based on win rate and risk/reward</p>
      </div>
    </div>
    <div class="kelly-summary">
      <div class="summary-item">
        <span class="label">Avg Kelly</span>
        <span class="value">{avgKelly.toFixed(1)}%</span>
      </div>
      <div class="summary-item">
        <span class="label">Avg Half-Kelly</span>
        <span class="value">{avgHalfKelly.toFixed(1)}%</span>
      </div>
      <div class="summary-item">
        <span class="label">Best Kelly</span>
        <span class="value success">{maxKelly.toFixed(1)}%</span>
      </div>
    </div>
  </div>

  <!-- Kelly Rankings -->
  <div class="kelly-rankings">
    <h4>Bot Rankings by Kelly Score</h4>
    <div class="kelly-table">
      <div class="table-header kelly-header-row">
        <span>Rank</span>
        <span>Bot</span>
        <span>Full Kelly</span>
        <span>Half Kelly</span>
        <span>Win Rate</span>
        <span>EV/Trade</span>
        <span>Kelly Score</span>
      </div>

      {#each kellyRankings as ranking, index}
        <div class="table-row kelly-row">
          <span class="rank kelly-rank">#{index + 1}</span>
          <span class="name">{ranking.name}</span>
          <span class="kelly-value full-kelly" title="Full Kelly - aggressive">
            <div class="kelly-bar" style="width: {ranking.kellyFraction * 100 * 4}%"></div>
            {(ranking.kellyFraction * 100).toFixed(1)}%
          </span>
          <span class="kelly-value half-kelly" title="Half Kelly - conservative">
            <div class="kelly-bar half" style="width: {ranking.halfKelly * 100 * 4}%"></div>
            {(ranking.halfKelly * 100).toFixed(1)}%
          </span>
          <span class="winrate">{ranking.winRate * 100}%</span>
          <span class="expected-value" class:positive={ranking.expectedValue > 0}>
            ${ranking.expectedValue.toFixed(2)}
          </span>
          <span class="kelly-score" class:top={index === 0}>
            {ranking.kellyScore}
          </span>
        </div>
      {/each}
    </div>
  </div>

  <!-- Kelly Details Grid -->
  <div class="kelly-details-grid">
    {#each Object.entries(kellyData) as [botId, data]}
      <div class="kelly-card">
        <div class="kelly-card-header">
          <span class="bot-name">{bots.find(b => b.id === botId)?.name || botId}</span>
          <span
            class="status-badge"
            class:optimal={data.kellyFraction < 0.15}
            class:caution={data.kellyFraction >= 0.15 && data.kellyFraction < 0.25}
            class:warning={data.kellyFraction >= 0.25}
          >
            {data.kellyFraction < 0.15 ? 'Optimal' : data.kellyFraction < 0.25 ? 'Moderate' : 'High Risk'}
          </span>
        </div>
        <div class="kelly-card-body">
          <div class="metric-row">
            <span class="metric-label">Win Rate</span>
            <span class="metric-value">{(data.winRate * 100).toFixed(0)}%</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Avg Win</span>
            <span class="metric-value success">${data.avgWin.toFixed(2)}</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Avg Loss</span>
            <span class="metric-value danger">${data.avgLoss.toFixed(2)}</span>
          </div>
          <div class="kelly-visual">
            <div class="kelly-gauge">
              <div class="gauge-track">
                <div class="gauge-fill" style="width: {data.kellyFraction * 100}%"></div>
                <div class="gauge-half" style="left: {data.halfKelly * 100}%"></div>
              </div>
              <div class="gauge-labels">
                <span>0%</span>
                <span class="half-mark">Half: {(data.halfKelly * 100).toFixed(1)}%</span>
                <span>{(data.kellyFraction * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
          <div class="suggested-fraction">
            <span class="label">Suggested Fraction:</span>
            <span class="value">{(data.suggestedFraction * 100).toFixed(0)}% of Kelly</span>
          </div>
        </div>
      </div>
    {/each}
  </div>

  <!-- Kelly History -->
  <div class="kelly-history">
    <h4>Recent Kelly Adjustments</h4>
    <div class="history-list">
      {#each kellyHistory.slice(0, 5) as entry}
        <div class="history-item">
          <span class="history-date">{entry.date}</span>
          <span class="history-bot">{bots.find(b => b.id === entry.botId)?.name || entry.botId}</span>
          <span class="history-fraction">Kelly: {(entry.fraction * 100).toFixed(1)}%</span>
          <span
            class="history-result"
            class:positive={entry.result > 0}
            class:negative={entry.result < 0}
          >
            {entry.result > 0 ? '+' : ''}{formatCurrency(entry.result)}
          </span>
        </div>
      {/each}
    </div>
  </div>
</div>

<style>
  .kelly-section {
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  .kelly-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 16px;
  }

  .kelly-info {
    display: flex;
    gap: 12px;
    align-items: flex-start;
  }

  .kelly-info h3 {
    margin: 0;
    font-size: 14px;
    color: var(--text-primary);
  }

  .kelly-info p {
    margin: 4px 0 0;
    font-size: 12px;
    color: var(--text-muted);
  }

  .kelly-summary {
    display: flex;
    gap: 24px;
  }

  .summary-item {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 2px;
  }

  .summary-item .label {
    font-size: 10px;
    color: var(--text-muted);
  }

  .summary-item .value {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .summary-item .value.success {
    color: #10b981;
  }

  .kelly-rankings {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 16px;
  }

  .kelly-rankings h4 {
    margin: 0 0 12px;
    font-size: 13px;
    color: var(--text-primary);
  }

  .kelly-table {
    overflow-x: auto;
  }

  .table-header {
    display: grid;
    grid-template-columns: 50px 1fr 100px 100px 70px 80px 80px;
    gap: 8px;
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    font-size: 10px;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
  }

  .table-row {
    display: grid;
    grid-template-columns: 50px 1fr 100px 100px 70px 80px 80px;
    gap: 8px;
    padding: 10px 12px;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 12px;
    color: var(--text-primary);
    align-items: center;
  }

  .table-row:last-child {
    border-bottom: none;
  }

  .kelly-rank {
    font-weight: 600;
    color: var(--accent-primary);
  }

  .kelly-value {
    position: relative;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
  }

  .kelly-bar {
    position: absolute;
    top: 50%;
    left: 0;
    height: 4px;
    background: var(--accent-primary);
    transform: translateY(-50%);
    border-radius: 2px;
  }

  .kelly-bar.half {
    background: #10b981;
  }

  .expected-value {
    font-family: 'JetBrains Mono', monospace;
  }

  .expected-value.positive {
    color: #10b981;
  }

  .kelly-score {
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
  }

  .kelly-score.top {
    color: #10b981;
  }

  .kelly-details-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
  }

  .kelly-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 16px;
  }

  .kelly-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .kelly-card-header .bot-name {
    font-weight: 500;
    color: var(--text-primary);
  }

  .status-badge {
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 10px;
    text-transform: uppercase;
  }

  .status-badge.optimal {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .status-badge.caution {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
  }

  .status-badge.warning {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .kelly-card-body {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .metric-row {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
  }

  .metric-label {
    color: var(--text-muted);
  }

  .metric-value {
    font-weight: 500;
    color: var(--text-primary);
  }

  .metric-value.success {
    color: #10b981;
  }

  .metric-value.danger {
    color: #ef4444;
  }

  .kelly-visual {
    margin: 8px 0;
  }

  .kelly-gauge {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .gauge-track {
    position: relative;
    height: 8px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    overflow: visible;
  }

  .gauge-fill {
    position: absolute;
    left: 0;
    top: 0;
    height: 100%;
    background: linear-gradient(90deg, #10b981, #f59e0b, #ef4444);
    border-radius: 4px;
    transition: width 0.3s;
  }

  .gauge-half {
    position: absolute;
    top: -2px;
    width: 2px;
    height: 12px;
    background: #fff;
    border-radius: 1px;
  }

  .gauge-labels {
    display: flex;
    justify-content: space-between;
    font-size: 9px;
    color: var(--text-muted);
  }

  .half-mark {
    color: #10b981;
  }

  .suggested-fraction {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    padding-top: 8px;
    border-top: 1px solid var(--border-subtle);
  }

  .suggested-fraction .label {
    color: var(--text-muted);
  }

  .suggested-fraction .value {
    font-weight: 500;
    color: var(--accent-primary);
  }

  .kelly-history {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 16px;
  }

  .kelly-history h4 {
    margin: 0 0 12px;
    font-size: 13px;
    color: var(--text-primary);
  }

  .history-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .history-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    font-size: 11px;
  }

  .history-date {
    color: var(--text-muted);
  }

  .history-bot {
    flex: 1;
    color: var(--text-primary);
    margin-left: 8px;
  }

  .history-fraction {
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', monospace;
  }

  .history-result {
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
  }

  .history-result.positive {
    color: #10b981;
  }

  .history-result.negative {
    color: #ef4444;
  }
</style>

<script lang="ts">
  import { Trophy, TrendingUp, Award, AlertTriangle, ArrowUpDown, ArrowUp, ArrowDown } from 'lucide-svelte';
  import { API_BASE } from '$lib/constants';

  const apiBase = API_BASE || '';

  interface LeaderboardEntry {
    bot_id: string;
    bot_name: string;
    composite_score: number;
    tier: string;
    session_specialist: boolean;
    session_win_rate: number;
    net_pnl: number;
    consistency: number;
    ev_per_trade: number;
    queue_position: number;
    rank: number;
  }

  interface LeaderboardResponse {
    session_id: string;
    entries: LeaderboardEntry[];
    count: number;
    timestamp_utc: string;
  }

  type SortKey = 'rank' | 'composite_score' | 'tier' | 'session_win_rate' | 'net_pnl' | 'consistency' | 'ev_per_trade';
  type SortDir = 'asc' | 'desc';

  let entries = $state<LeaderboardEntry[]>([]);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let sessionId = $state('LONDON');

  let sortKey = $state<SortKey>('rank');
  let sortDir = $state<SortDir>('asc');

  const sessions = ['LONDON', 'NY', 'ASIA', 'SYDNEY'];

  async function fetchLeaderboard() {
    try {
      loading = true;
      const baseUrl = apiBase || window.location.origin;
      const url = `${baseUrl}/api/dpr/leaderboard?session_id=${sessionId}`;

      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`Failed to fetch leaderboard: ${response.status}`);
      }

      const data: LeaderboardResponse = await response.json();
      entries = data.entries || [];
      error = null;
    } catch (e) {
      console.error('Failed to fetch DPR leaderboard:', e);
      error = e instanceof Error ? e.message : 'Unknown error';
    } finally {
      loading = false;
    }
  }

  function getTierColor(tier: string): string {
    switch (tier) {
      case 'TIER_1': return '#10b981';
      case 'TIER_2': return '#f59e0b';
      case 'TIER_3': return '#6b7280';
      default: return '#6b7280';
    }
  }

  function getTierBgColor(tier: string): string {
    switch (tier) {
      case 'TIER_1': return 'rgba(16, 185, 129, 0.15)';
      case 'TIER_2': return 'rgba(245, 158, 11, 0.15)';
      case 'TIER_3': return 'rgba(107, 114, 128, 0.15)';
      default: return 'rgba(107, 114, 128, 0.15)';
    }
  }

  function getTierBadge(tier: string): string {
    switch (tier) {
      case 'TIER_1': return 'T1';
      case 'TIER_2': return 'T2';
      case 'TIER_3': return 'T3';
      default: return tier;
    }
  }

  function getScoreColor(score: number): string {
    if (score >= 80) return '#10b981';
    if (score >= 50) return '#f59e0b';
    return '#ef4444';
  }

  function getPnLColor(pnl: number): string {
    if (pnl > 50) return '#10b981';
    if (pnl < 30) return '#ef4444';
    return '#9ca3af';
  }

  function getSortIcon(key: SortKey) {
    if (sortKey !== key) return ArrowUpDown;
    return sortDir === 'asc' ? ArrowUp : ArrowDown;
  }

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      sortDir = sortDir === 'asc' ? 'desc' : 'asc';
    } else {
      sortKey = key;
      sortDir = 'desc';
    }
  }

  function sortedEntries(): LeaderboardEntry[] {
    return [...entries].sort((a, b) => {
      let aVal: number | string;
      let bVal: number | string;

      switch (sortKey) {
        case 'rank':
          aVal = a.rank;
          bVal = b.rank;
          break;
        case 'composite_score':
          aVal = a.composite_score;
          bVal = b.composite_score;
          break;
        case 'tier':
          const tierOrder = { 'TIER_1': 1, 'TIER_2': 2, 'TIER_3': 3 };
          aVal = tierOrder[a.tier as keyof typeof tierOrder] || 4;
          bVal = tierOrder[b.tier as keyof typeof tierOrder] || 4;
          break;
        case 'session_win_rate':
          aVal = a.session_win_rate;
          bVal = b.session_win_rate;
          break;
        case 'net_pnl':
          aVal = a.net_pnl;
          bVal = b.net_pnl;
          break;
        case 'consistency':
          aVal = a.consistency;
          bVal = b.consistency;
          break;
        case 'ev_per_trade':
          aVal = a.ev_per_trade;
          bVal = b.ev_per_trade;
          break;
        default:
          return 0;
      }

      if (sortDir === 'asc') {
        return aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
      } else {
        return aVal > bVal ? -1 : aVal < bVal ? 1 : 0;
      }
    });
  }

  function formatPercent(value: number): string {
    return (value * 100).toFixed(1) + '%';
  }

  function formatScore(score: number): string {
    return score.toFixed(1);
  }

  $effect(() => {
    fetchLeaderboard();
  });
</script>

<div class="dpr-leaderboard">
  <div class="leaderboard-header">
    <div class="header-left">
      <Trophy size={16} />
      <h3>DPR Leaderboard</h3>
    </div>
    <div class="header-right">
      <select
        class="session-select"
        bind:value={sessionId}
        onchange={() => fetchLeaderboard()}
      >
        {#each sessions as session}
          <option value={session}>{session}</option>
        {/each}
      </select>
    </div>
  </div>

  {#if error}
    <div class="error-message">{error}</div>
  {/if}

  {#if loading}
    <div class="loading-placeholder">
      {#each Array(5) as _, i}
        <div class="skeleton-row" style="animation-delay: {i * 100}ms"></div>
      {/each}
    </div>
  {:else if entries.length === 0}
    <div class="empty-state">No bots in leaderboard</div>
  {:else}
    <div class="table-wrapper">
      <table class="leaderboard-table">
        <thead>
          <tr>
            <th class="sortable" onclick={() => toggleSort('rank')}>
              <span class="th-content">
                <span class="sort-icon"><ArrowUpDown size={12} /></span>
                Rank
              </span>
            </th>
            <th>Bot</th>
            <th class="sortable" onclick={() => toggleSort('composite_score')}>
              <span class="th-content">
                <span class="sort-icon"><ArrowUpDown size={12} /></span>
                Score
              </span>
            </th>
            <th class="sortable" onclick={() => toggleSort('tier')}>
              <span class="th-content">
                <span class="sort-icon"><ArrowUpDown size={12} /></span>
                Tier
              </span>
            </th>
            <th class="sortable" onclick={() => toggleSort('session_win_rate')}>
              <span class="th-content">
                <span class="sort-icon"><ArrowUpDown size={12} /></span>
                Win Rate
              </span>
            </th>
            <th class="sortable" onclick={() => toggleSort('net_pnl')}>
              <span class="th-content">
                <span class="sort-icon"><ArrowUpDown size={12} /></span>
                Net PnL
              </span>
            </th>
            <th class="sortable" onclick={() => toggleSort('consistency')}>
              <span class="th-content">
                <span class="sort-icon"><ArrowUpDown size={12} /></span>
                Consist.
              </span>
            </th>
            <th class="sortable" onclick={() => toggleSort('ev_per_trade')}>
              <span class="th-content">
                <span class="sort-icon"><ArrowUpDown size={12} /></span>
                EV/Trade
              </span>
            </th>
            <th>Badges</th>
          </tr>
        </thead>
        <tbody>
          {#each sortedEntries() as entry (entry.bot_id)}
            <tr class="leaderboard-row" class:top-three={entry.rank <= 3}>
              <td class="rank-cell">
                <span class="rank-number" class:gold={entry.rank === 1} class:silver={entry.rank === 2} class:bronze={entry.rank === 3}>
                  {entry.rank}
                </span>
              </td>
              <td class="bot-cell">
                <span class="bot-name">{entry.bot_name}</span>
                <span class="bot-id">{entry.bot_id}</span>
              </td>
              <td class="score-cell">
                <span class="composite-score" style="color: {getScoreColor(entry.composite_score)}">
                  {formatScore(entry.composite_score)}
                </span>
              </td>
              <td class="tier-cell">
                <span
                  class="tier-badge"
                  style="background: {getTierBgColor(entry.tier)}; color: {getTierColor(entry.tier)}"
                >
                  {getTierBadge(entry.tier)}
                </span>
              </td>
              <td class="winrate-cell">
                <span class="winrate-value">{formatPercent(entry.session_win_rate)}</span>
              </td>
              <td class="pnl-cell">
                <span class="pnl-value" style="color: {getPnLColor(entry.net_pnl)}">
                  {entry.net_pnl.toFixed(0)}
                </span>
              </td>
              <td class="consistency-cell">
                <span class="consistency-value">{formatPercent(entry.consistency)}</span>
              </td>
              <td class="ev-cell">
                <span class="ev-value">{formatScore(entry.ev_per_trade * 100)}</span>
              </td>
              <td class="badges-cell">
                <div class="badges">
                  {#if entry.session_specialist}
                    <span class="badge specialist" title="Session Specialist">
                      <Award size={12} />
                    </span>
                  {/if}
                </div>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</div>

<style>
  .dpr-leaderboard {
    background: rgba(8, 8, 12, 0.75);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
    color: #e4e4e7;
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  .leaderboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #f59e0b;
  }

  .header-left h3 {
    font-size: 14px;
    font-weight: 600;
    margin: 0;
    color: #e4e4e7;
  }

  .session-select {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    color: #e4e4e7;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 11px;
    padding: 4px 8px;
    cursor: pointer;
  }

  .session-select:hover {
    border-color: rgba(255, 255, 255, 0.2);
  }

  .error-message {
    color: #ef4444;
    font-size: 13px;
    padding: 12px;
    background: rgba(239, 68, 68, 0.1);
    border-radius: 6px;
    margin-bottom: 12px;
  }

  .loading-placeholder {
    display: flex;
    flex-direction: column;
    gap: 8px;
    flex: 1;
  }

  .skeleton-row {
    height: 40px;
    background: linear-gradient(90deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0.05) 100%);
    border-radius: 6px;
    animation: shimmer 1.5s infinite;
  }

  @keyframes shimmer {
    0% { opacity: 0.5; }
    50% { opacity: 1; }
    100% { opacity: 0.5; }
  }

  .empty-state {
    text-align: center;
    padding: 48px 24px;
    color: #6b7280;
    font-size: 13px;
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .table-wrapper {
    overflow-x: auto;
    flex: 1;
  }

  .leaderboard-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }

  .leaderboard-table th {
    text-align: left;
    padding: 8px 10px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #6b7280;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    white-space: nowrap;
  }

  .leaderboard-table th.sortable {
    cursor: pointer;
    user-select: none;
  }

  .leaderboard-table th.sortable:hover {
    color: #e4e4e7;
  }

  .th-content {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .sort-icon {
    opacity: 0.5;
  }

  .leaderboard-table td {
    padding: 10px 10px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    vertical-align: middle;
  }

  .leaderboard-row {
    transition: background 0.15s ease;
  }

  .leaderboard-row:hover {
    background: rgba(255, 255, 255, 0.03);
  }

  .leaderboard-row.top-three {
    background: rgba(245, 158, 11, 0.05);
  }

  .leaderboard-row.top-three:hover {
    background: rgba(245, 158, 11, 0.08);
  }

  .rank-cell {
    width: 50px;
    text-align: center;
  }

  .rank-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 12px;
    background: rgba(255, 255, 255, 0.05);
    color: #9ca3af;
  }

  .rank-number.gold {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
  }

  .rank-number.silver {
    background: rgba(156, 163, 175, 0.2);
    color: #9ca3af;
  }

  .rank-number.bronze {
    background: rgba(180, 83, 9, 0.2);
    color: #b45309;
  }

  .bot-cell {
    min-width: 140px;
  }

  .bot-name {
    display: block;
    font-weight: 500;
    color: #e4e4e7;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .bot-id {
    display: block;
    font-size: 10px;
    color: #6b7280;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .score-cell {
    width: 70px;
  }

  .composite-score {
    font-size: 16px;
    font-weight: 700;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .tier-cell {
    width: 60px;
  }

  .tier-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 4px;
    text-transform: uppercase;
  }

  .winrate-cell,
  .pnl-cell,
  .consistency-cell,
  .ev-cell {
    width: 80px;
    text-align: right;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
  }

  .pnl-value {
    font-weight: 600;
  }

  .badges-cell {
    width: 50px;
  }

  .badges {
    display: flex;
    gap: 4px;
  }

  .badge {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 4px;
  }

  .badge.specialist {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }
</style>

<script lang="ts">
  import { TrendingUp, TrendingDown, Minus, Award, AlertTriangle } from 'lucide-svelte';
  import { API_BASE } from '$lib/constants';

  const apiBase = API_BASE || '';

  interface DprScore {
    bot_id: string;
    composite_score: number;
    components: {
      session_win_rate: number;
      net_pnl: number;
      consistency: number;
      ev_per_trade: number;
    };
    rank: number;
    tier: string;
    session_specialist: boolean;
    session_concern: boolean;
    consecutive_negative_ev: number;
  }

  interface Props {
    botId?: string;
    scores?: DprScore[];
    compact?: boolean;
  }

  let { botId = '', scores = [], compact = false }: Props = $props();

  let dprScores = $state<DprScore[]>(scores);
  let loading = $state(true);
  let error = $state<string | null>(null);

  async function fetchScores() {
    try {
      const baseUrl = apiBase || window.location.origin;
      const url = botId
        ? `${baseUrl}/api/dead-zone/dpr/scores/${botId}`
        : `${baseUrl}/api/dead-zone/dpr/scores`;

      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`Failed to fetch: ${response.status}`);
      }

      const data = await response.json();

      if (botId) {
        dprScores = [data];
      } else {
        dprScores = data.scores || [];
      }

      error = null;
    } catch (e) {
      console.error('Failed to fetch DPR scores:', e);
      error = e instanceof Error ? e.message : 'Unknown error';
    } finally {
      loading = false;
    }
  }

  function getTierColor(tier: string): string {
    switch (tier) {
      case 'T1': return '#10b981';
      case 'T2': return '#f59e0b';
      case 'T3': return '#6b7280';
      default: return '#6b7280';
    }
  }

  function getTierBgColor(tier: string): string {
    switch (tier) {
      case 'T1': return 'rgba(16, 185, 129, 0.15)';
      case 'T2': return 'rgba(245, 158, 11, 0.15)';
      case 'T3': return 'rgba(107, 114, 128, 0.15)';
      default: return 'rgba(107, 114, 128, 0.15)';
    }
  }

  function getScoreColor(score: number): string {
    if (score >= 80) return '#10b981';
    if (score >= 50) return '#f59e0b';
    return '#ef4444';
  }

  function getPnLColor(pnl: number): string {
    if (pnl > 0) return '#10b981';
    if (pnl < 0) return '#ef4444';
    return '#9ca3af';
  }

  function formatScore(score: number): string {
    return score.toFixed(1);
  }

  function formatPercent(value: number): string {
    return (value * 100).toFixed(1) + '%';
  }

  function formatPnL(pnl: number): string {
    const sign = pnl >= 0 ? '+' : '';
    return sign + pnl.toFixed(2);
  }

  $effect(() => {
    if (scores.length > 0) {
      dprScores = scores;
      loading = false;
    }
  });

  // Auto-fetch when component mounts if no scores provided
  $effect(() => {
    if (scores.length === 0 && !botId) {
      fetchScores();
    }
  });
</script>

<div class="dpr-score-card" class:compact>
  <div class="card-header">
    <TrendingUp size={16} />
    <h4>DPR Scores</h4>
  </div>

  {#if error}
    <div class="error-message">{error}</div>
  {/if}

  {#if loading}
    <div class="loading-placeholder">
      <div class="skeleton-row"></div>
      <div class="skeleton-row"></div>
      <div class="skeleton-row"></div>
    </div>
  {:else if dprScores.length === 0}
    <div class="empty-state">No DPR scores available</div>
  {:else}
    <div class="scores-list">
      {#each dprScores as score (score.bot_id)}
        <div class="score-item">
          <div class="score-header">
            <div class="bot-info">
              <span class="rank">#{score.rank}</span>
              <span class="bot-id">{score.bot_id}</span>
            </div>
            <div class="badges">
              {#if score.session_specialist}
                <span class="badge specialist" title="Session Specialist - outperformed regime expectations">
                  <Award size={12} />
                </span>
              {/if}
              {#if score.session_concern}
                <span class="badge concern" title="Session Concern - 3+ consecutive negative EV sessions">
                  <AlertTriangle size={12} />
                </span>
              {/if}
              <span
                class="tier-badge"
                style="background: {getTierBgColor(score.tier)}; color: {getTierColor(score.tier)}"
              >
                {score.tier}
              </span>
            </div>
          </div>

          <div class="composite-score" style="color: {getScoreColor(score.composite_score)}">
            <span class="score-value">{formatScore(score.composite_score)}</span>
            <span class="score-label">DPR</span>
          </div>

          {#if !compact}
            <div class="components-grid">
              <div class="component">
                <span class="component-label">WR</span>
                <span class="component-value">{formatPercent(score.components.session_win_rate)}</span>
              </div>
              <div class="component">
                <span class="component-label">PnL</span>
                <span class="component-value" style="color: {getPnLColor(score.components.net_pnl)}">
                  {formatPnL(score.components.net_pnl)}
                </span>
              </div>
              <div class="component">
                <span class="component-label">Consist.</span>
                <span class="component-value">{formatPercent(score.components.consistency)}</span>
              </div>
              <div class="component">
                <span class="component-label">EV/Trade</span>
                <span class="component-value">{formatScore(score.components.ev_per_trade)}</span>
              </div>
            </div>

            {#if score.consecutive_negative_ev > 0}
              <div class="concern-info">
                <AlertTriangle size={12} />
                <span>{score.consecutive_negative_ev} consecutive negative EV sessions</span>
              </div>
            {/if}
          {/if}
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .dpr-score-card {
    background: rgba(8, 8, 12, 0.75);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
    color: #e4e4e7;
  }

  .card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
  }

  .card-header h4 {
    font-size: 14px;
    font-weight: 600;
    margin: 0;
  }

  .error-message {
    color: #ef4444;
    font-size: 13px;
    padding: 8px;
    background: rgba(239, 68, 68, 0.1);
    border-radius: 6px;
  }

  .loading-placeholder {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .skeleton-row {
    height: 48px;
    background: linear-gradient(90deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0.05) 100%);
    border-radius: 8px;
    animation: shimmer 1.5s infinite;
  }

  @keyframes shimmer {
    0% { opacity: 0.5; }
    50% { opacity: 1; }
    100% { opacity: 0.5; }
  }

  .empty-state {
    text-align: center;
    padding: 24px;
    color: #6b7280;
    font-size: 13px;
  }

  .scores-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .score-item {
    padding: 12px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.05);
  }

  .score-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }

  .bot-info {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .rank {
    font-size: 11px;
    font-weight: 600;
    color: #6b7280;
    min-width: 24px;
  }

  .bot-id {
    font-size: 13px;
    font-weight: 500;
    font-family: monospace;
  }

  .badges {
    display: flex;
    align-items: center;
    gap: 6px;
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

  .badge.concern {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .tier-badge {
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
  }

  .composite-score {
    display: flex;
    align-items: baseline;
    gap: 4px;
    margin-bottom: 8px;
  }

  .score-value {
    font-size: 28px;
    font-weight: 700;
    line-height: 1;
  }

  .score-label {
    font-size: 11px;
    color: #6b7280;
    text-transform: uppercase;
  }

  .components-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
  }

  .component {
    text-align: center;
  }

  .component-label {
    display: block;
    font-size: 9px;
    color: #6b7280;
    text-transform: uppercase;
    margin-bottom: 2px;
  }

  .component-value {
    font-size: 13px;
    font-weight: 500;
  }

  .concern-info {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 8px;
    padding: 6px 8px;
    background: rgba(239, 68, 68, 0.1);
    border-radius: 4px;
    font-size: 11px;
    color: #ef4444;
  }

  .compact .score-item {
    padding: 8px;
  }

  .compact .composite-score {
    margin-bottom: 4px;
  }

  .compact .score-value {
    font-size: 20px;
  }
</style>

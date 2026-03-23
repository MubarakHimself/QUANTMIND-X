<script lang="ts">
  /**
   * BotDetailPage - Expanded Bot Detail View
   *
   * Shows: session mask, force_close_hour, overnight_hold,
   * daily loss cap bar, equity exposure
   */
  import BreadcrumbNav from './BreadcrumbNav.svelte';
  import GlassTile from './GlassTile.svelte';
  import { selectedBot, loadBotDetails, selectBot } from '$lib/stores/trading';
  import { onMount } from 'svelte';
  import { TrendingUp, TrendingDown, Clock, Moon, AlertTriangle, Activity } from 'lucide-svelte';

  let loading = true;
  let detail = $selectedBot;

  $: if ($selectedBot) {
    detail = $selectedBot;
    loading = false;
  } else if ($selectedBot === null) {
    loading = false;
  }

  // Load details if we have a selected bot but no detail yet
  $: if ($selectedBot && !detail) {
    loadBotDetails($selectedBot.bot_id).then(() => {
      loading = false;
    });
  }

  // Session hours labels
  const hours = Array.from({ length: 24 }, (_, i) => i);

  function isSessionActive(mask: number[], hour: number): boolean {
    return mask && mask[hour] === 1;
  }

  function formatPnl(value: number): string {
    return value >= 0 ? `+$${value.toFixed(2)}` : `-$${Math.abs(value).toFixed(2)}`;
  }

  function formatTime(hour: number | null): string {
    if (hour === null) return 'N/A';
    return `${hour.toString().padStart(2, '0')}:45 UTC`;
  }
</script>

<div class="bot-detail">
  {#if detail}
    <BreadcrumbNav botName={detail.ea_name} />

    <div class="detail-grid">
      <!-- Session Mask -->
      <GlassTile>
        <div class="section">
          <h3>Session Mask</h3>
          <div class="session-grid">
            {#each hours as hour}
              <div
                class="hour-cell"
                class:active={isSessionActive(detail.session_mask, hour)}
                class:current={new Date().getUTCHours() === hour}
                title="{hour}:00 UTC"
              >
                {hour.toString().padStart(2, '0')}
              </div>
            {/each}
          </div>
          <div class="legend">
            <span class="legend-item"><span class="dot active"></span> Active</span>
            <span class="legend-item"><span class="dot current"></span> Current Hour</span>
          </div>
        </div>
      </GlassTile>

      <!-- Trading Parameters -->
      <GlassTile>
        <div class="section">
          <h3>Trading Parameters</h3>
          <div class="params-list">
            <div class="param">
              <Clock size={14} />
              <span class="label">Force Close:</span>
              <span class="value">{formatTime(detail.force_close_hour)}</span>
            </div>
            <div class="param">
              <Moon size={14} />
              <span class="label">Overnight Hold:</span>
              <span class="value" class:enabled={detail.overnight_hold}>
                {detail.overnight_hold ? 'Enabled' : 'Disabled'}
              </span>
            </div>
          </div>
        </div>
      </GlassTile>

      <!-- Daily Loss Cap -->
      <GlassTile>
        <div class="section">
          <h3>
            <AlertTriangle size={14} />
            Daily Loss Cap
          </h3>
          <div class="loss-cap-bar">
            <div class="bar-track">
              <div
                class="bar-fill"
                class:danger={detail.current_loss_pct > 80}
                class:warning={detail.current_loss_pct > 50}
                style="width: {Math.min(detail.current_loss_pct, 100)}%"
              ></div>
              <div class="bar-marker" style="left: 80%"></div>
            </div>
            <div class="bar-labels">
              <span class="current">{detail.current_loss_pct.toFixed(1)}%</span>
              <span class="limit">${detail.daily_loss_cap.toFixed(0)}</span>
            </div>
          </div>
        </div>
      </GlassTile>

      <!-- Equity Exposure -->
      <GlassTile>
        <div class="section">
          <h3>
            <Activity size={14} />
            Equity Exposure
          </h3>
          <div class="exposure-value" class:positive={detail.equity_exposure > 0} class:negative={detail.equity_exposure < 0}>
            {#if detail.equity_exposure > 0}
              <TrendingUp size={20} />
            {:else if detail.equity_exposure < 0}
              <TrendingDown size={20} />
            {/if}
            <span class="value">{formatPnl(detail.equity_exposure)}</span>
          </div>
          <div class="exposure-details">
            <div class="detail-row">
              <span>Open Positions:</span>
              <span>{detail.open_positions}</span>
            </div>
            <div class="detail-row">
              <span>Current P&L:</span>
              <span class:positive={detail.current_pnl > 0} class:negative={detail.current_pnl < 0}>
                {formatPnl(detail.current_pnl)}
              </span>
            </div>
          </div>
        </div>
      </GlassTile>
    </div>
  {:else}
    <div class="no-selection">
      <p>Select a bot to view details</p>
    </div>
  {/if}
</div>

<style>
  .bot-detail {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 16px;
    width: 100%;
  }

  .detail-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 16px;
  }

  .section {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  h3 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    color: #aaa;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0;
  }

  .session-grid {
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: 4px;
  }

  .hour-cell {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: #555;
    text-align: center;
    padding: 4px;
    border-radius: 3px;
    background: rgba(0, 0, 0, 0.2);
    transition: all 0.15s ease;
  }

  .hour-cell.active {
    background: rgba(0, 200, 150, 0.3);
    color: #00c896;
  }

  .hour-cell.current {
    border: 1px solid #00d4ff;
    color: #00d4ff;
  }

  .legend {
    display: flex;
    gap: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #666;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .dot {
    width: 8px;
    height: 8px;
    border-radius: 2px;
  }

  .dot.active {
    background: rgba(0, 200, 150, 0.5);
  }

  .dot.current {
    border: 1px solid #00d4ff;
  }

  .params-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .param {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #888;
  }

  .param .label {
    color: #666;
  }

  .param .value {
    color: #e0e0e0;
  }

  .param .value.enabled {
    color: #00c896;
  }

  .loss-cap-bar {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .bar-track {
    position: relative;
    height: 12px;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 6px;
    overflow: hidden;
  }

  .bar-fill {
    height: 100%;
    background: #00c896;
    border-radius: 6px;
    transition: width 0.3s ease;
  }

  .bar-fill.warning {
    background: #ffaa00;
  }

  .bar-fill.danger {
    background: #ff3b3b;
  }

  .bar-marker {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 2px;
    background: rgba(255, 255, 255, 0.3);
  }

  .bar-labels {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
  }

  .bar-labels .current {
    color: #e0e0e0;
  }

  .bar-labels .limit {
    color: #666;
  }

  .exposure-value {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 20px;
    font-weight: 600;
    color: #888;
  }

  .exposure-value.positive {
    color: #00c896;
  }

  .exposure-value.negative {
    color: #ff3b3b;
  }

  .exposure-details {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 8px;
  }

  .detail-row {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #666;
  }

  .detail-row .positive {
    color: #00c896;
  }

  .detail-row .negative {
    color: #ff3b3b;
  }

  .no-selection {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 48px;
    color: #555;
    font-family: 'JetBrains Mono', monospace;
  }
</style>

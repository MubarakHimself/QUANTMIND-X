<script lang="ts">
  import { RefreshCw, Newspaper, Bell } from "lucide-svelte";

  export let tradingStatus: "active" | "paused" | "kill-zone" = "active";
  export let countdown: { targetEvent: any } = { targetEvent: null };
  export let autoRefresh = false;

  export let onRefresh: () => void;
  export let onToggleAutoRefresh: () => void;

  function formatCountdown(): string {
    if (!countdown.targetEvent) return "";
    const target = new Date(countdown.targetEvent.datetime).getTime();
    const now = Date.now();
    const diff = target - now;

    if (diff <= 0) return "NOW";

    const hours = Math.floor(diff / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    const seconds = Math.floor((diff % (1000 * 60)) / 1000);

    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds}s`;
    } else {
      return `${seconds}s`;
    }
  }
</script>

<div class="news-header">
  <div class="header-left">
    <Newspaper size={24} class="news-icon" />
    <div>
      <h2>Economic Calendar</h2>
      <p>News events and kill zone monitoring</p>
    </div>
  </div>
  <div class="header-actions">
    <!-- Trading Status -->
    <div
      class="trading-status"
      class:active={tradingStatus === "active"}
      class:paused={tradingStatus === "paused"}
      class:kill-zone={tradingStatus === "kill-zone"}
    >
      <div class="status-indicator"></div>
      <span>
        {#if tradingStatus === "active"}
          Active
        {:else if tradingStatus === "paused"}
          Paused
        {:else}
          Kill Zone
        {/if}
      </span>
    </div>

    <!-- Next Event Countdown -->
    {#if countdown.targetEvent}
      <div class="countdown-display">
        <Bell size={14} />
        <span class="countdown-label">{countdown.targetEvent.name} in</span>
        <span class="countdown-time">{formatCountdown()}</span>
      </div>
    {/if}

    <!-- Auto Refresh -->
    <button
      class="btn"
      on:click={onToggleAutoRefresh}
      class:active={autoRefresh}
    >
      <RefreshCw size={14} />
      <span>Auto</span>
    </button>

    <!-- Refresh Button -->
    <button class="btn" on:click={onRefresh}>
      <RefreshCw size={14} />
      <span>Refresh</span>
    </button>
  </div>
</div>

<style>
  .news-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px;
    background: #1e293b;
    border-radius: 8px;
    margin-bottom: 16px;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .header-left h2 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
    color: #f8fafc;
  }

  .header-left p {
    margin: 4px 0 0;
    font-size: 12px;
    color: #94a3b8;
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .trading-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    background: #334155;
  }

  .trading-status.active {
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
  }

  .trading-status.paused {
    background: rgba(251, 191, 36, 0.2);
    color: #fbbf24;
  }

  .trading-status.kill-zone {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .status-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: currentColor;
  }

  .countdown-display {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    background: #334155;
    border-radius: 6px;
    font-size: 12px;
  }

  .countdown-label {
    color: #94a3b8;
  }

  .countdown-time {
    color: #f8fafc;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }

  .btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border: none;
    border-radius: 6px;
    font-size: 12px;
    cursor: pointer;
    background: #334155;
    color: #f8fafc;
    transition: all 0.2s;
  }

  .btn:hover {
    background: #475569;
  }

  .btn.active {
    background: #3b82f6;
  }
</style>

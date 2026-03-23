<script lang="ts">
  /**
   * CalendarGateTile - Calendar Gate Status Tile
   *
   * Displays upcoming high-impact news events and active blackout windows.
   * AC #3: Shows upcoming high-impact news events with configured blackout windows,
   * and currently active blackout windows showing which strategies are affected.
   */

  import { onMount, onDestroy } from 'svelte';
  import { Calendar, Clock, AlertTriangle, ShieldOff, Zap } from 'lucide-svelte';
  import {
    calendarGateStore,
    calendarGateData,
    calendarGateLoading,
    calendarGateError,
    type CalendarGateData
  } from '$lib/stores';

  let data: CalendarGateData | null = $state(null);
  let loading = $state(false);
  let error: string | null = $state(null);

  // Subscribe to store
  const unsubData = calendarGateData.subscribe(v => data = v);
  const unsubLoading = calendarGateLoading.subscribe(v => loading = v);
  const unsubError = calendarGateError.subscribe(v => error = v);

  onMount(() => {
    calendarGateStore.startPolling(5000);
  });

  onDestroy(() => {
    calendarGateStore.stopPolling();
    unsubData();
    unsubLoading();
    unsubError();
  });

  function getImpactColor(impact: string): string {
    switch (impact) {
      case 'high': return '#ff3b3b';
      case 'medium': return '#ffb700';
      default: return '#00d4ff';
    }
  }

  function formatTime(isoString: string): string {
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    });
  }

  function formatDate(isoString: string): string {
    const date = new Date(isoString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric'
    });
  }

  function isBlackoutActive(blackout: { start_utc: string; end_utc: string }): boolean {
    const now = new Date();
    const start = new Date(blackout.start_utc);
    const end = new Date(blackout.end_utc);
    return now >= start && now <= end;
  }
</script>

<div class="calendar-tile">
  <div class="tile-header">
    <h3 class="tile-title">
      <Calendar size={16} />
      Calendar Gate
    </h3>
    {#if data?.events?.length}
      <span class="event-count">{data.events.length} events</span>
    {/if}
  </div>

  <div class="tile-content">
    {#if loading}
      <div class="loading-state">
        <span class="loading-text">Loading...</span>
      </div>
    {:else if error}
      <div class="error-state">
        <span class="error-text">{error}</span>
      </div>
    {:else if data}
      <!-- Active Blackouts Section -->
      {#if data.blackouts && data.blackouts.length > 0}
        <div class="section">
          <h4 class="section-title">
            <ShieldOff size={14} />
            Active Blackouts
          </h4>
          <div class="blackout-list">
            {#each data.blackouts as blackout}
              <div class="blackout-item" class:active={isBlackoutActive(blackout)}>
                <div class="blackout-status">
                  {#if isBlackoutActive(blackout)}
                    <span class="status-badge active">ACTIVE</span>
                  {:else}
                    <span class="status-badge">UPCOMING</span>
                  {/if}
                </div>
                <div class="blackout-time">
                  <Clock size={12} />
                  {formatTime(blackout.start_utc)} - {formatTime(blackout.end_utc)}
                </div>
                <div class="affected-strategies">
                  {#each blackout.affected_strategies as strategy}
                    <span class="strategy-tag">{strategy}</span>
                  {/each}
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Upcoming Events Section -->
      <div class="section">
        <h4 class="section-title">
          <Zap size={14} />
          High-Impact News
        </h4>
        {#if data.events && data.events.length > 0}
          <div class="event-list">
            {#each data.events.slice(0, 5) as event}
              <div class="event-item">
                <div class="event-impact" style="color: {getImpactColor(event.impact)}">
                  <AlertTriangle size={12} />
                  {event.impact.toUpperCase()}
                </div>
                <div class="event-details">
                  <span class="event-title">{event.title}</span>
                  <span class="event-time">
                    {formatDate(event.event_time)} {formatTime(event.event_time)} UTC
                  </span>
                </div>
                {#if event.currencies && event.currencies.length > 0}
                  <div class="event-currencies">
                    {#each event.currencies as currency}
                      <span class="currency-tag">{currency}</span>
                    {/each}
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        {:else}
          <div class="empty-events">
            <span>No upcoming high-impact events</span>
          </div>
        {/if}
      </div>
    {/if}
  </div>
</div>

<style>
  .calendar-tile {
    background: rgba(8, 13, 20, 0.35);
    backdrop-filter: blur(16px) saturate(120%);
    -webkit-backdrop-filter: blur(16px) saturate(120%);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 8px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-height: 200px;
  }

  .tile-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .tile-title {
    font-size: 14px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.9);
    margin: 0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .event-count {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.4);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .tile-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 12px;
    overflow-y: auto;
  }

  .loading-state,
  .error-state {
    display: flex;
    align-items: center;
    justify-content: center;
    flex: 1;
  }

  .loading-text {
    color: rgba(255, 255, 255, 0.4);
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .error-text {
    color: #ff3b3b;
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .section {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .section-title {
    font-size: 11px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.6);
    margin: 0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .blackout-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .blackout-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 8px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 4px;
    border-left: 3px solid rgba(255, 183, 0, 0.3);
  }

  .blackout-item.active {
    border-left-color: #ff3b3b;
    background: rgba(255, 59, 59, 0.1);
  }

  .blackout-status {
    display: flex;
    align-items: center;
  }

  .status-badge {
    font-size: 9px;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 3px;
    background: rgba(255, 183, 0, 0.2);
    color: #ffb700;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .status-badge.active {
    background: rgba(255, 59, 59, 0.2);
    color: #ff3b3b;
  }

  .blackout-time {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: rgba(255, 255, 255, 0.7);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .affected-strategies {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 4px;
  }

  .strategy-tag {
    font-size: 9px;
    padding: 2px 6px;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 3px;
    color: #00d4ff;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .event-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .event-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 8px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 4px;
  }

  .event-impact {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 9px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .event-details {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .event-title {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.9);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .event-time {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.5);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .event-currencies {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 4px;
  }

  .currency-tag {
    font-size: 9px;
    padding: 2px 5px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 2px;
    color: rgba(255, 255, 255, 0.6);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .empty-events {
    padding: 12px;
    text-align: center;
    color: rgba(255, 255, 255, 0.4);
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }
</style>

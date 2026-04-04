<script lang="ts">
  /**
   * EconomicCalendarPanel - Economic Calendar Full View
   *
   * P3 Priority: Compact panel focused on blackout visibility
   * - Calendar event list for the day
   * - Impact color coding: HIGH=red, MEDIUM=amber, LOW=grey
   * - Blackout windows highlighted
   * - Time until each event countdown
   * - Currency flags/badges
   */

  import { onMount, onDestroy } from 'svelte';
  import { Calendar, Clock, AlertTriangle, ShieldOff, ChevronLeft, ChevronRight } from 'lucide-svelte';
  import {
    economicCalendarStore,
    economicCalendarData,
    economicCalendarEvents,
    economicCalendarBlackouts,
    economicCalendarLoading,
    economicCalendarError,
    getImpactColor,
    getCurrencyBadge,
    getTimeUntilEvent,
    isBlackoutActive,
    getNextEvent,
    type EconomicEvent,
    type BlackoutWindow
  } from '$lib/stores/economicCalendarStore';

  let selectedDate = $state<string | null>(null);
  let data = $state<typeof economicCalendarData extends import('svelte/store').Readable<infer T> ? T : null>(null);
  let events = $state<EconomicEvent[]>([]);
  let blackouts = $state<BlackoutWindow[]>([]);
  let loading = $state(false);
  let error = $state<string | null>(null);

  // Subscribe to store
  const unsubData = economicCalendarData.subscribe(v => data = v as typeof data);
  const unsubEvents = economicCalendarEvents.subscribe(v => events = v);
  const unsubBlackouts = economicCalendarBlackouts.subscribe(v => blackouts = v);
  const unsubLoading = economicCalendarLoading.subscribe(v => loading = v);
  const unsubError = economicCalendarError.subscribe(v => error = v);

  onMount(() => {
    economicCalendarStore.startPolling(30000); // Poll every 30 seconds
  });

  onDestroy(() => {
    economicCalendarStore.stopPolling();
    unsubData();
    unsubEvents();
    unsubBlackouts();
    unsubLoading();
    unsubError();
  });

  // Date navigation
  function navigateDate(days: number) {
    const current = data?.date ? new Date(data.date) : new Date();
    current.setDate(current.getDate() + days);
    const dateStr = current.toISOString().split('T')[0];
    selectedDate = dateStr;
    economicCalendarStore.fetchByDate(dateStr);
  }

  function goToToday() {
    selectedDate = null;
    economicCalendarStore.fetchToday();
  }

  // Format time for display
  function formatTime(isoString: string): string {
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    });
  }

  // Format date for header
  function formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric'
    });
  }

  // Check if event is in blackout
  function isEventInBlackout(event: EconomicEvent): boolean {
    const eventTime = new Date(event.time);
    return blackouts.some(bo => {
      const start = new Date(bo.start);
      const end = new Date(bo.end);
      return eventTime >= start && eventTime <= end;
    });
  }

  // Get active blackouts
  function getActiveBlackouts(): BlackoutWindow[] {
    return blackouts.filter(bo => isBlackoutActive(bo));
  }

  // Get upcoming blackouts (not started yet)
  function getUpcomingBlackouts(): BlackoutWindow[] {
    const now = new Date();
    return blackouts
      .filter(bo => new Date(bo.start) > now)
      .sort((a, b) => new Date(a.start).getTime() - new Date(b.start).getTime());
  }
</script>

<div class="calendar-panel">
  <div class="panel-header">
    <h3 class="panel-title">
      <Calendar size={16} />
      Economic Calendar
    </h3>
    <div class="date-navigation">
      <button class="nav-btn" onclick={() => navigateDate(-1)} aria-label="Previous day">
        <ChevronLeft size={14} />
      </button>
      <button class="today-btn" onclick={goToToday}>
        {data?.date ? formatDate(data.date) : 'Today'}
      </button>
      <button class="nav-btn" onclick={() => navigateDate(1)} aria-label="Next day">
        <ChevronRight size={14} />
      </button>
    </div>
  </div>

  <div class="panel-content">
    {#if loading && !data}
      <div class="loading-state">
        <span class="loading-text">Loading calendar...</span>
      </div>
    {:else if error}
      <div class="error-state">
        <AlertTriangle size={16} />
        <span class="error-text">{error}</span>
      </div>
    {:else if data}
      <!-- Active Blackouts Banner -->
      {#if getActiveBlackouts().length > 0}
        <div class="active-blackouts-banner">
          <ShieldOff size={14} />
          <span class="banner-text">
            {getActiveBlackouts().length} Active Blackout{getActiveBlackouts().length > 1 ? 's' : ''}
          </span>
        </div>
      {/if}

      <!-- Blackout Windows Section -->
      {#if blackouts.length > 0}
        <div class="section blackouts-section">
          <h4 class="section-title">
            <ShieldOff size={12} />
            Blackout Windows
          </h4>
          <div class="blackout-list">
            {#each blackouts as blackout}
              <div class="blackout-item" class:active={isBlackoutActive(blackout)}>
                <div class="blackout-time">
                  <Clock size={11} />
                  {formatTime(blackout.start)} - {formatTime(blackout.end)}
                </div>
                <div class="blackout-currency">
                  <span class="currency-badge">{getCurrencyBadge(blackout.currency)}</span>
                  <span class="currency-code">{blackout.currency}</span>
                </div>
                {#if blackout.reason}
                  <div class="blackout-reason">{blackout.reason}</div>
                {/if}
              </div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Events List -->
      <div class="section events-section">
        <h4 class="section-title">
          <Clock size={12} />
          Events ({events.length})
        </h4>

        {#if events.length === 0}
          <div class="empty-state">
            <span>No events scheduled</span>
          </div>
        {:else}
          <div class="event-list">
            {#each events as event}
              {@const impactColor = getImpactColor(event.impact)}
              {@const timeUntil = getTimeUntilEvent(event.time)}
              {@const inBlackout = isEventInBlackout(event)}
              <div
                class="event-item"
                class:blackout={inBlackout}
                class:high-impact={event.impact === 'high'}
              >
                <div class="event-indicator" style="background-color: {impactColor}"></div>
                <div class="event-content">
                  <div class="event-header">
                    <div class="event-currency">
                      <span class="currency-badge">{getCurrencyBadge(event.currency)}</span>
                      <span class="currency-code">{event.currency}</span>
                    </div>
                    <div class="event-countdown" class:past={timeUntil === 'Past'}>
                      {#if timeUntil !== 'Past'}
                        <Clock size={10} />
                        {timeUntil}
                      {:else}
                        PAST
                      {/if}
                    </div>
                  </div>
                  <div class="event-name">{event.event_name}</div>
                  <div class="event-meta">
                    <div class="event-impact-badge" style="color: {impactColor}">
                      {event.impact.toUpperCase()}
                    </div>
                    <div class="event-time">{formatTime(event.time)} UTC</div>
                  </div>
                  {#if event.previous || event.forecast}
                    <div class="event-values">
                      {#if event.previous}
                        <span class="value previous">Prev: {event.previous}</span>
                      {/if}
                      {#if event.forecast}
                        <span class="value forecast">Fcst: {event.forecast}</span>
                      {/if}
                    </div>
                  {/if}
                  {#if inBlackout}
                    <div class="blackout-tag">
                      <ShieldOff size={10} />
                      BLACKOUT
                    </div>
                  {/if}
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    {:else}
      <div class="empty-state">
        <span>No calendar data available</span>
      </div>
    {/if}
  </div>
</div>

<style>
  .calendar-panel {
    background: rgba(8, 13, 20, 0.35);
    backdrop-filter: blur(16px) saturate(120%);
    -webkit-backdrop-filter: blur(16px) saturate(120%);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    min-height: 300px;
    max-height: 500px;
    overflow: hidden;
  }

  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    border-bottom: 1px solid rgba(0, 212, 255, 0.08);
    background: rgba(0, 0, 0, 0.2);
  }

  .panel-title {
    font-size: 13px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.9);
    margin: 0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .date-navigation {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .nav-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 4px;
    color: rgba(255, 255, 255, 0.7);
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .nav-btn:hover {
    background: rgba(0, 212, 255, 0.1);
    border-color: rgba(0, 212, 255, 0.3);
    color: #00d4ff;
  }

  .today-btn {
    padding: 4px 10px;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 4px;
    color: #00d4ff;
    font-size: 11px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .today-btn:hover {
    background: rgba(0, 212, 255, 0.2);
    border-color: rgba(0, 212, 255, 0.4);
  }

  .panel-content {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .loading-state,
  .error-state,
  .empty-state {
    display: flex;
    align-items: center;
    justify-content: center;
    flex: 1;
    gap: 8px;
    padding: 24px;
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

  .active-blackouts-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: rgba(255, 59, 59, 0.15);
    border: 1px solid rgba(255, 59, 59, 0.3);
    border-radius: 6px;
    color: #ff3b3b;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }

  .banner-text {
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .section {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .section-title {
    font-size: 10px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.5);
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
    padding: 8px 10px;
    background: rgba(0, 0, 0, 0.25);
    border-radius: 4px;
    border-left: 3px solid rgba(255, 183, 0, 0.4);
  }

  .blackout-item.active {
    border-left-color: #ff3b3b;
    background: rgba(255, 59, 59, 0.1);
  }

  .blackout-time {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: rgba(255, 255, 255, 0.8);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .blackout-currency {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .currency-badge {
    font-size: 10px;
    padding: 2px 5px;
    background: rgba(0, 212, 255, 0.15);
    border: 1px solid rgba(0, 212, 255, 0.25);
    border-radius: 3px;
    color: #00d4ff;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-weight: 600;
  }

  .currency-code {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.5);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .blackout-reason {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.6);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-style: italic;
  }

  .event-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .event-item {
    display: flex;
    gap: 10px;
    padding: 10px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 6px;
    transition: all 0.15s ease;
  }

  .event-item:hover {
    background: rgba(0, 0, 0, 0.3);
  }

  .event-item.blackout {
    background: rgba(255, 59, 59, 0.08);
    border: 1px solid rgba(255, 59, 59, 0.2);
  }

  .event-item.high-impact {
    border-left: 3px solid #ff3b3b;
  }

  .event-indicator {
    width: 3px;
    border-radius: 2px;
    flex-shrink: 0;
  }

  .event-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 0;
  }

  .event-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }

  .event-currency {
    display: flex;
    align-items: center;
    gap: 5px;
  }

  .event-countdown {
    display: flex;
    align-items: center;
    gap: 3px;
    font-size: 10px;
    color: #00d4ff;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-weight: 600;
  }

  .event-countdown.past {
    color: rgba(255, 255, 255, 0.3);
  }

  .event-name {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.9);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .event-meta {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .event-impact-badge {
    font-size: 9px;
    font-weight: 700;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    letter-spacing: 0.5px;
  }

  .event-time {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.4);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .event-values {
    display: flex;
    gap: 10px;
    margin-top: 2px;
  }

  .value {
    font-size: 10px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .value.previous {
    color: rgba(255, 255, 255, 0.5);
  }

  .value.forecast {
    color: rgba(0, 212, 255, 0.8);
  }

  .blackout-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    margin-top: 4px;
    padding: 2px 6px;
    background: rgba(255, 59, 59, 0.15);
    border: 1px solid rgba(255, 59, 59, 0.3);
    border-radius: 3px;
    color: #ff3b3b;
    font-size: 9px;
    font-weight: 700;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    letter-spacing: 0.5px;
  }
</style>

<script lang="ts">
  import { createEventDispatcher, onMount, onDestroy, tick } from 'svelte';
  import {
    Newspaper, Calendar, Clock, AlertTriangle, Shield, Settings as SettingsIcon,
    RefreshCw, Play, Pause, SkipForward, ChevronDown, ChevronUp, X,
    Filter, TrendingUp, Globe, DollarSign, Zap, Bell, BellOff,
    Info, BarChart3, Activity, Eye, CheckCircle, XCircle
  } from 'lucide-svelte';

  const dispatch = createEventDispatcher();

  // Types for News Events and Kill Zones
  interface NewsEvent {
    id: string;
    name: string;
    currency: string;
    impact: 'high' | 'medium' | 'low';
    date_time: string;
    actual?: string;
    forecast?: string;
    previous?: string;
    affected_pairs: string[];
    description?: string;
  }

  interface KillZone {
    event_id: string;
    start_time: string;
    end_time: string;
    is_active: boolean;
  }

  // View state
  let activeTab: 'calendar' | 'timeline' | 'settings' = 'calendar';
  let selectedEvent: NewsEvent | null = null;
  let detailPanelOpen = false;
  let calendarView: 'list' | 'weekly' | 'monthly' = 'list';
  let autoRefresh = true;
  let refreshInterval: number | null = null;

  // Trading state
  let tradingStatus = 'active' as 'active' | 'paused' | 'kill-zone';
  let nextHighImpactEvent: NewsEvent | null = null;

  // Kill zone settings
  let killZoneSettings = {
    enabled: true,
    duration: 30, // minutes before news
    autoPause: true,
    resumeDelay: 15 // minutes after news
  };

  // Filter settings
  let filters = {
    impactThreshold: 'high' as 'high' | 'medium' | 'low', // 'high' = only high, 'medium' = high+medium, 'low' = all
    currencies: ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF'] as string[]
  };

  // Mock economic calendar data
  let newsEvents: NewsEvent[] = [];
  let filteredEvents: NewsEvent[] = [];
  let killZones: KillZone[] = [];

  // Countdown state
  let countdown = {
    hours: 0,
    minutes: 0,
    seconds: 0,
    targetEvent: null as NewsEvent | null
  };

  // Current kill zone status
  let currentKillZone: KillZone | null = null;

  onMount(() => {
    loadNewsEvents();
    updateKillZones();
    startCountdown();

    if (autoRefresh) {
      refreshInterval = window.setInterval(() => {
        loadNewsEvents();
        updateKillZones();
      }, 60000); // Refresh every minute
    }
  });

  onDestroy(() => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }
  });

  // Load news events (mock data for now, will connect to API)
  async function loadNewsEvents() {
    try {
      // TODO: Replace with actual API call
      // const res = await fetch('http://localhost:8000/api/news/calendar');
      // if (res.ok) {
      //   newsEvents = await res.json();
      // }

      // Mock data for development
      const now = new Date();
      const today = now.toISOString().split('T')[0];

      newsEvents = [
        {
          id: 'nfp-001',
          name: 'Non-Farm Payrolls',
          currency: 'USD',
          impact: 'high',
          date_time: new Date(now.getTime() + 2 * 60 * 60 * 1000).toISOString(), // 2 hours from now
          actual: undefined,
          forecast: '200K',
          previous: '175K',
          affected_pairs: ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF'],
          description: 'The change in the number of employed people during the previous month, excluding the farming industry. Job creation is an important leading indicator of consumer spending, which accounts for a majority of overall economic activity.'
        },
        {
          id: 'cpi-001',
          name: 'CPI m/m',
          currency: 'EUR',
          impact: 'high',
          date_time: new Date(now.getTime() + 4 * 60 * 60 * 1000).toISOString(),
          actual: undefined,
          forecast: '0.3%',
          previous: '0.2%',
          affected_pairs: ['EURUSD', 'EURGBP', 'EURJPY'],
          description: 'Consumer Price Index is a measure of the average price level of a fixed basket of goods and services purchased by consumers. Monthly changes in CPI represent the rate of inflation.'
        },
        {
          id: 'ir-001',
          name: 'Interest Rate Decision',
          currency: 'GBP',
          impact: 'high',
          date_time: new Date(now.getTime() + 6 * 60 * 60 * 1000).toISOString(),
          actual: undefined,
          forecast: '5.25%',
          previous: '5.25%',
          affected_pairs: ['GBPUSD', 'EURGBP', 'GBPJPY'],
          description: 'The Bank of England\'s MPC announces its interest rate decision. Short-term interest rates are the primary factor in currency valuation.'
        },
        {
          id: 'adp-001',
          name: 'ADP Non-Farm Employment Change',
          currency: 'USD',
          impact: 'medium',
          date_time: new Date(now.getTime() + 1 * 60 * 60 * 1000).toISOString(),
          actual: undefined,
          forecast: '150K',
          previous: '130K',
          affected_pairs: ['EURUSD', 'GBPUSD', 'USDJPY'],
          description: 'The estimated change in the number of employed people during the previous month, excluding the farming industry and government. ADP provides a sneak peek into the official NFP report.'
        },
        {
          id: 'pmi-001',
          name: 'Manufacturing PMI',
          currency: 'USD',
          impact: 'low',
          date_time: new Date(now.getTime() + 3 * 60 * 60 * 1000).toISOString(),
          actual: undefined,
          forecast: '50.5',
          previous: '50.2',
          affected_pairs: ['EURUSD', 'USDJPY'],
          description: 'The Purchasing Managers\' Index measures the activity level of purchasing managers in the manufacturing sector. A reading above 50 indicates expansion.'
        },
        {
          id: 'retail-001',
          name: 'Retail Sales m/m',
          currency: 'AUD',
          impact: 'medium',
          date_time: new Date(now.getTime() + 5 * 60 * 60 * 1000).toISOString(),
          actual: undefined,
          forecast: '0.4%',
          previous: '0.2%',
          affected_pairs: ['AUDUSD', 'EURAUD', 'AUDJPY'],
          description: 'The change in the total value of sales at the retail level. It is the foremost indicator of consumer spending, which accounts for the majority of overall economic activity.'
        },
        {
          id: 'gdp-001',
          name: 'GDP q/q',
          currency: 'CAD',
          impact: 'medium',
          date_time: new Date(now.getTime() + 8 * 60 * 60 * 1000).toISOString(),
          actual: undefined,
          forecast: '0.4%',
          previous: '0.3%',
          affected_pairs: ['USDCAD', 'CADJPY'],
          description: 'Gross Domestic Product is the broadest measure of economic activity and is a key indicator of economic health. The quarterly percent changes measure the growth rate.'
        },
        {
          id: 'fomc-001',
          name: 'FOMC Meeting Minutes',
          currency: 'USD',
          impact: 'high',
          date_time: new Date(now.getTime() + 24 * 60 * 60 * 1000).toISOString(), // Tomorrow
          actual: undefined,
          forecast: '',
          previous: '',
          affected_pairs: ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'],
          description: 'The Federal Open Market Committee publishes the minutes of its meetings. Detailed records of the FOMC\'s policy discussions provide insights into future policy decisions.'
        }
      ];

      applyFilters();
      findNextHighImpactEvent();
    } catch (e) {
      console.error('Failed to load news events:', e);
    }
  }

  // Apply filters to news events
  function applyFilters() {
    filteredEvents = newsEvents.filter(event => {
      // Filter by impact threshold
      const impactLevels = ['high', 'medium', 'low'];
      const thresholdIndex = impactLevels.indexOf(filters.impactThreshold);
      const eventIndex = impactLevels.indexOf(event.impact);
      if (eventIndex > thresholdIndex) return false;

      // Filter by selected currencies
      if (!filters.currencies.includes(event.currency)) return false;

      return true;
    });

    // Sort by date/time
    filteredEvents.sort((a, b) =>
      new Date(a.date_time).getTime() - new Date(b.date_time).getTime()
    );
  }

  // Update kill zones based on upcoming events
  function updateKillZones() {
    const now = new Date();
    killZones = [];

    newsEvents.forEach(event => {
      if (event.impact === 'high') {
        const eventTime = new Date(event.date_time);
        const killZoneStart = new Date(eventTime.getTime() - killZoneSettings.duration * 60 * 1000);
        const killZoneEnd = new Date(eventTime.getTime() + killZoneSettings.resumeDelay * 60 * 1000);

        // Check if kill zone is upcoming or active
        if (killZoneEnd > now) {
          killZones.push({
            event_id: event.id,
            start_time: killZoneStart.toISOString(),
            end_time: killZoneEnd.toISOString(),
            is_active: now >= killZoneStart && now < killZoneEnd
          });
        }
      }
    });

    // Check if currently in kill zone
    currentKillZone = killZones.find(kz => kz.is_active) || null;

    // Update trading status based on kill zone
    if (currentKillZone && killZoneSettings.autoPause) {
      tradingStatus = 'kill-zone';
    } else if (tradingStatus === 'kill-zone' && !currentKillZone) {
      tradingStatus = 'active';
    }
  }

  // Find the next high-impact event for countdown
  function findNextHighImpactEvent() {
    const now = new Date();
    const upcomingHighImpact = newsEvents
      .filter(e => e.impact === 'high' && new Date(e.date_time) > now)
      .sort((a, b) => new Date(a.date_time).getTime() - new Date(b.date_time).getTime());

    nextHighImpactEvent = upcomingHighImpact[0] || null;
    countdown.targetEvent = nextHighImpactEvent;
  }

  // Countdown timer
  let countdownInterval: number | null = null;

  function startCountdown() {
    updateCountdown();
    countdownInterval = window.setInterval(() => {
      updateCountdown();
    }, 1000);
  }

  function updateCountdown() {
    if (!countdown.targetEvent) {
      countdown = { hours: 0, minutes: 0, seconds: 0, targetEvent: null };
      return;
    }

    const now = new Date();
    const eventTime = new Date(countdown.targetEvent.date_time);
    const diff = eventTime.getTime() - now.getTime();

    if (diff <= 0) {
      countdown = { hours: 0, minutes: 0, seconds: 0, targetEvent: null };
      // Find next event after this one passes
      findNextHighImpactEvent();
      return;
    }

    countdown.hours = Math.floor(diff / (1000 * 60 * 60));
    countdown.minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    countdown.seconds = Math.floor((diff % (1000 * 60)) / 1000);
  }

  // Toggle trading status
  async function toggleTrading() {
    if (tradingStatus === 'active') {
      tradingStatus = 'paused';
    } else {
      tradingStatus = 'active';
    }

    // TODO: Send to API
    try {
      // await fetch('http://localhost:8000/api/router/trading', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ status: tradingStatus })
      // });
    } catch (e) {
      console.error('Failed to update trading status:', e);
    }
  }

  // Event selection
  function selectEvent(event: NewsEvent) {
    selectedEvent = event;
    detailPanelOpen = true;
  }

  // Close detail panel
  function closeDetailPanel() {
    detailPanelOpen = false;
    selectedEvent = null;
  }

  // Helper functions
  function getImpactColor(impact: string) {
    switch (impact) {
      case 'high': return '#ef4444';
      case 'medium': return '#f59e0b';
      case 'low': return '#10b981';
      default: return '#6b7280';
    }
  }

  function getImpactBadgeClass(impact: string) {
    return `impact-badge impact-${impact}`;
  }

  function getCurrencyFlag(currency: string) {
    const flags: Record<string, string> = {
      USD: '🇺🇸',
      EUR: '🇪🇺',
      GBP: '🇬🇧',
      JPY: '🇯🇵',
      AUD: '🇦🇺',
      CAD: '🇨🇦',
      CHF: '🇨🇭',
      NZD: '🇳🇿'
    };
    return flags[currency] || '🌐';
  }

  function formatDateTime(dateStr: string) {
    const date = new Date(dateStr);
    const today = new Date();
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    const isToday = date.toDateString() === today.toDateString();
    const isTomorrow = date.toDateString() === tomorrow.toDateString();

    const time = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    if (isToday) return `Today ${time}`;
    if (isTomorrow) return `Tomorrow ${time}`;
    return `${date.toLocaleDateString()} ${time}`;
  }

  function formatCountdown() {
    const parts = [];
    if (countdown.hours > 0) parts.push(`${countdown.hours}h`);
    if (countdown.minutes > 0) parts.push(`${countdown.minutes}m`);
    parts.push(`${countdown.seconds}s`);
    return parts.join(' ');
  }

  function isEventInKillZone(event: NewsEvent) {
    if (!killZoneSettings.enabled) return false;
    const killZone = killZones.find(kz => kz.event_id === event.id);
    return killZone?.is_active || false;
  }

  function getEventStatus(event: NewsEvent) {
    const now = new Date();
    const eventTime = new Date(event.date_time);

    if (eventTime < now) return 'passed';
    if (isEventInKillZone(event)) return 'kill-zone';
    return 'upcoming';
  }
</script>

<div class="news-view">
  <!-- Kill Zone Warning Banner -->
  {#if currentKillZone && killZoneSettings.enabled}
    <div class="kill-zone-banner">
      <div class="banner-content">
        <div class="banner-left">
          <AlertTriangle size={20} class="warning-icon" />
          <div class="banner-text">
            <h3>Kill Zone Active</h3>
            <p>High-impact news event approaching. Trading is paused.</p>
          </div>
        </div>
        <div class="banner-right">
          <button class="btn" on:click={() => killZoneSettings.autoPause = false}>
            <Play size={14} />
            <span>Resume Anyway</span>
          </button>
          <button class="btn close" on:click={() => currentKillZone = null}>
            <X size={14} />
          </button>
        </div>
      </div>
    </div>
  {/if}

  <!-- Header -->
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
      <div class="trading-status" class:active={tradingStatus === 'active'} class:paused={tradingStatus === 'paused'} class:kill-zone={tradingStatus === 'kill-zone'}>
        <div class="status-indicator"></div>
        <span>
          {#if tradingStatus === 'active'}
            Active
          {:else if tradingStatus === 'paused'}
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
      <button class="btn" on:click={() => autoRefresh = !autoRefresh} class:active={autoRefresh}>
        <RefreshCw size={14} />
        <span>Auto</span>
      </button>

      <!-- Refresh Button -->
      <button class="btn" on:click={loadNewsEvents}>
        <RefreshCw size={14} />
        <span>Refresh</span>
      </button>
    </div>
  </div>

  <!-- Tabs -->
  <div class="news-tabs">
    <button class="tab" class:active={activeTab === 'calendar'} on:click={() => activeTab = 'calendar'}>
      <Calendar size={14} />
      <span>Calendar</span>
    </button>
    <button class="tab" class:active={activeTab === 'timeline'} on:click={() => activeTab = 'timeline'}>
      <Clock size={14} />
      <span>Timeline</span>
    </button>
    <button class="tab" class:active={activeTab === 'settings'} on:click={() => activeTab = 'settings'}>
      <SettingsIcon size={14} />
      <span>Settings</span>
    </button>
  </div>

  <!-- Tab Content -->
  <div class="news-content">
    {#if activeTab === 'calendar'}
      <!-- Calendar View -->
      <div class="calendar-section">
        <!-- View Toggle -->
        <div class="view-toggle">
          <button class="view-btn" class:active={calendarView === 'list'} on:click={() => calendarView = 'list'}>
            List
          </button>
          <button class="view-btn" class:active={calendarView === 'weekly'} on:click={() => calendarView = 'weekly'}>
            Weekly
          </button>
          <button class="view-btn" class:active={calendarView === 'monthly'} on:click={() => calendarView = 'monthly'}>
            Monthly
          </button>
        </div>

        {#if calendarView === 'list'}
          <!-- List View -->
          <div class="events-list">
            {#each filteredEvents as event}
              <div class="event-card" class:in-kill-zone={isEventInKillZone(event)} on:click={() => selectEvent(event)}>
                <div class="event-left">
                  <div class="event-impact">
                    <span class={getImpactBadgeClass(event.impact)}>{event.impact.toUpperCase()}</span>
                  </div>
                  <div class="event-time">
                    <Clock size={12} />
                    <span>{formatDateTime(event.date_time)}</span>
                  </div>
                </div>

                <div class="event-main">
                  <div class="event-header">
                    <span class="currency-flag">{getCurrencyFlag(event.currency)}</span>
                    <h4 class="event-name">{event.name}</h4>
                    <span class="event-status status-{getEventStatus(event)}">
                      {#if getEventStatus(event) === 'passed'}
                        <CheckCircle size={12} />
                        Passed
                      {:else if getEventStatus(event) === 'kill-zone'}
                        <AlertTriangle size={12} />
                        Kill Zone
                      {:else}
                        <Clock size={12} />
                        Upcoming
                      {/if}
                    </span>
                  </div>

                  <div class="event-details">
                    <span class="affected-pairs">
                      <Globe size={10} />
                      {event.affected_pairs.join(', ')}
                    </span>
                  </div>

                  <div class="event-values">
                    {#if event.actual}
                      <span class="value actual">
                        <span class="label">Actual:</span>
                        <span class="data">{event.actual}</span>
                      </span>
                    {/if}
                    {#if event.forecast}
                      <span class="value forecast">
                        <span class="label">Forecast:</span>
                        <span class="data">{event.forecast}</span>
                      </span>
                    {/if}
                    {#if event.previous}
                      <span class="value previous">
                        <span class="label">Previous:</span>
                        <span class="data">{event.previous}</span>
                      </span>
                    {/if}
                  </div>
                </div>

                <div class="event-right">
                  <button class="icon-btn" title="View details">
                    <Eye size={14} />
                  </button>
                </div>
              </div>
            {/each}

            {#if filteredEvents.length === 0}
              <div class="empty-state">
                <Newspaper size={32} />
                <p>No events match your filters</p>
              </div>
            {/if}
          </div>

        {:else if calendarView === 'weekly'}
          <!-- Weekly View Placeholder -->
          <div class="calendar-grid-view">
            <div class="empty-state">
              <Calendar size={32} />
              <p>Weekly calendar view coming soon</p>
              <small>Switch to list view for full event details</small>
            </div>
          </div>

        {:else}
          <!-- Monthly View Placeholder -->
          <div class="calendar-grid-view">
            <div class="empty-state">
              <Calendar size={32} />
              <p>Monthly calendar view coming soon</p>
              <small>Switch to list view for full event details</small>
            </div>
          </div>
        {/if}
      </div>

    {:else if activeTab === 'timeline'}
      <!-- Timeline View -->
      <div class="timeline-section">
        <div class="timeline-header">
          <h3>Today's Kill Zones</h3>
          <span class="kill-zone-count">{killZones.length} active zones</span>
        </div>

        <div class="timeline-container">
          {#if killZones.length > 0}
            <div class="timeline-track">
              {#each killZones as zone}
                {@const event = newsEvents.find(e => e.id === zone.event_id)}
                {#if event}
                  <div class="timeline-zone" class:active={zone.is_active}>
                    <div class="zone-marker"></div>
                    <div class="zone-content">
                      <span class="zone-time">
                        {formatDateTime(zone.start_time)} - {formatDateTime(zone.end_time)}
                      </span>
                      <div class="zone-event">
                        <span class="currency-flag">{getCurrencyFlag(event.currency)}</span>
                        <span class="event-name">{event.name}</span>
                        <span class={getImpactBadgeClass(event.impact)}>{event.impact.toUpperCase()}</span>
                      </div>
                      {#if zone.is_active}
                        <div class="zone-indicator">
                          <Activity size={10} />
                          <span>Currently active</span>
                        </div>
                      {/if}
                    </div>
                  </div>
                {/if}
              {/each}
            </div>
          {:else}
            <div class="empty-state">
              <Shield size={32} />
              <p>No kill zones scheduled for today</p>
            </div>
          {/if}
        </div>
      </div>

    {:else if activeTab === 'settings'}
      <!-- Settings View -->
      <div class="settings-section">
        <div class="settings-group">
          <h3><Zap size={16} /> Kill Zone Settings</h3>

          <div class="setting-row">
            <label class="setting-label">
              <input type="checkbox" bind:checked={killZoneSettings.enabled} />
              <span>Enable Kill Zones</span>
            </label>
            <small>Automatically pause trading before high-impact news</small>
          </div>

          <div class="setting-row">
            <label class="setting-label">Duration Before News</label>
            <div class="setting-options">
              <button class="option-btn" class:active={killZoneSettings.duration === 15} on:click={() => killZoneSettings.duration = 15}>
                15 min
              </button>
              <button class="option-btn" class:active={killZoneSettings.duration === 30} on:click={() => killZoneSettings.duration = 30}>
                30 min
              </button>
              <button class="option-btn" class:active={killZoneSettings.duration === 60} on:click={() => killZoneSettings.duration = 60}>
                60 min
              </button>
            </div>
          </div>

          <div class="setting-row">
            <label class="setting-label">
              <input type="checkbox" bind:checked={killZoneSettings.autoPause} />
              <span>Auto-Pause Trading</span>
            </label>
            <small>Automatically pause when entering kill zone</small>
          </div>

          <div class="setting-row">
            <label class="setting-label">Resume Delay After News</label>
            <input
              type="number"
              bind:value={killZoneSettings.resumeDelay}
              min="0"
              max="60"
              step="5"
            />
            <span class="unit">minutes</span>
          </div>
        </div>

        <div class="settings-group">
          <h3><Filter size={16} /> News Filters</h3>

          <div class="setting-row">
            <label class="setting-label">Impact Threshold</label>
            <select bind:value={filters.impactThreshold} on:change={applyFilters}>
              <option value="high">High Impact Only</option>
              <option value="medium">High + Medium</option>
              <option value="low">All Events</option>
            </select>
          </div>

          <div class="setting-row">
            <label class="setting-label">Currencies</label>
            <div class="currency-grid">
              {#each ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD'] as currency}
                <label class="currency-checkbox">
                  <input type="checkbox" bind:group={filters.currencies} value={currency} on:change={applyFilters} />
                  <span>{getCurrencyFlag(currency)} {currency}</span>
                </label>
              {/each}
            </div>
          </div>
        </div>

        <div class="settings-group">
          <h3><Activity size={16} /> Trading Controls</h3>

          <div class="setting-row full-width">
            <button
              class="trading-toggle-btn"
              class:active={tradingStatus === 'active'}
              class:paused={tradingStatus === 'paused'}
              on:click={toggleTrading}
            >
              {#if tradingStatus === 'active'}
                <Pause size={16} />
                <span>Pause Trading</span>
              {:else}
                <Play size={16} />
                <span>Resume Trading</span>
              {/if}
            </button>
          </div>
        </div>
      </div>
    {/if}
  </div>

  <!-- Event Detail Panel -->
  {#if detailPanelOpen && selectedEvent}
    <div class="detail-panel-overlay" on:click={closeDetailPanel}>
      <div class="detail-panel" on:click|stopPropagation>
        <div class="detail-header">
          <div class="header-left">
            <span class="currency-flag large">{getCurrencyFlag(selectedEvent.currency)}</span>
            <div>
              <h3>{selectedEvent.name}</h3>
              <p class="event-meta">
                <span class={getImpactBadgeClass(selectedEvent.impact)}>{selectedEvent.impact.toUpperCase()}</span>
                <span class="currency">{selectedEvent.currency}</span>
              </p>
            </div>
          </div>
          <button class="icon-btn" on:click={closeDetailPanel}>
            <X size={18} />
          </button>
        </div>

        <div class="detail-content">
          <!-- Event Time -->
          <div class="detail-section">
            <h4><Clock size={14} /> Schedule</h4>
            <div class="time-display">
              <span class="time-value">{formatDateTime(selectedEvent.date_time)}</span>
              {#if isEventInKillZone(selectedEvent)}
                <span class="kill-zone-badge">
                  <AlertTriangle size={12} />
                  Kill Zone Active
                </span>
              {/if}
            </div>
          </div>

          <!-- Event Values -->
          <div class="detail-section">
            <h4><BarChart3 size={14} /> Values</h4>
            <div class="values-grid">
              {#if selectedEvent.actual}
                <div class="value-card">
                  <span class="value-label">Actual</span>
                  <span class="value-data actual">{selectedEvent.actual}</span>
                </div>
              {/if}
              {#if selectedEvent.forecast}
                <div class="value-card">
                  <span class="value-label">Forecast</span>
                  <span class="value-data forecast">{selectedEvent.forecast}</span>
                </div>
              {/if}
              {#if selectedEvent.previous}
                <div class="value-card">
                  <span class="value-label">Previous</span>
                  <span class="value-data previous">{selectedEvent.previous}</span>
                </div>
              {/if}
            </div>
          </div>

          <!-- Affected Pairs -->
          <div class="detail-section">
            <h4><Globe size={14} /> Affected Trading Pairs</h4>
            <div class="pairs-grid">
              {#each selectedEvent.affected_pairs as pair}
                <span class="pair-badge">{pair}</span>
              {/each}
            </div>
          </div>

          <!-- Description -->
          {#if selectedEvent.description}
            <div class="detail-section">
              <h4><Info size={14} /> Description</h4>
              <p class="event-description">{selectedEvent.description}</p>
            </div>
          {/if}

          <!-- Historical Impact Placeholder -->
          <div class="detail-section">
            <h4><TrendingUp size={14} /> Historical Impact</h4>
            <div class="historical-impact-placeholder">
              <BarChart3 size={32} />
              <p>Historical volatility chart coming soon</p>
              <small>This will show price movement patterns for previous occurrences</small>
            </div>
          </div>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .news-view {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary);
    overflow: hidden;
  }

  /* Kill Zone Banner */
  .kill-zone-banner {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(245, 158, 11, 0.2));
    border-bottom: 1px solid rgba(239, 68, 68, 0.3);
    animation: slideDown 0.3s ease-out;
  }

  @keyframes slideDown {
    from {
      opacity: 0;
      transform: translateY(-100%);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .banner-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 24px;
  }

  .banner-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .warning-icon {
    color: #ef4444;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .banner-text h3 {
    margin: 0;
    font-size: 14px;
    color: #ef4444;
    font-weight: 600;
  }

  .banner-text p {
    margin: 2px 0 0;
    font-size: 12px;
    color: var(--text-muted);
  }

  .banner-right {
    display: flex;
    gap: 8px;
  }

  /* Header */
  .news-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 24px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .news-icon {
    color: var(--accent-primary);
  }

  .header-left h2 {
    margin: 0;
    font-size: 18px;
    color: var(--text-primary);
  }

  .header-left p {
    margin: 2px 0 0;
    font-size: 12px;
    color: var(--text-muted);
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  /* Trading Status */
  .trading-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    background: var(--bg-tertiary);
    border-radius: 20px;
    font-size: 12px;
    color: var(--text-muted);
  }

  .trading-status.active {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .trading-status.paused {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
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
    animation: pulse 2s infinite;
  }

  /* Countdown Display */
  .countdown-display {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    background: var(--bg-tertiary);
    border-radius: 8px;
    font-size: 12px;
  }

  .countdown-label {
    color: var(--text-muted);
  }

  .countdown-time {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: #ef4444;
  }

  /* Buttons */
  .btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn:hover {
    background: var(--bg-surface);
  }

  .btn.active {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: var(--bg-primary);
  }

  .btn.close {
    padding: 8px;
  }

  /* Tabs */
  .news-tabs {
    display: flex;
    gap: 4px;
    padding: 12px 24px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .tab {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
  }

  .tab:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .tab.active {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  /* Content */
  .news-content {
    flex: 1;
    overflow-y: auto;
  }

  /* Calendar Section */
  .calendar-section {
    padding: 24px;
  }

  .view-toggle {
    display: flex;
    gap: 4px;
    margin-bottom: 16px;
    padding: 4px;
    background: var(--bg-tertiary);
    border-radius: 8px;
    width: fit-content;
  }

  .view-btn {
    padding: 8px 16px;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .view-btn.active {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  /* Events List */
  .events-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .event-card {
    display: flex;
    gap: 16px;
    padding: 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .event-card:hover {
    background: var(--bg-surface);
    border-color: var(--border-default);
  }

  .event-card.in-kill-zone {
    border-color: rgba(239, 68, 68, 0.5);
    background: rgba(239, 68, 68, 0.1);
  }

  .event-left {
    display: flex;
    flex-direction: column;
    gap: 8px;
    align-items: center;
    min-width: 70px;
  }

  .impact-badge {
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
  }

  .impact-badge.impact-high {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .impact-badge.impact-medium {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
  }

  .impact-badge.impact-low {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .event-time {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .event-main {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .event-header {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .currency-flag {
    font-size: 16px;
  }

  .currency-flag.large {
    font-size: 32px;
  }

  .event-name {
    margin: 0;
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .event-status {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 10px;
    margin-left: auto;
  }

  .event-status.status-passed {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .event-status.status-upcoming {
    background: rgba(107, 114, 128, 0.2);
    color: var(--text-muted);
  }

  .event-status.status-kill-zone {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .event-details {
    display: flex;
    gap: 12px;
  }

  .affected-pairs {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .event-values {
    display: flex;
    gap: 16px;
  }

  .value {
    display: flex;
    gap: 4px;
    font-size: 11px;
  }

  .value .label {
    color: var(--text-muted);
  }

  .value .data {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary);
  }

  .value.actual .data {
    color: #10b981;
  }

  .event-right {
    display: flex;
    align-items: center;
  }

  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: transparent;
    border: none;
    border-radius: 4px;
    color: var(--text-muted);
    cursor: pointer;
  }

  .icon-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  /* Calendar Grid View */
  .calendar-grid-view {
    padding: 40px 24px;
  }

  /* Empty State */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px 24px;
    color: var(--text-muted);
    text-align: center;
  }

  .empty-state small {
    margin-top: 8px;
    font-size: 11px;
  }

  /* Timeline Section */
  .timeline-section {
    padding: 24px;
  }

  .timeline-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
  }

  .timeline-header h3 {
    margin: 0;
    font-size: 14px;
    color: var(--text-primary);
  }

  .kill-zone-count {
    padding: 4px 10px;
    background: var(--bg-tertiary);
    border-radius: 10px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .timeline-container {
    background: var(--bg-secondary);
    border-radius: 10px;
    padding: 20px;
  }

  .timeline-track {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .timeline-zone {
    display: flex;
    gap: 16px;
  }

  .zone-marker {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--bg-tertiary);
    margin-top: 4px;
    position: relative;
  }

  .zone-marker::after {
    content: '';
    position: absolute;
    top: 12px;
    left: 50%;
    transform: translateX(-50%);
    width: 2px;
    height: calc(100% + 20px);
    background: var(--bg-tertiary);
  }

  .timeline-zone:last-child .zone-marker::after {
    display: none;
  }

  .timeline-zone.active .zone-marker {
    background: #ef4444;
    box-shadow: 0 0 0 4px rgba(239, 68, 68, 0.2);
  }

  .zone-content {
    flex: 1;
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .timeline-zone.active .zone-content {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
  }

  .zone-time {
    display: block;
    font-size: 11px;
    color: var(--text-muted);
    margin-bottom: 8px;
  }

  .zone-event {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
  }

  .zone-event .event-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .zone-indicator {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 10px;
    color: #ef4444;
  }

  /* Settings Section */
  .settings-section {
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  .settings-group {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 20px;
  }

  .settings-group h3 {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0 0 16px;
    font-size: 14px;
    color: var(--text-primary);
  }

  .setting-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid var(--border-subtle);
  }

  .setting-row:last-child {
    border-bottom: none;
  }

  .setting-row.full-width {
    flex-direction: column;
    align-items: stretch;
    gap: 12px;
    padding: 16px 0;
  }

  .setting-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: var(--text-primary);
    cursor: pointer;
  }

  .setting-options {
    display: flex;
    gap: 4px;
  }

  .option-btn {
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .option-btn.active {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: var(--bg-primary);
  }

  .setting-row select,
  .setting-row input[type="number"] {
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
  }

  .unit {
    margin-left: 8px;
    font-size: 12px;
    color: var(--text-muted);
  }

  .currency-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
    gap: 8px;
  }

  .currency-checkbox {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
  }

  .trading-toggle-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 12px 20px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .trading-toggle-btn.active {
    background: #ef4444;
    border-color: #ef4444;
    color: white;
  }

  .trading-toggle-btn.paused {
    background: #10b981;
    border-color: #10b981;
    color: white;
  }

  /* Detail Panel */
  .detail-panel-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
  }

  .detail-panel {
    background: var(--bg-secondary);
    border-radius: 12px;
    width: 600px;
    max-width: 90%;
    max-height: 80vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .detail-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .detail-header .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .detail-header h3 {
    margin: 0;
    font-size: 16px;
    color: var(--text-primary);
  }

  .event-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 4px;
    font-size: 12px;
  }

  .event-meta .currency {
    color: var(--text-muted);
  }

  .detail-content {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  .detail-section h4 {
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 0 0 12px;
    font-size: 13px;
    color: var(--text-primary);
  }

  .time-display {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .time-value {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .kill-zone-badge {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    background: rgba(239, 68, 68, 0.2);
    border-radius: 4px;
    font-size: 11px;
    color: #ef4444;
  }

  .values-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
    gap: 12px;
  }

  .value-card {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .value-label {
    font-size: 11px;
    color: var(--text-muted);
  }

  .value-data {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .value-data.actual {
    color: #10b981;
  }

  .pairs-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .pair-badge {
    padding: 6px 10px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    font-size: 12px;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
  }

  .event-description {
    margin: 0;
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.6;
  }

  .historical-impact-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px 20px;
    background: var(--bg-tertiary);
    border-radius: 8px;
    color: var(--text-muted);
    text-align: center;
  }

  .historical-impact-placeholder p {
    margin: 12px 0 4px;
    font-size: 13px;
  }
</style>

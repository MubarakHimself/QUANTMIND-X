<script lang="ts">
  import { Clock, Globe, Eye, CheckCircle, AlertTriangle, Calendar, Newspaper, Shield, BarChart3, Info, TrendingUp, Zap } from "lucide-svelte";

  export let events: Array<any> = [];
  export let killZones: Array<any> = [];
  export let calendarView: "list" | "weekly" | "monthly" = "list";
  export let killZoneSettings: any;
  export let activeTab: "calendar" | "timeline" | "settings" = "calendar";

  export let onSelectEvent: (event: any) => void;
  export let onToggleView: (view: "list" | "weekly" | "monthly") => void;

  // Helper functions exposed for the template
  export let getImpactBadgeClass: (impact: string) => string;
  export let getCurrencyFlag: (currency: string) => string;
  export let formatDateTime: (dateStr: string) => string;
  export let isEventInKillZone: (event: any) => boolean;
  export let getEventStatus: (event: any) => string;
</script>

<div class="news-content">
  {#if activeTab === "calendar"}
    <!-- Calendar View -->
    <div class="calendar-section">
      <!-- View Toggle -->
      <div class="view-toggle">
        <button
          class="view-btn"
          class:active={calendarView === "list"}
          on:click={() => onToggleView("list")}
        >
          List
        </button>
        <button
          class="view-btn"
          class:active={calendarView === "weekly"}
          on:click={() => onToggleView("weekly")}
        >
          Weekly
        </button>
        <button
          class="view-btn"
          class:active={calendarView === "monthly"}
          on:click={() => onToggleView("monthly")}
        >
          Monthly
        </button>
      </div>

      {#if calendarView === "list"}
        <!-- List View -->
        <div class="events-list">
          {#each events as event}
            <div
              class="event-card"
              class:in-kill-zone={isEventInKillZone(event)}
              on:click={() => onSelectEvent(event)}
              on:keydown={(e) => e.key === "Enter" && onSelectEvent(event)}
              role="button"
              tabindex="0"
            >
              <div class="event-left">
                <div class="event-impact">
                  <span class={getImpactBadgeClass(event.impact)}
                    >{event.impact.toUpperCase()}</span
                  >
                </div>
                <div class="event-time">
                  <Clock size={12} />
                  <span>{formatDateTime(event.date_time)}</span>
                </div>
              </div>

              <div class="event-main">
                <div class="event-header">
                  <span class="currency-flag"
                    >{getCurrencyFlag(event.currency)}</span
                  >
                  <h4 class="event-name">{event.name}</h4>
                  <span class="event-status status-{getEventStatus(event)}">
                    {#if getEventStatus(event) === "passed"}
                      <CheckCircle size={12} />
                      Passed
                    {:else if getEventStatus(event) === "kill-zone"}
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
                    {event.affected_pairs.join(", ")}
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

          {#if events.length === 0}
            <div class="empty-state">
              <Newspaper size={32} />
              <p>No events match your filters</p>
            </div>
          {/if}
        </div>
      {:else if calendarView === "weekly"}
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
  {:else if activeTab === "timeline"}
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
              {@const event = events.find((e) => e.id === zone.event_id)}
              {#if event}
                <div class="timeline-zone" class:active={zone.is_active}>
                  <div class="zone-marker"></div>
                  <div class="zone-content">
                    <span class="zone-time">
                      {formatDateTime(zone.start_time)} - {formatDateTime(
                        zone.end_time,
                      )}
                    </span>
                    <div class="zone-event">
                      <span class="currency-flag"
                        >{getCurrencyFlag(event.currency)}</span
                      >
                      <span class="event-name">{event.name}</span>
                      <span class={getImpactBadgeClass(event.impact)}
                        >{event.impact.toUpperCase()}</span
                      >
                    </div>
                    {#if zone.is_active}
                      <div class="zone-indicator">
                        <TrendingUp size={10} />
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
  {:else if activeTab === "settings"}
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
          <label for="before-news" class="setting-label"
            >Duration Before News</label
          >
          <div id="before-news" class="setting-options">
            <button
              class="option-btn"
              class:active={killZoneSettings.duration === 15}
              on:click={() => (killZoneSettings.duration = 15)}
            >
              15 min
            </button>
            <button
              class="option-btn"
              class:active={killZoneSettings.duration === 30}
              on:click={() => (killZoneSettings.duration = 30)}
            >
              30 min
            </button>
            <button
              class="option-btn"
              class:active={killZoneSettings.duration === 60}
              on:click={() => (killZoneSettings.duration = 60)}
            >
              60 min
            </button>
          </div>
        </div>

        <div class="setting-row">
          <label class="setting-label">
            <input
              type="checkbox"
              bind:checked={killZoneSettings.autoPause}
            />
            <span>Auto-Pause Trading</span>
          </label>
          <small>Automatically pause when entering kill zone</small>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .news-content {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
  }

  /* Calendar Section */
  .calendar-section {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .view-toggle {
    display: flex;
    gap: 8px;
  }

  .view-btn {
    padding: 8px 16px;
    border: none;
    border-radius: 6px;
    font-size: 13px;
    cursor: pointer;
    background: #334155;
    color: #94a3b8;
    transition: all 0.2s;
  }

  .view-btn:hover {
    background: #475569;
  }

  .view-btn.active {
    background: #3b82f6;
    color: white;
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
    background: #1e293b;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
    border: 1px solid transparent;
  }

  .event-card:hover {
    background: #334155;
    border-color: #3b82f6;
  }

  .event-card.in-kill-zone {
    border-color: rgba(239, 68, 68, 0.5);
    background: rgba(239, 68, 68, 0.1);
  }

  .event-left {
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-width: 80px;
  }

  .event-impact span {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
  }

  .impact-high {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .impact-medium {
    background: rgba(251, 191, 36, 0.2);
    color: #fbbf24;
  }

  .impact-low {
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
  }

  .event-time {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: #94a3b8;
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
    font-size: 14px;
  }

  .event-name {
    margin: 0;
    font-size: 14px;
    font-weight: 500;
    color: #f8fafc;
  }

  .event-status {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
  }

  .status-passed {
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
  }

  .status-kill-zone {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .status-upcoming {
    background: rgba(59, 130, 246, 0.2);
    color: #3b82f6;
  }

  .event-details {
    font-size: 12px;
    color: #94a3b8;
  }

  .affected-pairs {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .event-values {
    display: flex;
    gap: 16px;
    font-size: 12px;
  }

  .value {
    display: flex;
    gap: 4px;
  }

  .value .label {
    color: #64748b;
  }

  .value .data {
    font-weight: 500;
    color: #f8fafc;
  }

  .event-right {
    display: flex;
    align-items: flex-start;
  }

  .icon-btn {
    padding: 6px;
    border: none;
    border-radius: 4px;
    background: transparent;
    color: #94a3b8;
    cursor: pointer;
    transition: all 0.2s;
  }

  .icon-btn:hover {
    background: #475569;
    color: #f8fafc;
  }

  /* Empty State */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 48px;
    color: #64748b;
    text-align: center;
  }

  .empty-state p {
    margin: 12px 0 4px;
    font-size: 14px;
  }

  .empty-state small {
    font-size: 12px;
    color: #94a3b8;
  }

  .calendar-grid-view {
    min-height: 300px;
  }

  /* Timeline Section */
  .timeline-section {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .timeline-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .timeline-header h3 {
    margin: 0;
    font-size: 16px;
    color: #f8fafc;
  }

  .kill-zone-count {
    font-size: 12px;
    color: #94a3b8;
    background: #334155;
    padding: 4px 8px;
    border-radius: 4px;
  }

  .timeline-container {
    background: #1e293b;
    border-radius: 8px;
    padding: 16px;
  }

  .timeline-track {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .timeline-zone {
    display: flex;
    gap: 12px;
    padding: 12px;
    background: #334155;
    border-radius: 6px;
    border-left: 3px solid #64748b;
  }

  .timeline-zone.active {
    border-left-color: #ef4444;
    background: rgba(239, 68, 68, 0.1);
  }

  .zone-marker {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #64748b;
    margin-top: 4px;
  }

  .timeline-zone.active .zone-marker {
    background: #ef4444;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .zone-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .zone-time {
    font-size: 11px;
    color: #94a3b8;
  }

  .zone-event {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .event-name {
    font-size: 13px;
    font-weight: 500;
    color: #f8fafc;
  }

  .zone-indicator {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: #ef4444;
    margin-top: 4px;
  }

  /* Settings Section */
  .settings-section {
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  .settings-group {
    background: #1e293b;
    border-radius: 8px;
    padding: 16px;
  }

  .settings-group h3 {
    margin: 0 0 16px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    color: #f8fafc;
  }

  .setting-row {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 16px;
  }

  .setting-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: #f8fafc;
    cursor: pointer;
  }

  .setting-label input[type="checkbox"] {
    width: 16px;
    height: 16px;
    accent-color: #3b82f6;
  }

  .setting-row small {
    font-size: 11px;
    color: #64748b;
    margin-left: 24px;
  }

  .setting-options {
    display: flex;
    gap: 8px;
    margin-left: 24px;
  }

  .option-btn {
    padding: 6px 12px;
    border: 1px solid #475569;
    border-radius: 4px;
    font-size: 12px;
    cursor: pointer;
    background: transparent;
    color: #94a3b8;
    transition: all 0.2s;
  }

  .option-btn:hover {
    background: #334155;
  }

  .option-btn.active {
    background: #3b82f6;
    border-color: #3b82f6;
    color: white;
  }
</style>

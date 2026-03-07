<script lang="ts">
  import { X, Clock, Globe, AlertTriangle, BarChart3, Info, TrendingUp } from "lucide-svelte";

  export let event: any = null;
  export let isOpen = false;

  export let onClose: () => void;
  export let getCurrencyFlag: (currency: string) => string;
  export let getImpactBadgeClass: (impact: string) => string;
  export let formatDateTime: (dateStr: string) => string;
  export let isEventInKillZone: (event: any) => boolean;
</script>

{#if isOpen && event}
  <div
    class="detail-panel-overlay"
    on:click={onClose}
    role="button"
    tabindex="0"
    on:keydown={(e) => e.key === "Escape" && onClose()}
  >
    <div class="detail-panel" on:click|stopPropagation role="presentation">
      <div class="detail-header">
        <div class="header-left">
          <span class="currency-flag large"
            >{getCurrencyFlag(event.currency)}</span
          >
          <div>
            <h3>{event.name}</h3>
            <p class="event-meta">
              <span class={getImpactBadgeClass(event.impact)}
                >{event.impact.toUpperCase()}</span
              >
              <span class="currency">{event.currency}</span>
            </p>
          </div>
        </div>
        <button class="icon-btn" on:click={onClose}>
          <X size={18} />
        </button>
      </div>

      <div class="detail-content">
        <!-- Event Time -->
        <div class="detail-section">
          <h4><Clock size={14} /> Schedule</h4>
          <div class="time-display">
            <span class="time-value"
              >{formatDateTime(event.date_time)}</span
            >
            {#if isEventInKillZone(event)}
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
            {#if event.actual}
              <div class="value-card">
                <span class="value-label">Actual</span>
                <span class="value-data actual">{event.actual}</span>
              </div>
            {/if}
            {#if event.forecast}
              <div class="value-card">
                <span class="value-label">Forecast</span>
                <span class="value-data forecast"
                  >{event.forecast}</span
                >
              </div>
            {/if}
            {#if event.previous}
              <div class="value-card">
                <span class="value-label">Previous</span>
                <span class="value-data previous"
                  >{event.previous}</span
                >
              </div>
            {/if}
          </div>
        </div>

        <!-- Affected Pairs -->
        <div class="detail-section">
          <h4><Globe size={14} /> Affected Trading Pairs</h4>
          <div class="pairs-grid">
            {#each event.affected_pairs as pair}
              <span class="pair-badge">{pair}</span>
            {/each}
          </div>
        </div>

        <!-- Description -->
        {#if event.description}
          <div class="detail-section">
            <h4><Info size={14} /> Description</h4>
            <p class="event-description">{event.description}</p>
          </div>
        {/if}

        <!-- Historical Impact Placeholder -->
        <div class="detail-section">
          <h4><TrendingUp size={14} /> Historical Impact</h4>
          <div class="historical-impact-placeholder">
            <BarChart3 size={32} />
            <p>Historical volatility chart coming soon</p>
            <small
              >This will show price movement patterns for previous occurrences</small
            >
          </div>
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  .detail-panel-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    justify-content: flex-end;
    z-index: 1000;
    animation: fadeIn 0.2s ease-out;
  }

  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  .detail-panel {
    width: 400px;
    max-width: 90vw;
    height: 100%;
    background: #1e293b;
    box-shadow: -4px 0 20px rgba(0, 0, 0, 0.3);
    display: flex;
    flex-direction: column;
    animation: slideIn 0.3s ease-out;
  }

  @keyframes slideIn {
    from { transform: translateX(100%); }
    to { transform: translateX(0); }
  }

  .detail-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 20px;
    border-bottom: 1px solid #334155;
  }

  .header-left {
    display: flex;
    gap: 12px;
  }

  .currency-flag.large {
    font-size: 28px;
  }

  .detail-header h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    color: #f8fafc;
  }

  .event-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 6px 0 0;
  }

  .currency {
    font-size: 12px;
    color: #94a3b8;
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
    background: #334155;
    color: #f8fafc;
  }

  .detail-content {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
  }

  .detail-section {
    margin-bottom: 24px;
  }

  .detail-section h4 {
    margin: 0 0 12px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 500;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .time-display {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .time-value {
    font-size: 16px;
    font-weight: 500;
    color: #f8fafc;
  }

  .kill-zone-badge {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
  }

  .values-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }

  .value-card {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 12px;
    background: #334155;
    border-radius: 6px;
  }

  .value-label {
    font-size: 11px;
    color: #94a3b8;
  }

  .value-data {
    font-size: 14px;
    font-weight: 600;
    color: #f8fafc;
  }

  .pairs-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .pair-badge {
    padding: 4px 8px;
    background: #334155;
    border-radius: 4px;
    font-size: 12px;
    color: #f8fafc;
  }

  .event-description {
    margin: 0;
    font-size: 13px;
    line-height: 1.6;
    color: #cbd5e1;
  }

  .historical-impact-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 32px;
    background: #334155;
    border-radius: 8px;
    text-align: center;
    color: #64748b;
  }

  .historical-impact-placeholder p {
    margin: 12px 0 4px;
    font-size: 13px;
    color: #94a3b8;
  }

  .historical-impact-placeholder small {
    font-size: 11px;
    color: #64748b;
  }

  /* Impact badges */
  .impact-high {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
  }

  .impact-medium {
    background: rgba(251, 191, 36, 0.2);
    color: #fbbf24;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
  }

  .impact-low {
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
  }
</style>

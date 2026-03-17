<script lang="ts">
  import { stopPropagation } from 'svelte/legacy';

  import { createEventDispatcher } from "svelte";
  import {
    AlertTriangle,
    AlertCircle,
    Info,
    X,
    Check,
    Bell,
    BellOff,
    Trash2,
    ChevronDown,
    ChevronUp,
    Filter,
  } from "lucide-svelte";
  import {
    alerts,
    criticalAlerts,
    activeAlerts,
  } from "../services/metricsWebSocket";
  import type { AlertData } from "../services/metricsWebSocket";

  interface Props {
    maxHeight?: number;
    showFilters?: boolean;
    showAcknowledge?: boolean;
  }

  let { maxHeight = 400, showFilters = true, showAcknowledge = true }: Props = $props();

  const dispatch = createEventDispatcher();

  let filterSeverity: "all" | "info" | "warning" | "critical" = $state("all");
  let sortBy: "time" | "severity" = $state("time");
  let expanded = $state(true);


  function getFilteredAlerts(): AlertData[] {
    let result = $alerts;

    // Filter by severity
    if (filterSeverity !== "all") {
      result = result.filter((a) => a.severity === filterSeverity);
    }

    // Sort
    if (sortBy === "severity") {
      const severityOrder = { critical: 0, warning: 1, info: 2 };
      result = [...result].sort(
        (a, b) => severityOrder[a.severity] - severityOrder[b.severity],
      );
    } else {
      result = [...result].sort(
        (a, b) =>
          new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
      );
    }

    return result;
  }

  function getSeverityIcon(severity: string) {
    switch (severity) {
      case "critical":
        return AlertTriangle;
      case "warning":
        return AlertCircle;
      default:
        return Info;
    }
  }

  function getSeverityClass(severity: string): string {
    return severity;
  }

  function formatTime(timestamp: Date): string {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString();
  }

  function acknowledgeAlert(alert: AlertData) {
    dispatch("acknowledge", { alert });
  }

  function clearAlert(alert: AlertData) {
    dispatch("clear", { alert });
  }

  function clearAll() {
    dispatch("clearAll");
  }

  function acknowledgeAll() {
    filteredAlerts.forEach((alert) => {
      if (!alert.acknowledged) {
        acknowledgeAlert(alert);
      }
    });
  }
  let filteredAlerts = $derived(getFilteredAlerts());
  let criticalCount = $derived($criticalAlerts.length);
  let activeCount = $derived($activeAlerts.length);
</script>

<div class="alerts-panel" style="max-height: {maxHeight}px;">
  <!-- Header -->
  <button
    class="panel-header"
    onclick={() => (expanded = !expanded)}
    aria-expanded={expanded}
  >
    <div class="header-left">
      <Bell size={16} />
      <span class="title">Alerts</span>
      <div class="alert-counters">
        {#if criticalCount > 0}
          <span class="counter critical">{criticalCount}</span>
        {/if}
        {#if activeCount > 0}
          <span class="counter active">{activeCount}</span>
        {/if}
      </div>
    </div>
    <div class="header-actions">
      {#if expanded && showAcknowledge && activeCount > 0}
        <button
          class="action-btn"
          onclick={stopPropagation(acknowledgeAll)}
          title="Acknowledge All"
        >
          <Check size={14} />
        </button>
      {/if}
      {#if expanded}
        <button
          class="action-btn"
          onclick={stopPropagation(clearAll)}
          title="Clear All"
        >
          <Trash2 size={14} />
        </button>
      {/if}
      {#if expanded}
        <ChevronUp size={16} />
      {:else}
        <ChevronDown size={16} />
      {/if}
    </div>
  </button>

  {#if expanded}
    <!-- Filters -->
    {#if showFilters}
      <div class="panel-filters">
        <div class="filter-group">
          <Filter size={14} />
          <select bind:value={filterSeverity}>
            <option value="all">All</option>
            <option value="critical">Critical</option>
            <option value="warning">Warning</option>
            <option value="info">Info</option>
          </select>
        </div>
        <div class="filter-group">
          <span>Sort:</span>
          <select bind:value={sortBy}>
            <option value="time">Time</option>
            <option value="severity">Severity</option>
          </select>
        </div>
      </div>
    {/if}

    <!-- Alert List -->
    <div class="alert-list">
      {#if filteredAlerts.length === 0}
        <div class="no-alerts">
          <BellOff size={24} />
          <span>No active alerts</span>
        </div>
      {:else}
        {#each filteredAlerts as alert (alert.id)}
          {@const SvelteComponent = getSeverityIcon(alert.severity)}
          <div
            class="alert-item {getSeverityClass(alert.severity)}"
            class:acknowledged={alert.acknowledged}
          >
            <div class="alert-icon">
              <SvelteComponent
                size={16}
              />
            </div>
            <div class="alert-content">
              <div class="alert-message">{alert.message}</div>
              <div class="alert-meta">
                <span class="alert-source">{alert.source}</span>
                <span class="alert-time">{formatTime(alert.timestamp)}</span>
              </div>
            </div>
            <div class="alert-actions">
              {#if !alert.acknowledged && showAcknowledge}
                <button
                  class="action-btn"
                  onclick={() => acknowledgeAlert(alert)}
                  title="Acknowledge"
                >
                  <Check size={14} />
                </button>
              {/if}
              <button
                class="action-btn"
                onclick={() => clearAlert(alert)}
                title="Dismiss"
              >
                <X size={14} />
              </button>
            </div>
          </div>
        {/each}
      {/if}
    </div>
  {/if}
</div>

<style>
  .alerts-panel {
    display: flex;
    flex-direction: column;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    overflow: hidden;
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: var(--bg-tertiary);
    cursor: pointer;
    user-select: none;
    width: 100%;
    border: none;
    font-family: inherit;
    text-align: left;
  }

  .panel-header:hover {
    background: var(--bg-secondary);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .title {
    font-weight: 600;
    color: var(--text-primary);
  }

  .alert-counters {
    display: flex;
    gap: 4px;
    margin-left: 8px;
  }

  .counter {
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 600;
  }

  .counter.critical {
    background: rgba(244, 67, 54, 0.2);
    color: #f44336;
  }

  .counter.active {
    background: rgba(255, 152, 0, 0.2);
    color: #ff9800;
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .action-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: transparent;
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .action-btn:hover {
    background: var(--bg-primary);
    color: var(--text-primary);
  }

  .panel-filters {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 16px;
    background: var(--bg-primary);
    border-bottom: 1px solid var(--border-subtle);
    font-size: 12px;
  }

  .filter-group {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--text-muted);
  }

  .filter-group select {
    padding: 4px 8px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 12px;
    cursor: pointer;
  }

  .alert-list {
    flex: 1;
    overflow-y: auto;
    max-height: calc(var(--max-height, 400px) - 100px);
  }

  .no-alerts {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 40px;
    color: var(--text-muted);
  }

  .alert-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-subtle);
    transition: background 0.15s ease;
  }

  .alert-item:last-child {
    border-bottom: none;
  }

  .alert-item:hover {
    background: var(--bg-tertiary);
  }

  .alert-item.critical {
    border-left: 3px solid #f44336;
  }

  .alert-item.warning {
    border-left: 3px solid #ff9800;
  }

  .alert-item.info {
    border-left: 3px solid #2196f3;
  }

  .alert-item.acknowledged {
    opacity: 0.5;
  }

  .alert-icon {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    margin-top: 2px;
  }

  .alert-item.critical .alert-icon {
    background: rgba(244, 67, 54, 0.2);
    color: #f44336;
  }

  .alert-item.warning .alert-icon {
    background: rgba(255, 152, 0, 0.2);
    color: #ff9800;
  }

  .alert-item.info .alert-icon {
    background: rgba(33, 150, 243, 0.2);
    color: #2196f3;
  }

  .alert-content {
    flex: 1;
    min-width: 0;
  }

  .alert-message {
    color: var(--text-primary);
    font-size: 13px;
    line-height: 1.4;
    word-wrap: break-word;
  }

  .alert-meta {
    display: flex;
    gap: 12px;
    margin-top: 4px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .alert-source {
    font-weight: 500;
  }

  .alert-actions {
    display: flex;
    gap: 4px;
    opacity: 0;
    transition: opacity 0.15s ease;
  }

  .alert-item:hover .alert-actions {
    opacity: 1;
  }
</style>

<script lang="ts">
/**
 * MorningDigestCard - Morning Digest Display Component
 *
 * Shows overnight agent activity summary, pending approvals, node health,
 * critical alerts, and market session indicator on first load of Live Trading canvas.
 *
 * Uses Frosted Terminal aesthetic (Tier 2 glass) with Lucide icons.
 */

import { onMount } from 'svelte';
import GlassTile from './GlassTile.svelte';
import NodeHealthBadge from './NodeHealthBadge.svelte';
import DegradedIndicator from './DegradedIndicator.svelte';
import { nodeHealthState, checkNodeHealth, isContaboDegraded } from '$lib/stores/node-health';
import { Sun, Bell, Server, AlertTriangle, Clock, Wifi, WifiOff } from 'lucide-svelte';
import { buildApiUrl } from '$lib/api';

// Morning digest data from API
interface MorningDigestData {
  agent_activity: Array<{
    agent_id: string;
    agent_type: string;
    action: string;
    timestamp_utc: string;
  }>;
  pending_approvals: number;
  node_health: {
    contabo: { status: string; latency_ms: number };
    cloudzy: { status: string; latency_ms: number };
  };
  critical_alerts: Array<{
    id: string;
    severity: 'high' | 'medium' | 'low';
    message: string;
    timestamp_utc: string;
  }>;
  market_session: {
    tokyo: 'open' | 'closed';
    london: 'open' | 'closed';
    new_york: 'open' | 'closed';
  };
}

let digestData = $state<MorningDigestData | null>(null);
let isLoading = $state(true);
let error = $state<string | null>(null);

// Market session colors
const sessionColors = {
  open: '#00c896', // green
  closed: '#666'   // gray
};

// Fetch morning digest data
async function fetchMorningDigest() {
  try {
    isLoading = true;
    error = null;

    const response = await fetch(buildApiUrl('/api/v1/server/morning-digest'), {
      credentials: 'include'
    });
    if (!response.ok) {
      throw new Error(`Failed to fetch digest: ${response.status}`);
    }

    digestData = await response.json();
  } catch (e) {
    error = e instanceof Error ? e.message : 'Failed to load morning digest';
    console.error('[MorningDigest] Error:', error);
  } finally {
    isLoading = false;
  }
}

// Determine if this is the first load in a session
let hasLoaded = $state(false);

onMount(() => {
  if (!hasLoaded) {
    fetchMorningDigest();
    checkNodeHealth();
    hasLoaded = true;
  }
});

// Format timestamp to readable time
function formatTime(timestamp: string): string {
  if (!timestamp) return '--:--';
  try {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      timeZone: 'UTC'
    });
  } catch {
    return '--:--';
  }
}

// Get market session display
function getMarketSessionInfo() {
  if (!digestData?.market_session) return null;
  const { tokyo, london, new_york } = digestData.market_session;
  return [
    { name: 'Tokyo', status: tokyo },
    { name: 'London', status: london },
    { name: 'New York', status: new_york }
  ];
}
</script>

<div class="morning-digest-wrapper">
  {#if $isContaboDegraded}
    <DegradedIndicator />
  {/if}

  <GlassTile>
    <div class="digest-header">
      <div class="header-left">
        <Sun size={18} strokeWidth={1.5} />
        <h2>Morning Digest</h2>
      </div>
      <div class="header-right">
        <NodeHealthBadge />
      </div>
    </div>

    {#if isLoading}
      <div class="digest-loading">
        <div class="skeleton-line title-skeleton"></div>
        <div class="skeleton-grid">
          <div class="skeleton-item"></div>
          <div class="skeleton-item"></div>
          <div class="skeleton-item"></div>
        </div>
      </div>
    {:else if error}
      <div class="digest-error">
        <AlertTriangle size={16} />
        <span>{error}</span>
      </div>
    {:else if digestData}
      <div class="digest-content">
        <!-- Agent Activity Summary -->
        <div class="digest-section">
          <div class="section-header">
            <Server size={14} strokeWidth={1.5} />
            <h3>Agent Activity (Overnight)</h3>
            <span class="count">{digestData.agent_activity.length}</span>
          </div>
          <div class="activity-list">
            {#if digestData.agent_activity.length > 0}
              {#each digestData.agent_activity.slice(0, 3) as activity}
                <div class="activity-item">
                  <span class="agent-type">{activity.agent_type}</span>
                  <span class="action">{activity.action}</span>
                  <span class="timestamp">{formatTime(activity.timestamp_utc)}</span>
                </div>
              {/each}
              {#if digestData.agent_activity.length > 3}
                <div class="activity-more">
                  +{digestData.agent_activity.length - 3} more
                </div>
              {/if}
            {:else}
              <div class="empty-activity">No overnight activity</div>
            {/if}
          </div>
        </div>

        <!-- Pending Approvals -->
        <div class="digest-section approvals">
          <div class="section-header">
            <Bell size={14} strokeWidth={1.5} />
            <h3>Pending Approvals</h3>
          </div>
          {#if digestData.pending_approvals > 0}
            <div class="approval-chip">
              <span class="number">{digestData.pending_approvals}</span>
            </div>
          {:else}
            <span class="no-approvals">None</span>
          {/if}
        </div>

        <!-- Critical Alerts -->
        <div class="digest-section alerts">
          <div class="section-header">
            <AlertTriangle size={14} strokeWidth={1.5} />
            <h3>Critical Alerts</h3>
            <span class="count">{digestData.critical_alerts.length}</span>
          </div>
          {#if digestData.critical_alerts.length > 0}
            <div class="alert-list">
              {#each digestData.critical_alerts.slice(0, 2) as alert}
                <div class="alert-item" class:high={alert.severity === 'high'} class:medium={alert.severity === 'medium'}>
                  <span class="alert-message">{alert.message}</span>
                  <span class="alert-time">{formatTime(alert.timestamp_utc)}</span>
                </div>
              {/each}
            </div>
          {:else}
            <span class="no-alerts">No critical alerts</span>
          {/if}
        </div>

        <!-- Market Session -->
        <div class="digest-section market">
          <div class="section-header">
            <Clock size={14} strokeWidth={1.5} />
            <h3>Market Sessions</h3>
          </div>
          <div class="session-indicators">
            {#each getMarketSessionInfo() || [] as session}
              <div class="session-item">
                <span class="session-name">{session.name}</span>
                <span class="session-status" style="color: {sessionColors[session.status]}">
                  {session.status === 'open' ? '●' : '○'} {session.status}
                </span>
              </div>
            {/each}
          </div>
        </div>
      </div>
    {/if}
  </GlassTile>
</div>

<style>
  .morning-digest-wrapper {
    position: relative;
    margin-bottom: 16px;
  }

  .digest-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 12px;
    margin-bottom: 12px;
    border-bottom: 1px solid rgba(0, 212, 255, 0.1);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #e0e0e0;
  }

  .header-left h2 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 600;
    margin: 0;
    color: #e0e0e0;
  }

  .digest-loading {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .skeleton-line {
    background: linear-gradient(90deg, rgba(0, 212, 255, 0.1) 25%, rgba(0, 212, 255, 0.2) 50%, rgba(0, 212, 255, 0.1) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 4px;
  }

  .title-skeleton {
    height: 20px;
    width: 40%;
  }

  .skeleton-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
  }

  .skeleton-item {
    height: 40px;
    background: rgba(0, 212, 255, 0.05);
    border-radius: 4px;
    animation: pulse 1.5s ease-in-out infinite;
  }

  @keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  }

  @keyframes pulse {
    0%, 100% { opacity: 0.4; }
    50% { opacity: 0.7; }
  }

  .digest-error {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #f59e0b;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }

  .digest-content {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr 1fr;
    gap: 16px;
  }

  .digest-section {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .section-header {
    display: flex;
    align-items: center;
    gap: 6px;
    color: #888;
  }

  .section-header h3 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    margin: 0;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .count {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    padding: 2px 6px;
    background: rgba(0, 212, 255, 0.1);
    border-radius: 4px;
    color: #00d4ff;
  }

  .activity-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .activity-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #aaa;
  }

  .agent-type {
    color: #00d4ff;
    font-weight: 500;
  }

  .action {
    flex: 1;
    color: #888;
  }

  .timestamp {
    color: #555;
    font-size: 10px;
  }

  .activity-more {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #666;
    font-style: italic;
  }

  .empty-activity {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #555;
  }

  .approval-chip {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 12px;
    padding: 4px 12px;
  }

  .approval-chip .number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 600;
    color: #f59e0b;
  }

  .no-approvals {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #555;
  }

  .alert-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .alert-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    padding: 4px 8px;
    border-radius: 4px;
    background: rgba(0, 0, 0, 0.2);
  }

  .alert-item.high {
    border-left: 2px solid #ff3b3b;
  }

  .alert-item.medium {
    border-left: 2px solid #f59e0b;
  }

  .alert-message {
    color: #aaa;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .alert-time {
    color: #555;
    font-size: 10px;
  }

  .no-alerts {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #555;
  }

  .session-indicators {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .session-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
  }

  .session-name {
    color: #888;
  }

  .session-status {
    text-transform: capitalize;
    font-weight: 500;
  }
</style>

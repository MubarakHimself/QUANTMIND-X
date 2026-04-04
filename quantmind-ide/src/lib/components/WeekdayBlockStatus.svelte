<script lang="ts">
  import { onMount } from 'svelte';
  import { Shield, ShieldAlert, Lock, Unlock } from 'lucide-svelte';
  import { API_BASE } from '$lib/constants';

  const apiBase = API_BASE || '';

  let blockStatus = $state<{
    is_blocked: boolean;
    current_day: string;
    current_time: string;
    next_allowed_window: string | null;
    message: string;
  }>({
    is_blocked: true,
    current_day: '',
    current_time: '',
    next_allowed_window: null,
    message: '',
  });

  let loading = $state(true);
  let error = $state<string | null>(null);

  async function fetchBlockStatus() {
    try {
      const baseUrl = apiBase || window.location.origin;
      const response = await fetch(`${baseUrl}/api/weekend-cycle/block/status`);

      if (!response.ok) {
        throw new Error(`Failed to fetch: ${response.status}`);
      }

      const data = await response.json();
      blockStatus = data;
      error = null;
    } catch (e) {
      console.error('Failed to fetch block status:', e);
      error = e instanceof Error ? e.message : 'Unknown error';
    } finally {
      loading = false;
    }
  }

  function getStatusColor(): string {
    return blockStatus.is_blocked ? '#ef4444' : '#10b981';
  }

  function getStatusBgColor(): string {
    return blockStatus.is_blocked
      ? 'rgba(239, 68, 68, 0.15)'
      : 'rgba(16, 185, 129, 0.15)';
  }

  onMount(() => {
    fetchBlockStatus();
    const interval = setInterval(fetchBlockStatus, 60000); // Refresh every minute
    return () => clearInterval(interval);
  });
</script>

<div class="weekday-block-status">
  <div class="status-header">
    {#if blockStatus.is_blocked}
      <ShieldAlert size={16} color={getStatusColor()} />
      <span class="status-title">Parameter Updates BLOCKED</span>
    {:else}
      <Shield size={16} color={getStatusColor()} />
      <span class="status-title">Parameter Updates ALLOWED</span>
    {/if}
  </div>

  {#if loading}
    <div class="loading">Checking...</div>
  {:else if error}
    <div class="error">{error}</div>
  {:else}
    <div class="status-details">
      <div class="detail-row">
        <span class="detail-label">Current Time:</span>
        <span class="detail-value">{blockStatus.current_day} {blockStatus.current_time}</span>
      </div>

      <div class="status-indicator" style="background: {getStatusBgColor()}">
        {#if blockStatus.is_blocked}
          <Lock size={14} color={getStatusColor()} />
          <span style="color: {getStatusColor()}">Weekday - No Changes Allowed</span>
        {:else}
          <Unlock size={14} color={getStatusColor()} />
          <span style="color: {getStatusColor()}">Weekend - Changes Permitted</span>
        {/if}
      </div>

      {#if blockStatus.is_blocked && blockStatus.next_allowed_window}
        <div class="next-window">
          <span class="detail-label">Next Allowed:</span>
          <span class="detail-value next-time">
            Friday 21:00 GMT
          </span>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .weekday-block-status {
    background: rgba(8, 8, 12, 0.75);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 12px;
    color: #e4e4e7;
    font-size: 12px;
  }

  .status-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
  }

  .status-title {
    font-size: 13px;
    font-weight: 600;
  }

  .loading {
    color: #6b7280;
    text-align: center;
    padding: 8px;
  }

  .error {
    color: #ef4444;
    text-align: center;
    padding: 8px;
  }

  .status-details {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .detail-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .detail-label {
    color: #6b7280;
  }

  .detail-value {
    color: #e4e4e7;
  }

  .status-indicator {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 500;
  }

  .next-window {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: 6px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
  }

  .next-time {
    font-weight: 500;
  }
</style>

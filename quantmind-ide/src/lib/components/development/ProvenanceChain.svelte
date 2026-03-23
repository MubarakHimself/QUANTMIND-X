<script lang="ts">
  /**
   * Provenance Chain Component
   *
   * Timeline visualization of EA origin.
   * Shows source -> Research -> Dev -> review -> approval.
   */
  import { onMount, onDestroy } from 'svelte';
  import { provenanceStore, type ProvenanceChain, type ProvenanceNode } from '$lib/stores/provenance';
  import { GitBranch, CheckCircle, Clock, XCircle, AlertCircle, ExternalLink } from 'lucide-svelte';

  // Props
  export let strategyId: string = '';
  export let versionTag: string = '';

  let chain: ProvenanceChain | null = null;
  let loading = false;
  let error: string | null = null;

  // Subscribe to store
  const unsubscribe = provenanceStore.subscribe(state => {
    chain = state.chain;
    loading = state.loading;
    error = state.error;
  });

  onMount(async () => {
    if (strategyId) {
      await provenanceStore.loadProvenance(strategyId, versionTag || undefined);
    }
  });

  onDestroy(() => {
    unsubscribe();
  });

  function getStageIcon(stage: string, status: string) {
    if (status === 'completed') return CheckCircle;
    if (status === 'in_progress') return Clock;
    if (status === 'failed') return XCircle;
    return AlertCircle;
  }

  function getStageColor(status: string): string {
    switch (status) {
      case 'completed': return '#22c55e';
      case 'in_progress': return '#f59e0b';
      case 'failed': return '#ff3b3b';
      default: return '#6b7280';
    }
  }

  function formatTimestamp(timestamp: string): string {
    try {
      return new Date(timestamp).toLocaleString();
    } catch {
      return timestamp;
    }
  }

  function getStageLabel(stage: string): string {
    switch (stage) {
      case 'source': return 'Source';
      case 'research': return 'Research';
      case 'dev': return 'Development';
      case 'review': return 'Code Review';
      case 'approval': return 'Approval';
      default: return stage;
    }
  }
</script>

<div class="provenance-chain">
  <header class="chain-header">
    <h2>
      <GitBranch size={16} />
      EA Provenance Chain
    </h2>
    {#if chain}
      <span class="version-tag">v{chain.version_tag}</span>
    {/if}
  </header>

  {#if loading && !chain}
    <div class="loading-state">
      <Clock size={24} />
      <span>Loading provenance...</span>
    </div>
  {:else if error}
    <div class="error-state">
      <span>Error: {error}</span>
    </div>
  {:else if chain}
    <div class="timeline">
      {#each chain.chain as node, index}
        <div class="timeline-node">
          <!-- Connector line -->
          {#if index > 0}
            <div class="connector-line"></div>
          {/if}

          <!-- Node -->
          <div class="node-content">
            <div class="node-icon" style="color: {getStageColor(node.status)}">
              <svelte:component this={getStageIcon(node.stage, node.status)} size={20} />
            </div>

            <div class="node-details">
              <div class="node-header">
                <span class="stage-name">{getStageLabel(node.stage)}</span>
                <span class="stage-status" style="color: {getStageColor(node.status)}">
                  {node.status.replace('_', ' ')}
                </span>
              </div>

              <div class="node-meta">
                <span class="actor">{node.actor}</span>
                <span class="timestamp">{formatTimestamp(node.timestamp)}</span>
              </div>

              {#if node.details && Object.keys(node.details).length > 0}
                <div class="node-details-list">
                  {#each Object.entries(node.details) as [key, value]}
                    <div class="detail-item">
                      <span class="detail-key">{key}:</span>
                      <span class="detail-value">
                        {#if key === 'url' && typeof value === 'string'}
                          <a href={value} target="_blank" rel="noopener" class="detail-link">
                            {value.substring(0, 40)}...
                            <ExternalLink size={10} />
                          </a>
                        {:else if typeof value === 'boolean'}
                          {value ? 'Yes' : 'No'}
                        {:else if typeof value === 'number'}
                          {value.toFixed(2)}
                        {:else}
                          {value}
                        {/if}
                      </span>
                    </div>
                  {/each}
                </div>
              {/if}
            </div>
          </div>
        </div>
      {/each}
    </div>

    <!-- Source URL -->
    {#if chain.source_url}
      <div class="source-link">
        <ExternalLink size={14} />
        <a href={chain.source_url} target="_blank" rel="noopener">
          {chain.source_url}
        </a>
      </div>
    {/if}

    <div class="chain-footer">
      <span>{chain.total_stages} stages completed</span>
    </div>
  {:else}
    <div class="empty-state">
      <span>No provenance data available</span>
    </div>
  {/if}
</div>

<style>
  .provenance-chain {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 16px;
    background: rgba(10, 15, 26, 0.95);
    backdrop-filter: blur(12px);
    border-radius: 8px;
    border: 1px solid rgba(0, 212, 255, 0.1);
  }

  .chain-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(0, 212, 255, 0.1);
  }

  .chain-header h2 {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 600;
    color: #a855f7;
    margin: 0;
  }

  .version-tag {
    padding: 4px 8px;
    background: rgba(168, 85, 247, 0.15);
    border: 1px solid rgba(168, 85, 247, 0.3);
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #a855f7;
  }

  .loading-state, .error-state, .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
    color: rgba(255, 255, 255, 0.5);
    gap: 12px;
  }

  .error-state {
    color: #ff3b3b;
  }

  .timeline {
    display: flex;
    flex-direction: column;
    gap: 0;
  }

  .timeline-node {
    position: relative;
    display: flex;
    flex-direction: column;
  }

  .connector-line {
    position: absolute;
    left: 19px;
    top: -8px;
    width: 2px;
    height: 16px;
    background: rgba(0, 212, 255, 0.2);
  }

  .node-content {
    display: flex;
    gap: 12px;
    padding: 12px;
    background: rgba(8, 13, 20, 0.6);
    border-radius: 8px;
    border: 1px solid rgba(0, 212, 255, 0.1);
  }

  .node-icon {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
  }

  .node-details {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-width: 0;
  }

  .node-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .stage-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    color: #00d4ff;
  }

  .stage-status {
    font-size: 11px;
    text-transform: capitalize;
  }

  .node-meta {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: rgba(255, 255, 255, 0.4);
  }

  .node-details-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 4px;
    padding-top: 8px;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
  }

  .detail-item {
    display: flex;
    gap: 8px;
    font-size: 11px;
  }

  .detail-key {
    color: rgba(255, 255, 255, 0.4);
  }

  .detail-value {
    color: rgba(255, 255, 255, 0.7);
    font-family: 'JetBrains Mono', monospace;
  }

  .detail-link {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: #00d4ff;
    text-decoration: none;
    word-break: break-all;
  }

  .detail-link:hover {
    text-decoration: underline;
  }

  .source-link {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 6px;
    font-size: 12px;
    color: rgba(255, 255, 255, 0.6);
  }

  .source-link a {
    color: #00d4ff;
    text-decoration: none;
    word-break: break-all;
  }

  .source-link a:hover {
    text-decoration: underline;
  }

  .chain-footer {
    padding-top: 12px;
    border-top: 1px solid rgba(0, 212, 255, 0.1);
    font-size: 11px;
    color: rgba(255, 255, 255, 0.3);
    text-align: right;
  }
</style>
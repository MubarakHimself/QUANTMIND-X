<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { fade, scale } from 'svelte/transition';
  import { X, FileText, TrendingUp, Link, Activity, Check, AlertCircle, Clock } from 'lucide-svelte';
  
  // Props
  export let item: {
    id: string;
    name: string;
    type: 'file' | 'strategy' | 'broker' | 'backtest';
    icon: any;
    status?: string;
    size?: number;
  };
  
  const dispatch = createEventDispatcher();
  
  // State
  let isHovered = false;
  let showTooltip = false;
  
  // Icon mapping
  const iconMap: Record<string, any> = {
    file: FileText,
    strategy: TrendingUp,
    broker: Link,
    backtest: Activity
  };
  
  // Get the icon component
  $: IconComponent = iconMap[item.type] || FileText;
  
  // Get status icon
  $: statusIcon = getStatusIcon(item.status);
  
  // Get status color
  $: statusColor = getStatusColor(item.status);
  
  function getStatusIcon(status?: string): any {
    switch (status) {
      case 'connected':
      case 'completed':
        return Check;
      case 'error':
      case 'failed':
        return AlertCircle;
      case 'pending':
      case 'running':
        return Clock;
      default:
        return null;
    }
  }
  
  function getStatusColor(status?: string): string {
    switch (status) {
      case 'connected':
      case 'completed':
        return 'var(--accent-success)';
      case 'error':
      case 'failed':
        return 'var(--accent-danger)';
      case 'pending':
      case 'running':
        return 'var(--accent-warning)';
      default:
        return 'var(--text-muted)';
    }
  }
  
  // Get type label
  function getTypeLabel(type: string): string {
    const labels: Record<string, string> = {
      file: 'File',
      strategy: 'Strategy',
      broker: 'Broker',
      backtest: 'Backtest'
    };
    return labels[type] || type;
  }
  
  // Format file size
  function formatSize(bytes?: number): string {
    if (!bytes) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  
  // Handle remove
  function handleRemove(e: MouseEvent) {
    e.stopPropagation();
    dispatch('remove');
  }
  
  // Truncate name
  function truncateName(name: string, maxLength: number = 20): string {
    if (name.length <= maxLength) return name;
    const ext = name.split('.').pop();
    const baseName = name.slice(0, maxLength - (ext?.length || 0) - 4);
    return `${baseName}...${ext ? '.' + ext : ''}`;
  }
</script>

<!-- svelte-ignore a11y-no-static-element-interactions -->
<div 
  class="context-tag {item.type}"
  on:mouseenter={() => { isHovered = true; showTooltip = true; }}
  on:mouseleave={() => { isHovered = false; showTooltip = false; }}
  transition:scale={{ duration: 150 }}
  role="listitem"
  aria-label="{getTypeLabel(item.type)}: {item.name}"
>
  <!-- Icon -->
  <div class="tag-icon">
    <svelte:component this={IconComponent} size={12} />
  </div>
  
  <!-- Name -->
  <span class="tag-name" title={item.name}>
    {truncateName(item.name)}
  </span>
  
  <!-- Status indicator -->
  {#if item.status && statusIcon}
    <svelte:component 
      this={statusIcon} 
      size={10} 
      class="status-icon"
      style="color: {statusColor}"
    />
  {/if}
  
  <!-- Remove button -->
  {#if isHovered}
    <button 
      class="remove-btn" 
      on:click={handleRemove}
      transition:fade={{ duration: 100 }}
      title="Remove"
      aria-label="Remove {item.name}"
    >
      <X size={10} />
    </button>
  {/if}
  
  <!-- Tooltip -->
  {#if showTooltip}
    <div class="tooltip" transition:fade={{ duration: 100 }}>
      <div class="tooltip-header">
        <svelte:component this={IconComponent} size={14} />
        <span class="tooltip-name">{item.name}</span>
      </div>
      <div class="tooltip-details">
        <div class="tooltip-row">
          <span class="tooltip-label">Type:</span>
          <span class="tooltip-value">{getTypeLabel(item.type)}</span>
        </div>
        {#if item.size}
          <div class="tooltip-row">
            <span class="tooltip-label">Size:</span>
            <span class="tooltip-value">{formatSize(item.size)}</span>
          </div>
        {/if}
        {#if item.status}
          <div class="tooltip-row">
            <span class="tooltip-label">Status:</span>
            <span class="tooltip-value" style="color: {statusColor}">
              {item.status}
            </span>
          </div>
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
  .context-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    font-size: 11px;
    color: var(--text-secondary);
    position: relative;
    cursor: default;
    transition: all 0.15s;
  }
  
  .context-tag:hover {
    background: var(--bg-secondary);
    border-color: var(--border-subtle);
  }
  
  /* Type-specific colors */
  .context-tag.file {
    border-left: 2px solid var(--accent-primary);
  }
  
  .context-tag.strategy {
    border-left: 2px solid var(--accent-secondary);
  }
  
  .context-tag.broker {
    border-left: 2px solid var(--accent-success);
  }
  
  .context-tag.backtest {
    border-left: 2px solid var(--accent-warning);
  }
  
  .tag-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
  }
  
  .tag-name {
    max-width: 100px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  
  .status-icon {
    flex-shrink: 0;
  }
  
  .remove-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    background: transparent;
    border: none;
    border-radius: 4px;
    color: var(--text-muted);
    cursor: pointer;
    padding: 0;
    margin-left: 2px;
    transition: all 0.15s;
  }
  
  .remove-btn:hover {
    background: rgba(239, 68, 68, 0.2);
    color: var(--accent-danger);
  }
  
  /* Tooltip */
  .tooltip {
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 10px 12px;
    min-width: 180px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    z-index: 100;
    pointer-events: none;
  }
  
  .tooltip::after {
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 6px solid transparent;
    border-top-color: var(--border-subtle);
  }
  
  .tooltip-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-subtle);
  }
  
  .tooltip-name {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
    word-break: break-all;
  }
  
  .tooltip-details {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  
  .tooltip-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
  }
  
  .tooltip-label {
    font-size: 10px;
    color: var(--text-muted);
  }
  
  .tooltip-value {
    font-size: 11px;
    color: var(--text-primary);
  }
</style>

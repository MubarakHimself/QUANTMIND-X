<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { fade, scale } from 'svelte/transition';
  import { X, FileText, TrendingUp, Link, Activity, Check, AlertCircle, Clock } from 'lucide-svelte';
  
  
  interface Props {
    // Props
    item: {
    id: string;
    name: string;
    type: 'file' | 'strategy' | 'broker' | 'backtest';
    icon: any;
    status?: string;
    size?: number;
  };
  }

  let { item }: Props = $props();
  
  const dispatch = createEventDispatcher();
  
  // State
  let isHovered = $state(false);
  let showTooltip = $state(false);
  
  // Icon mapping
  const iconMap: Record<string, any> = {
    file: FileText,
    strategy: TrendingUp,
    broker: Link,
    backtest: Activity
  };
  
  
  
  
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
        return 'var(--color-accent-green)';
      case 'error':
      case 'failed':
        return 'var(--color-accent-red)';
      case 'pending':
      case 'running':
        return 'var(--color-accent-amber)';
      default:
        return 'var(--color-text-muted)';
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
  // Get the icon component
  let IconComponent = $derived(iconMap[item.type] || FileText);
  // Get status icon
  let statusIcon = $derived(getStatusIcon(item.status));
  // Get status color
  let statusColor = $derived(getStatusColor(item.status));
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div 
  class="context-tag {item.type}"
  onmouseenter={() => { isHovered = true; showTooltip = true; }}
  onmouseleave={() => { isHovered = false; showTooltip = false; }}
  transition:scale={{ duration: 150 }}
  role="listitem"
  aria-label="{getTypeLabel(item.type)}: {item.name}"
>
  <!-- Icon -->
  <div class="tag-icon">
    <IconComponent size={12} />
  </div>
  
  <!-- Name -->
  <span class="tag-name" title={item.name}>
    {truncateName(item.name)}
  </span>
  
  <!-- Status indicator -->
  {#if item.status && statusIcon}
    {@const SvelteComponent = statusIcon}
    <SvelteComponent 
      size={10} 
      class="status-icon"
      style="color: {statusColor}"
    />
  {/if}
  
  <!-- Remove button -->
  {#if isHovered}
    <button 
      class="remove-btn" 
      onclick={handleRemove}
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
        <IconComponent size={14} />
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
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    font-size: 11px;
    color: var(--color-text-secondary);
    position: relative;
    cursor: default;
    transition: all 0.15s;
  }
  
  .context-tag:hover {
    background: var(--color-bg-surface);
    border-color: var(--color-border-subtle);
  }
  
  /* Type-specific colors */
  .context-tag.file {
    border-left: 2px solid var(--color-accent-cyan);
  }
  
  .context-tag.strategy {
    border-left: 2px solid var(--color-accent-amber);
  }
  
  .context-tag.broker {
    border-left: 2px solid var(--color-accent-green);
  }
  
  .context-tag.backtest {
    border-left: 2px solid var(--color-accent-amber);
  }
  
  .tag-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--color-text-muted);
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
    color: var(--color-text-muted);
    cursor: pointer;
    padding: 0;
    margin-left: 2px;
    transition: all 0.15s;
  }
  
  .remove-btn:hover {
    background: rgba(239, 68, 68, 0.2);
    color: var(--color-accent-red);
  }
  
  /* Tooltip */
  .tooltip {
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    background: var(--color-bg-surface);
    border: 1px solid var(--color-border-subtle);
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
    border-top-color: var(--color-border-subtle);
  }
  
  .tooltip-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--color-border-subtle);
  }
  
  .tooltip-name {
    font-size: 12px;
    font-weight: 600;
    color: var(--color-text-primary);
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
    color: var(--color-text-muted);
  }
  
  .tooltip-value {
    font-size: 11px;
    color: var(--color-text-primary);
  }
</style>

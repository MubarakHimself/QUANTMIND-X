<script lang="ts">
  /**
   * ModeIndicator Component
   * 
   * Displays a visual indicator for demo/live trading mode.
   * Demo: Yellow/amber badge with 🧪 icon
   * Live: Red badge with 🔴 icon
   */
  
  export let mode: 'demo' | 'live' | string = 'live';
  export let size: 'sm' | 'md' | 'lg' = 'md';
  export let showLabel: boolean = true;
  
  $: isDemo = mode === 'demo';
  $: modeLabel = isDemo ? 'DEMO' : 'LIVE';
  $: modeIcon = isDemo ? '🧪' : '🔴';
  
  // Size classes
  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-sm px-3 py-1',
    lg: 'text-base px-4 py-1.5'
  };
</script>

{#if isDemo}
  <!-- Demo mode badge -->
  <span 
    class="mode-indicator demo inline-flex items-center gap-1.5 font-semibold rounded-full {sizeClasses[size]}"
    style="background-color: rgba(251, 191, 36, 0.2); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.4);"
  >
    <span class="icon">{modeIcon}</span>
    {#if showLabel}
      <span class="label">{modeLabel}</span>
    {/if}
  </span>
{:else}
  <!-- Live mode badge -->
  <span 
    class="mode-indicator live inline-flex items-center gap-1.5 font-semibold rounded-full {sizeClasses[size]}"
    style="background-color: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.4);"
  >
    <span class="icon">{modeIcon}</span>
    {#if showLabel}
      <span class="label">{modeLabel}</span>
    {/if}
  </span>
{/if}

<style>
  .mode-indicator {
    transition: all 0.2s ease;
  }
  
  .mode-indicator:hover {
    transform: scale(1.05);
  }
  
  .mode-indicator.demo {
    animation: pulse-demo 2s infinite;
  }
  
  .mode-indicator.live {
    animation: pulse-live 3s infinite;
  }
  
  @keyframes pulse-demo {
    0%, 100% {
      box-shadow: 0 0 0 0 rgba(251, 191, 36, 0.4);
    }
    50% {
      box-shadow: 0 0 0 4px rgba(251, 191, 36, 0);
    }
  }
  
  @keyframes pulse-live {
    0%, 100% {
      box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4);
    }
    50% {
      box-shadow: 0 0 0 4px rgba(239, 68, 68, 0);
    }
  }
</style>
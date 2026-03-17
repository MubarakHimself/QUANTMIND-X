<script lang="ts">
  import CopilotPanel from './trading-floor/CopilotPanel.svelte';

  let activeTab: 'copilot' | 'floor-manager' = $state('floor-manager');
  const tabs = [
    { id: 'floor-manager', label: 'Floor Manager' },
    { id: 'copilot', label: 'QuantMind Copilot' }
  ];

  function selectTab(tabId: string) {
    activeTab = tabId as 'copilot' | 'floor-manager';
  }
</script>

<div class="trading-floor-panel">
  <div class="panel-tabs">
    {#each tabs as tab}
      <button
        class="tab-btn"
        class:active={activeTab === tab.id}
        onclick={() => selectTab(tab.id)}
      >
        {tab.label}
      </button>
    {/each}
  </div>

  <div class="panel-content">
    {#if activeTab === 'copilot'}
      <div class="copilot-chat">
        <CopilotPanel isCopilot={true} />
      </div>
    {:else}
      <div class="floor-manager-panel">
        <CopilotPanel isCopilot={false} />
      </div>
    {/if}
  </div>
</div>

<style>
  .trading-floor-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary, #121212);
    overflow: hidden;
  }

  .panel-tabs {
    display: flex;
    flex-shrink: 0;
    border-bottom: 1px solid var(--border-color, #333);
  }

  .tab-btn {
    flex: 1;
    padding: 10px 8px;
    background: transparent;
    border: none;
    color: var(--text-secondary, #888);
    cursor: pointer;
    transition: all 0.2s;
    font-size: 12px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .tab-btn:hover {
    color: var(--text-primary, #e0e0e0);
    background: var(--bg-hover, #1a1a1a);
  }

  .tab-btn.active {
    color: var(--accent-color, #4a9eff);
    border-bottom: 2px solid var(--accent-color, #4a9eff);
  }

  .panel-content {
    flex: 1;
    overflow: hidden;
    min-height: 0;
  }

  .copilot-chat,
  .floor-manager-panel {
    height: 100%;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .copilot-chat :global(*),
  .floor-manager-panel :global(*) {
    overflow: hidden !important;
  }
</style>

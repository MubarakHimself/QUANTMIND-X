<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { Bot, Code, Wand2, Plus, Settings, PanelLeftClose, PanelLeft } from 'lucide-svelte';
  import type { AgentType } from '../../stores/chatStore';
  
  // Props
  export let agents: Array<{ id: AgentType; name: string; icon: string; description: string }>;
  export let activeAgent: AgentType;
  export let showChatList: boolean = true;
  
  const dispatch = createEventDispatcher();
  
  // Icon mapping
  const iconMap: Record<string, any> = {
    Bot,
    Code,
    Wand2
  };
  
  // Handle agent tab click
  function handleAgentClick(agentId: AgentType) {
    dispatch('agentSwitch', agentId);
  }
  
  // Handle new chat
  function handleNewChat() {
    dispatch('newChat');
  }
  
  // Handle settings
  function handleSettings() {
    dispatch('toggleSettings');
  }
  
  // Handle chat list toggle
  function handleToggleChatList() {
    dispatch('toggleChatList');
  }
</script>

<header class="agent-header">
  <!-- Agent Tabs -->
  <div class="agent-tabs" role="tablist" aria-label="Agent selection">
    {#each agents as agent}
      <button
        class="agent-tab"
        class:active={activeAgent === agent.id}
        on:click={() => handleAgentClick(agent.id)}
        role="tab"
        aria-selected={activeAgent === agent.id}
        aria-controls="chat-panel-{agent.id}"
        title={agent.description}
      >
        <svelte:component this={iconMap[agent.icon] || Bot} size={18} />
        <span class="agent-name">{agent.name}</span>
      </button>
    {/each}
  </div>
  
  <!-- Action Buttons -->
  <div class="header-actions">
    <button 
      class="action-btn" 
      on:click={handleNewChat}
      title="New Chat"
      aria-label="Create new chat"
    >
      <Plus size={16} />
    </button>
    
    <button 
      class="action-btn toggle-list-btn"
      class:active={showChatList}
      on:click={handleToggleChatList}
      title={showChatList ? 'Hide chat list' : 'Show chat list'}
      aria-label={showChatList ? 'Hide chat list' : 'Show chat list'}
    >
      {#if showChatList}
        <PanelLeft size={16} />
      {:else}
        <PanelLeftClose size={16} />
      {/if}
    </button>
    
    <button 
      class="action-btn" 
      on:click={handleSettings}
      title="Settings"
      aria-label="Open settings"
    >
      <Settings size={16} />
    </button>
  </div>
</header>

<style>
  .agent-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 8px;
    border-bottom: 1px solid var(--border-subtle);
    background: var(--bg-secondary);
    min-height: 48px;
  }
  
  .agent-tabs {
    display: flex;
    flex: 1;
    gap: 2px;
  }
  
  .agent-tab {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    padding: 8px 16px;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s ease;
    border-radius: 4px 4px 0 0;
  }
  
  .agent-tab:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }
  
  .agent-tab.active {
    color: var(--accent-primary);
    border-bottom-color: var(--accent-primary);
    background: var(--bg-tertiary);
  }
  
  .agent-tab.active .agent-name {
    font-weight: 600;
  }
  
  .agent-name {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.02em;
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
    width: 32px;
    height: 32px;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s ease;
  }
  
  .action-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }
  
  .action-btn:active {
    transform: scale(0.95);
  }
  
  .toggle-list-btn.active {
    color: var(--accent-primary);
  }
  
  /* Responsive adjustments */
  @media (max-width: 400px) {
    .agent-tab {
      padding: 8px 12px;
    }
    
    .agent-name {
      display: none;
    }
  }
</style>

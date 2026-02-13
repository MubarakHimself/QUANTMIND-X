<script lang="ts">
  import { createEventDispatcher, onMount, tick } from 'svelte';
  import { fade, slide } from 'svelte/transition';
  import { 
    BarChart3, 
    Paperclip, 
    FileSearch, 
    PlusCircle, 
    Zap, 
    Database, 
    Sparkles,
    Settings,
    HelpCircle,
    Terminal,
    Trash2,
    Upload
  } from 'lucide-svelte';
  import commandHandler from '../../services/commandHandler';
  import type { Command } from '../../services/commandHandler';
  
  // Props
  export let filter: string = '';
  
  const dispatch = createEventDispatcher();
  
  // Icon mapping for commands
  const iconMap: Record<string, typeof BarChart3> = {
    '/backtest': BarChart3,
    '/attach': Paperclip,
    '/analyze': FileSearch,
    '/add-broker': PlusCircle,
    '/kill': Zap,
    '/memory': Database,
    '/skills': Sparkles,
    '/settings': Settings,
    '/help': HelpCircle,
    '/terminal': Terminal,
    '/clear': Trash2,
    '/export': Upload
  };
  
  // Get commands from commandHandler
  $: allCommands = commandHandler.getAll().map(cmd => ({
    name: cmd.name,
    params: cmd.params,
    description: cmd.description,
    icon: iconMap[cmd.name] || BarChart3,
    category: cmd.category
  }));
  
  // State
  let selectedIndex = 0;
  let commandListElement: HTMLDivElement;
  
  // Filter commands based on input
  $: filteredCommands = filter.trim()
    ? allCommands.filter(cmd => 
        cmd.name.toLowerCase().includes(filter.toLowerCase()) ||
        cmd.description.toLowerCase().includes(filter.toLowerCase())
      )
    : allCommands;
  
  // Group commands by category
  $: groupedCommands = groupCommandsByCategory(filteredCommands);
  
  // Reset selection when filter changes
  $: {
    filter;
    selectedIndex = 0;
  }
  
  function groupCommandsByCategory(commands: typeof allCommands) {
    const groups: Record<string, typeof allCommands> = {};
    commands.forEach(cmd => {
      if (!groups[cmd.category]) {
        groups[cmd.category] = [];
      }
      groups[cmd.category].push(cmd);
    });
    return groups;
  }
  
  function getCategoryLabel(category: string): string {
    const labels: Record<string, string> = {
      trading: 'Trading',
      context: 'Context',
      agent: 'Agent',
      system: 'System'
    };
    return labels[category] || category;
  }
  
  function getCategoryColor(category: string): string {
    const colors: Record<string, string> = {
      trading: 'var(--accent-primary)',
      context: 'var(--accent-secondary)',
      agent: 'var(--accent-success)',
      system: 'var(--accent-warning)'
    };
    return colors[category] || 'var(--text-muted)';
  }
  
  // Handle command selection
  function selectCommand(command: typeof allCommands[0]) {
    dispatch('select', command.name);
  }
  
  // Handle keyboard navigation
  export function handleKeyDown(e: KeyboardEvent): boolean {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      selectedIndex = Math.min(selectedIndex + 1, filteredCommands.length - 1);
      scrollToSelected();
      return true;
    }
    
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      selectedIndex = Math.max(selectedIndex - 1, 0);
      scrollToSelected();
      return true;
    }
    
    if (e.key === 'Enter' && filteredCommands[selectedIndex]) {
      e.preventDefault();
      selectCommand(filteredCommands[selectedIndex]);
      return true;
    }
    
    if (e.key === 'Escape') {
      dispatch('close');
      return true;
    }
    
    return false;
  }
  
  // Scroll to keep selected item visible
  async function scrollToSelected() {
    await tick();
    if (commandListElement) {
      const selectedElement = commandListElement.querySelector('.command-item.selected');
      if (selectedElement) {
        selectedElement.scrollIntoView({ block: 'nearest' });
      }
    }
  }
  
  // Get global index for selection
  function getGlobalIndex(categoryIndex: number, cmdIndex: number): number {
    let index = 0;
    const categories = Object.keys(groupedCommands);
    for (let i = 0; i < categoryIndex; i++) {
      index += groupedCommands[categories[i]].length;
    }
    return index + cmdIndex;
  }
</script>

<div 
  class="slash-command-palette" 
  bind:this={commandListElement}
  transition:slide={{ duration: 150 }}
  role="listbox"
  aria-label="Available commands"
>
  <!-- Header -->
  <div class="palette-header">
    <span class="header-title">Commands</span>
    {#if filter}
      <span class="filter-badge">/{filter}</span>
    {/if}
  </div>
  
  <!-- Command list -->
  <div class="command-list">
    {#if filteredCommands.length === 0}
      <div class="empty-state">
        <span>No commands found</span>
      </div>
    {:else}
      {#each Object.entries(groupedCommands) as [category, commands], categoryIndex}
        <div class="command-group">
          <div class="group-header" style="color: {getCategoryColor(category)}">
            {getCategoryLabel(category)}
          </div>
          {#each commands as command, cmdIndex}
            {@const globalIndex = getGlobalIndex(categoryIndex, cmdIndex)}
            <button
              class="command-item"
              class:selected={selectedIndex === globalIndex}
              on:click={() => selectCommand(command)}
              on:mouseenter={() => selectedIndex = globalIndex}
              role="option"
              aria-selected={selectedIndex === globalIndex}
            >
              <div class="command-icon" style="color: {getCategoryColor(category)}">
                <svelte:component this={command.icon} size={14} />
              </div>
              <div class="command-info">
                <span class="command-name">{command.name}</span>
                {#if command.params}
                  <span class="command-params">{command.params}</span>
                {/if}
              </div>
              <span class="command-description">{command.description}</span>
            </button>
          {/each}
        </div>
      {/each}
    {/if}
  </div>
  
  <!-- Footer hint -->
  <div class="palette-footer">
    <span class="hint">
      <kbd>↑↓</kbd> Navigate
      <kbd>Enter</kbd> Select
      <kbd>Esc</kbd> Close
    </span>
  </div>
</div>

<style>
  .slash-command-palette {
    position: absolute;
    bottom: calc(100% + 8px);
    left: 0;
    right: 0;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    z-index: 100;
    overflow: hidden;
    max-height: 320px;
    display: flex;
    flex-direction: column;
  }
  
  .palette-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    border-bottom: 1px solid var(--border-subtle);
    background: var(--bg-tertiary);
  }
  
  .header-title {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  
  .filter-badge {
    font-size: 10px;
    padding: 2px 6px;
    background: var(--accent-primary);
    color: var(--bg-primary);
    border-radius: 4px;
    font-family: monospace;
  }
  
  .command-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px 0;
  }
  
  .empty-state {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
    color: var(--text-muted);
    font-size: 12px;
  }
  
  .command-group {
    margin-bottom: 8px;
  }
  
  .command-group:last-child {
    margin-bottom: 0;
  }
  
  .group-header {
    padding: 6px 12px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  
  .command-item {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    padding: 8px 12px;
    background: transparent;
    border: none;
    color: var(--text-primary);
    cursor: pointer;
    transition: background 0.1s;
    text-align: left;
  }
  
  .command-item:hover,
  .command-item.selected {
    background: var(--bg-tertiary);
  }
  
  .command-item.selected {
    background: rgba(107, 200, 230, 0.1);
  }
  
  .command-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: var(--bg-primary);
    border-radius: 6px;
    flex-shrink: 0;
  }
  
  .command-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 100px;
  }
  
  .command-name {
    font-size: 12px;
    font-weight: 600;
    font-family: monospace;
  }
  
  .command-params {
    font-size: 10px;
    color: var(--text-muted);
    font-family: monospace;
  }
  
  .command-description {
    flex: 1;
    font-size: 11px;
    color: var(--text-secondary);
    text-align: right;
  }
  
  .palette-footer {
    padding: 8px 12px;
    border-top: 1px solid var(--border-subtle);
    background: var(--bg-tertiary);
  }
  
  .hint {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 10px;
    color: var(--text-muted);
  }
  
  .hint kbd {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 2px 6px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    font-family: inherit;
    font-size: 9px;
  }
</style>

<script lang="ts">
  import { run } from 'svelte/legacy';

  import { createEventDispatcher, tick } from 'svelte';
  import { slide } from 'svelte/transition';
  import {
    FileText,
    Database,
    Network,
    Zap,
    BarChart3,
    Settings,
    RefreshCw,
    Folder,
    Terminal,
    Code,
    Search,
    Sparkles
  } from 'lucide-svelte';
  import skillChatService from '../../services/skillChatService';
  import type { AgentType } from '../../stores/chatStore';

  
  interface Props {
    // Props
    agent: AgentType;
    filter?: string;
  }

  let { agent, filter = '' }: Props = $props();

  const dispatch = createEventDispatcher();

  // Icon mapping for skill categories
  const categoryIconMap: Record<string, typeof FileText> = {
    file_operations: FileText,
    broker: Database,
    deployment: Network,
    sync: RefreshCw,
    trading: BarChart3,
    analysis: Search,
    code: Code,
    general: Sparkles
  };




  // State
  let selectedIndex = $state(0);


  function groupSkillsByCategory(skills: typeof allSkills) {
    const groups: Record<string, typeof skills> = {};
    skills.forEach(skill => {
      const category = skill.category || 'general';
      if (!groups[category]) {
        groups[category] = [];
      }
      groups[category].push(skill);
    });
    return groups;
  }

  function getCategoryLabel(category: string): string {
    const labels: Record<string, string> = {
      file_operations: 'File Operations',
      broker: 'Broker',
      deployment: 'Deployment',
      sync: 'Synchronization',
      trading: 'Trading',
      analysis: 'Analysis',
      code: 'Code',
      general: 'General'
    };
    return labels[category] || category;
  }

  function getCategoryColor(category: string): string {
    const colors: Record<string, string> = {
      file_operations: 'var(--color-accent-cyan)',
      broker: 'var(--color-accent-amber)',
      deployment: 'var(--color-accent-green)',
      sync: 'var(--color-accent-amber)',
      trading: 'var(--color-accent-cyan)',
      analysis: 'var(--color-accent-amber)',
      code: 'var(--color-accent-green)',
      general: 'var(--color-text-muted)'
    };
    return colors[category] || 'var(--color-text-muted)';
  }

  function getSkillIcon(category: string) {
    return categoryIconMap[category] || Sparkles;
  }

  // Handle skill selection
  function selectSkill(skill: typeof allSkills[0]) {
    dispatch('select', `@${skill.id}`);
  }

  // Handle keyboard navigation
  export function handleKeyDown(e: KeyboardEvent): boolean {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      selectedIndex = Math.min(selectedIndex + 1, filteredSkills.length - 1);
      return true;
    }

    if (e.key === 'ArrowUp') {
      e.preventDefault();
      selectedIndex = Math.max(selectedIndex - 1, 0);
      return true;
    }

    if (e.key === 'Enter' && filteredSkills[selectedIndex]) {
      e.preventDefault();
      selectSkill(filteredSkills[selectedIndex]);
      return true;
    }

    if (e.key === 'Escape') {
      dispatch('close');
      return true;
    }

    return false;
  }

  // Get global index for selection
  function getGlobalIndex(categoryIndex: number, skillIndex: number): number {
    let index = 0;
    const categories = Object.keys(groupedSkills);
    for (let i = 0; i < categoryIndex; i++) {
      index += groupedSkills[categories[i]].length;
    }
    return index + skillIndex;
  }
  // Get available skills
  let allSkills = $derived(skillChatService.getSuggestions(agent));
  // Filter skills based on input
  let filteredSkills = $derived(filter.trim()
    ? allSkills.filter(skill =>
        skill.name.toLowerCase().includes(filter.toLowerCase()) ||
        skill.description.toLowerCase().includes(filter.toLowerCase()) ||
        skill.id.toLowerCase().includes(filter.toLowerCase())
      )
    : allSkills);
  // Group skills by category
  let groupedSkills = $derived(groupSkillsByCategory(filteredSkills));
  // Reset selection when filter changes
  run(() => {
    filter;
    selectedIndex = 0;
  });
</script>

<div
  class="skill-palette"
  transition:slide={{ duration: 150 }}
  role="listbox"
  aria-label="Available skills"
>
  <!-- Header -->
  <div class="palette-header">
    <span class="header-title">Skills</span>
    {#if filter}
      <span class="filter-badge">@{filter}</span>
    {/if}
  </div>

  <!-- Skill list -->
  <div class="skill-list">
    {#if filteredSkills.length === 0}
      <div class="empty-state">
        <span>No skills found</span>
      </div>
    {:else}
      {#each Object.entries(groupedSkills) as [category, skills], categoryIndex}
        <div class="skill-group">
          <div class="group-header" style="color: {getCategoryColor(category)}">
            {getCategoryLabel(category)}
          </div>
          {#each skills as skill, skillIndex}
            {@const globalIndex = getGlobalIndex(categoryIndex, skillIndex)}
            {@const SvelteComponent = getSkillIcon(category)}
            <button
              class="skill-item"
              class:selected={selectedIndex === globalIndex}
              onclick={() => selectSkill(skill)}
              onmouseenter={() => selectedIndex = globalIndex}
              role="option"
              aria-selected={selectedIndex === globalIndex}
            >
              <div class="skill-icon" style="color: {getCategoryColor(category)}">
                <SvelteComponent size={14} />
              </div>
              <div class="skill-info">
                <span class="skill-name">@{skill.id.replace(/_/g, ' ')}</span>
              </div>
              <span class="skill-description">{skill.description}</span>
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
  .skill-palette {
    position: absolute;
    bottom: calc(100% + 8px);
    left: 0;
    right: 0;
    background: var(--color-bg-surface);
    border: 1px solid var(--color-border-subtle);
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
    border-bottom: 1px solid var(--color-border-subtle);
    background: var(--color-bg-elevated);
  }

  .header-title {
    font-size: 11px;
    font-weight: 600;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .filter-badge {
    font-size: 10px;
    padding: 2px 6px;
    background: var(--color-accent-amber);
    color: var(--color-bg-base);
    border-radius: 4px;
    font-family: monospace;
  }

  .skill-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px 0;
  }

  .empty-state {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
    color: var(--color-text-muted);
    font-size: 12px;
  }

  .skill-group {
    margin-bottom: 8px;
  }

  .skill-group:last-child {
    margin-bottom: 0;
  }

  .group-header {
    padding: 6px 12px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .skill-item {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    padding: 8px 12px;
    background: transparent;
    border: none;
    color: var(--color-text-primary);
    cursor: pointer;
    transition: background 0.1s;
    text-align: left;
  }

  .skill-item:hover,
  .skill-item.selected {
    background: var(--color-bg-elevated);
  }

  .skill-item.selected {
    background: rgba(107, 200, 230, 0.1);
  }

  .skill-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: var(--color-bg-base);
    border-radius: 6px;
    flex-shrink: 0;
  }

  .skill-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 100px;
  }

  .skill-name {
    font-size: 12px;
    font-weight: 600;
    font-family: monospace;
  }

  .skill-description {
    flex: 1;
    font-size: 11px;
    color: var(--color-text-secondary);
    text-align: right;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .palette-footer {
    padding: 8px 12px;
    border-top: 1px solid var(--color-border-subtle);
    background: var(--color-bg-elevated);
  }

  .hint {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 10px;
    color: var(--color-text-muted);
  }

  .hint kbd {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 2px 6px;
    background: var(--color-bg-base);
    border: 1px solid var(--color-border-subtle);
    border-radius: 4px;
    font-family: inherit;
    font-size: 9px;
  }
</style>

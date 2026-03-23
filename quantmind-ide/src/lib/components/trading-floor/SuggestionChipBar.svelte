<script lang="ts">
  import { onMount } from 'svelte';
  import { ChevronRight, Sparkles, Shield, Activity, Hammer, FolderOpen, Calendar, Pause, Sun, FileText, Briefcase, Search, Code } from 'lucide-svelte';
  import { canvasContextService, type CanvasSuggestionChip } from '$lib/services/canvasContextService';
  import { activeCanvasStore } from '$lib/stores/canvasStore';
  import { goto } from '$app/navigation';
  import { intentService } from '$lib/services/intentService';

  interface Props {
    chips?: CanvasSuggestionChip[];
    onNavigate?: (targetCanvas: string, targetEntity?: string) => void;
  }

  let { chips = [], onNavigate }: Props = $props();

  // Reactive state for suggestion chips
  let suggestionChips = $state<CanvasSuggestionChip[]>([]);
  let loading = $state(true);

  // Subscribe to canvas changes and load chips reactively
  $effect(() => {
    // This effect runs when:
    // 1. Component mounts
    // 2. chips prop changes
    // 3. active canvas changes (via subscription)

    async function loadChips() {
      loading = true;
      try {
        if (chips.length > 0) {
          suggestionChips = chips;
        } else {
          suggestionChips = await canvasContextService.getSuggestionChips();
        }
      } catch (error) {
        console.error('Failed to load suggestion chips:', error);
        suggestionChips = [];
      } finally {
        loading = false;
      }
    }

    loadChips();

    // Subscribe to canvas store for reactive updates
    const unsubscribe = activeCanvasStore.subscribe(() => {
      // Reload chips when canvas changes
      if (chips.length === 0) {
        loadChips();
      }
    });

    return () => {
      unsubscribe();
    };
  });

  async function handleChipClick(chip: CanvasSuggestionChip) {
    // Check if this is a slash command that should be executed via FloorManager
    if (chip.label.startsWith('/')) {
      try {
        // Execute the command via intent service
        const response = await intentService.sendCommand(chip.label, false);

        if (response.type === 'error') {
          console.error('Command execution failed:', response.message);
          // Optionally show error to user
        } else if (response.type === 'confirmation_needed') {
          // Handle confirmation if needed - for now just log
          console.log('Command needs confirmation:', response.message);
        } else {
          // Command executed successfully - message will be handled by chat store
          console.log('Command executed:', response.message);
        }
      } catch (error) {
        console.error('Failed to execute command:', error);
      }
      return;
    }

    // Otherwise, navigate to target canvas
    if (onNavigate) {
      onNavigate(chip.target_canvas, chip.target_entity);
    } else {
      // Default navigation behavior - navigate to target canvas
      await goto(`/${chip.target_canvas}`);
    }
  }

  function getIconForChip(iconName?: string) {
    switch (iconName?.toLowerCase()) {
      case 'hammer':
        return Hammer;
      case 'shield':
        return Shield;
      case 'activity':
        return Activity;
      case 'sparkles':
        return Sparkles;
      case 'folder':
        return FolderOpen;
      case 'calendar':
        return Calendar;
      case 'pause':
        return Pause;
      case 'sun':
        return Sun;
      case 'filetext':
        return FileText;
      case 'briefcase':
        return Briefcase;
      case 'search':
        return Search;
      case 'code':
        return Code;
      default:
        return ChevronRight;
    }
  }
</script>

<div class="suggestion-chip-bar">
  <div class="chips-container">
    {#each suggestionChips as chip (chip.id)}
      <button
        class="suggestion-chip"
        onclick={() => handleChipClick(chip)}
        type="button"
      >
        {#if chip.icon}
          <svelte:component this={getIconForChip(chip.icon)} size={14} />
        {/if}
        <span class="chip-label">{chip.label}</span>
        <ChevronRight size={12} class="chip-arrow" />
      </button>
    {/each}
  </div>
</div>

<style>
  .suggestion-chip-bar {
    display: flex;
    align-items: center;
    padding: 8px 12px;
    background: transparent;
    overflow-x: auto;
    scrollbar-width: thin;
    -webkit-overflow-scrolling: touch;
    scroll-behavior: smooth;
  }

  .suggestion-chip-bar::-webkit-scrollbar {
    height: 4px;
  }

  .suggestion-chip-bar::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 2px;
  }

  .suggestion-chip-bar::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 2px;
  }

  .chips-container {
    display: flex;
    gap: 8px;
    align-items: center;
    padding: 2px 0;
  }

  .suggestion-chip {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    color: rgba(255, 255, 255, 0.85);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    white-space: nowrap;
  }

  .suggestion-chip:hover {
    background: rgba(255, 255, 255, 0.15);
    border-color: rgba(255, 255, 255, 0.25);
    transform: translateY(-1px);
  }

  .chip-label {
    max-width: 120px;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .chip-arrow {
    opacity: 0.5;
    transition: opacity 0.2s ease;
  }

  .suggestion-chip:hover .chip-arrow {
    opacity: 1;
  }

  /* Glass effect matching frosted terminal aesthetic */
  .suggestion-chip {
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
  }
</style>

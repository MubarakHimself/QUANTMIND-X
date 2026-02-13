<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import {
    BookOpen, Boxes, Bot, TestTube, PlayCircle, Settings,
    Database, Server, FileText, Edit3
  } from 'lucide-svelte';
  import { navigationStore } from '../stores/navigationStore';

  const dispatch = createEventDispatcher();
  let unsubscribe: (() => void) | null = null;

  // Left sidebar icons - clicking opens view in MAIN EDITOR
  // Removed duplicates: backtest-results, shared-assets, kill-switch, database-view, news
  // These are now accessible as sub-tabs within their parent sections
  const activities = [
    { id: 'knowledge', icon: BookOpen, label: 'Knowledge Hub' },
    { id: 'assets', icon: Boxes, label: 'Shared Assets & Database' },
    { id: 'ea', icon: Bot, label: 'EA Management' },
    { id: 'backtest', icon: TestTube, label: 'Backtests' },
    { id: 'live', icon: PlayCircle, label: 'Live Trading' },
    { id: 'router', icon: Server, label: 'Strategy Router' },
    { id: 'journal', icon: FileText, label: 'Trade Journal' },
    { id: 'editor', icon: Edit3, label: 'Editor Workspace' },
  ];

  export let activeView = 'live';

  // Subscribe to navigation store for view changes
  $: activeView = $navigationStore.currentView;

  function selectView(viewId: string) {
    const activity = activities.find(a => a.id === viewId);
    if (activity) {
      navigationStore.navigateToView(viewId, activity.label);
      dispatch('viewChange', { view: viewId });
    }
  }
</script>

<aside class="activity-bar">
  <div class="top-icons">
    {#each activities as activity}
      <button
        class="activity-icon"
        class:active={activeView === activity.id}
        on:click={() => selectView(activity.id)}
        title={activity.label}
      >
        <svelte:component this={activity.icon} size={22} />
      </button>
    {/each}
  </div>
  
  <div class="bottom-icons">
    <button
      class="activity-icon"
      class:active={activeView === 'settings'}
      on:click={() => selectView('settings')}
      title="Settings"
    >
      <Settings size={22} />
    </button>
  </div>
</aside>

<style>
  .activity-bar {
    grid-area: activity;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    width: 48px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-subtle);
    padding: 8px 0;
  }
  
  .top-icons, .bottom-icons {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
  }
  
  .activity-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    background: transparent;
    border: none;
    border-radius: 8px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s ease;
    position: relative;
  }
  
  .activity-icon:hover {
    color: var(--text-primary);
    background: var(--bg-tertiary);
  }
  
  .activity-icon.active {
    color: var(--accent-primary);
    background: var(--bg-tertiary);
  }
  
  .activity-icon.active::before {
    content: '';
    position: absolute;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    width: 3px;
    height: 24px;
    background: var(--accent-primary);
    border-radius: 0 2px 2px 0;
  }
</style>

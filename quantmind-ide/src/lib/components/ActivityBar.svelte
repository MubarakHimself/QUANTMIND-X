<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { activeCanvasStore, CANVAS_SHORTCUTS, CANVASES } from "../stores/canvasStore";
  import { Activity, Search, Code2, Shield, BarChart2, PieChart, FolderOpen, MessageSquare, GitBranch } from 'lucide-svelte';

  const ICONS: Record<string, any> = {
    'live-trading':  Activity,
    'research':      Search,
    'development':   Code2,
    'risk':          Shield,
    'trading':       BarChart2,
    'portfolio':     PieChart,
    'shared-assets': FolderOpen,
    'workshop':      MessageSquare,
    'flowforge':     GitBranch,
  };

  let activeView = $derived($activeCanvasStore);

  function selectCanvas(canvasId: string) {
    activeCanvasStore.setActiveCanvas(canvasId);
  }

  function handleKeydown(event: KeyboardEvent) {
    const key = event.key;
    if (key >= '1' && key <= '9' && !isInputFocused()) {
      const canvasId = CANVAS_SHORTCUTS[key];
      if (canvasId) {
        event.preventDefault();
        selectCanvas(canvasId);
      }
    }
  }

  function isInputFocused(): boolean {
    const el = document.activeElement;
    return el?.tagName === 'INPUT' ||
           el?.tagName === 'TEXTAREA' ||
           el?.getAttribute('contenteditable') === 'true';
  }

  onMount(() => {
    window.addEventListener('keydown', handleKeydown);
  });

  onDestroy(() => {
    window.removeEventListener('keydown', handleKeydown);
  });
</script>

<aside class="activity-bar">
  <div class="canvas-icons">
    {#each CANVASES as canvas}
      <button
        class="canvas-icon"
        class:active={activeView === canvas.id}
        onclick={() => selectCanvas(canvas.id)}
        title={canvas.name}
      >
        <svelte:component this={ICONS[canvas.id] ?? Activity} size={16} strokeWidth={1.5} />
      </button>
    {/each}
  </div>
</aside>

<style>
  .activity-bar {
    grid-area: activity;
    width: var(--sidebar-width);
    min-width: var(--sidebar-width);
    background: var(--glass-tier-1);
    backdrop-filter: var(--glass-blur);
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 8px 0;
    border-right: 1px solid rgba(255, 255, 255, 0.04);
    z-index: 10;
    overflow: hidden;
  }

  .canvas-icons {
    display: flex;
    flex-direction: column;
    gap: 1px;
    width: 100%;
    align-items: center;
  }

  .canvas-icon {
    width: 40px;
    height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: transparent;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    transition: all 0.12s ease;
    position: relative;
    color: var(--color-text-muted);
  }

  .canvas-icon:hover {
    background: rgba(255, 255, 255, 0.05);
    color: var(--color-text-primary);
  }

  .canvas-icon.active {
    background: rgba(0, 170, 204, 0.1);
    color: var(--color-accent-cyan);
  }

  .canvas-icon.active::before {
    content: "";
    position: absolute;
    left: 0;
    top: 4px;
    bottom: 4px;
    width: 2px;
    background: var(--color-accent-cyan);
    border-radius: 0 2px 2px 0;
    box-shadow: 0 0 6px var(--color-accent-cyan);
  }
</style>

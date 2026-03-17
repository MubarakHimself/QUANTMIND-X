<script lang="ts">
  import { run } from 'svelte/legacy';
  import { createEventDispatcher, onMount, onDestroy } from 'svelte';
  import {
    Wrench,
    BookOpen,
    Bot,
    FlaskConical,
    MonitorPlay,
    PlayCircle,
    Activity,
    Server,
    FileText,
  } from "lucide-svelte";
  import { navigationStore } from "../stores/navigationStore";
  import { activeCanvasStore, CANVAS_SHORTCUTS, CANVASES } from "../stores/canvasStore";

  const dispatch = createEventDispatcher();
  let unsubscribe: (() => void) | null = null;

  // Type definition for canvas items
  interface CanvasItem {
    id: string;
    icon: any;
    label: string;
    shortcut: string;
  }

  // 9 canvas navigation icons
  const canvases: CanvasItem[] = [
    { id: "workshop", icon: Wrench, label: "Workshop", shortcut: "1" },
    { id: "knowledge", icon: BookOpen, label: "Knowledge Hub", shortcut: "2" },
    { id: "ea", icon: Bot, label: "EA Management", shortcut: "3" },
    { id: "backtest", icon: FlaskConical, label: "Backtests", shortcut: "4" },
    { id: "paper-trading", icon: MonitorPlay, label: "Paper Trading", shortcut: "5" },
    { id: "live", icon: PlayCircle, label: "Live Trading", shortcut: "6" },
    { id: "router", icon: Server, label: "Strategy Router", shortcut: "7" },
    { id: "hmm", icon: Activity, label: "HMM Dashboard", shortcut: "8" },
    { id: "journal", icon: FileText, label: "Trade Journal", shortcut: "9" },
  ];

  interface Props {
    activeView?: string;
  }

  let { activeView = $bindable("live") }: Props = $props();

  // Subscribe to navigation store for view changes
  run(() => {
    activeView = $navigationStore.currentView;
  });

  function selectCanvas(canvasId: string) {
    const canvas = canvases.find((c) => c.id === canvasId);
    if (canvas) {
      navigationStore.navigateToView(canvasId, canvas.label);
      dispatch("viewChange", { view: canvasId });
    }
  }

  // Keyboard shortcut handler
  function handleKeydown(event: KeyboardEvent) {
    const key = event.key;
    // Check if key is 1-9 and not in an input field
    if (key >= '1' && key <= '9' && !isInputFocused()) {
      const index = parseInt(key) - 1;
      if (canvases[index]) {
        event.preventDefault();
        selectCanvas(canvases[index].id);
      }
    }
  }

  function isInputFocused(): boolean {
    const activeElement = document.activeElement;
    return activeElement?.tagName === 'INPUT' ||
           activeElement?.tagName === 'TEXTAREA' ||
           activeElement?.getAttribute('contenteditable') === 'true';
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
    {#each canvases as canvas, index}
      <button
        class="canvas-icon"
        class:active={activeView === canvas.id}
        onclick={() => selectCanvas(canvas.id)}
        title="{canvas.label} (Press {canvas.shortcut})"
      >
        <canvas.icon size={20} strokeWidth={1.5} />
        <span class="shortcut-badge">{canvas.shortcut}</span>
      </button>
    {/each}
  </div>
  <div class="shortcut-hint">
    <span class="hint">1-9</span>
  </div>
</aside>

<style>
  .activity-bar {
    grid-column: 1;
    grid-row: 1;
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
    transition: width 0.2s ease;
  }

  .canvas-icons {
    display: flex;
    flex-direction: column;
    gap: 2px;
    width: 100%;
    align-items: center;
  }

  .canvas-icon {
    position: relative;
    width: 44px;
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: transparent;
    border: none;
    border-radius: 8px;
    color: var(--text-muted);
    cursor: pointer;
    opacity: 0.6;
    transition: all 0.15s ease;
  }

  .canvas-icon:hover {
    opacity: 1;
    background: var(--glass-tier-2);
    color: var(--text-primary);
  }

  .canvas-icon.active {
    opacity: 1;
    color: var(--color-accent-cyan);
    background: var(--glass-tier-2);
  }

  .canvas-icon.active::before {
    content: "";
    position: absolute;
    left: 0;
    top: 8px;
    bottom: 8px;
    width: 2px;
    background: var(--color-accent-cyan);
    border-radius: 0 2px 2px 0;
    box-shadow: 0 0 8px var(--color-accent-cyan);
  }

  .shortcut-badge {
    position: absolute;
    bottom: 2px;
    right: 4px;
    font-family: var(--font-mono);
    font-size: 8px;
    font-weight: 500;
    color: var(--text-muted);
    opacity: 0;
    transition: opacity 0.15s ease;
  }

  .canvas-icon:hover .shortcut-badge {
    opacity: 0.6;
  }

  .canvas-icon.active .shortcut-badge {
    color: var(--color-accent-cyan);
    opacity: 0.8;
  }

  /* Expanded state - show labels */
  .activity-bar:hover {
    width: var(--sidebar-expanded-width);
  }

  .activity-bar:hover .canvas-icon {
    width: calc(var(--sidebar-expanded-width) - 16px);
    justify-content: flex-start;
    padding-left: 12px;
    gap: 10px;
  }

  .activity-bar:hover .canvas-icon::before {
    left: 0;
  }

  .activity-bar:hover .shortcut-badge {
    position: static;
    margin-left: auto;
    margin-right: 8px;
    opacity: 0.5;
  }

  .activity-bar:hover .canvas-icon.active .shortcut-badge {
    opacity: 0.8;
  }

  /* Add label on hover */
  .activity-bar:hover .canvas-icon::after {
    content: attr(title);
    position: absolute;
    left: 52px;
    font-family: var(--font-nav);
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
    white-space: nowrap;
    opacity: 0;
    transition: opacity 0.15s ease;
    pointer-events: none;
  }

  .activity-bar:hover .canvas-icon::after {
    opacity: 1;
  }

  .activity-bar:hover .canvas-icon.active::after {
    color: var(--color-accent-cyan);
  }
</style>
<script lang="ts">
  import { run } from 'svelte/legacy';

  import { createEventDispatcher } from "svelte";
  import {
    BookOpen,
    Boxes,
    Bot,
    TestTube,
    PlayCircle,
    Server,
    FileText,
    Edit3,
    Activity,
    MonitorPlay,
    Github,
    Wrench,
    Workflow,
    Layers,
    TrendingUp,
    Briefcase,
    Puzzle,
    GitBranch,
  } from "lucide-svelte";
  import { navigationStore } from "../stores/navigationStore";
  import { activeCanvasStore, CANVAS_SHORTCUTS, CANVASES } from "../stores/canvasStore";

  const dispatch = createEventDispatcher();
  let unsubscribe: (() => void) | null = null;

  // Type definition for activity items
  interface ActivityItem {
    id: string;
    icon: any;
    label: string;
    route?: string;
    canvasId?: string; // Map to canvas route
  }

  // Canvas-mapped activities for 9-canvas routing
  // These map the ActivityBar items to the 9 canvas routes
  const activities: ActivityItem[] = [
    { id: "workshop", icon: Wrench, label: "Workshop (8)", canvasId: "workshop" },
    { id: "knowledge", icon: BookOpen, label: "Research (2)", canvasId: "research" },
    { id: "assets", icon: Boxes, label: "Shared Assets (7)", canvasId: "shared-assets" },
    { id: "ea", icon: Bot, label: "Development (3)", canvasId: "development" },
    { id: "backtest", icon: TestTube, label: "Backtests" },
    { id: "paper-trading", icon: MonitorPlay, label: "Trading (5)", canvasId: "trading" },
    { id: "live", icon: PlayCircle, label: "Live Trading (1)", canvasId: "live-trading" },
    { id: "router", icon: Server, label: "FlowForge (9)", canvasId: "flowforge" },
    { id: "risk", icon: TrendingUp, label: "Risk (4)", canvasId: "risk" },
    { id: "portfolio", icon: Briefcase, label: "Portfolio (6)", canvasId: "portfolio" },
    { id: "journal", icon: FileText, label: "Trade Journal" },
    { id: "github-ea", icon: Github, label: "GitHub EA Library" },
    { id: "batch", icon: Layers, label: "Batch Processing" },
    { id: "editor", icon: Edit3, label: "Editor Workspace" },
    { id: "workflow", icon: Puzzle, label: "Workflows" },
    { id: "workflow-builder", icon: GitBranch, label: "Workflow Builder" },
  ];

  interface Props {
    activeView?: string;
  }

  let { activeView = $bindable("live") }: Props = $props();

  // Subscribe to navigation store for view changes
  run(() => {
    activeView = $navigationStore.currentView;
  });

  // Subscribe to canvas store to sync active canvas state
  let activeCanvas = $state('workshop');
  $effect(() => {
    const unsub = activeCanvasStore.subscribe((canvasId) => {
      activeCanvas = canvasId;
    });
    return () => unsub();
  });

  function selectView(viewId: string) {
    const activity = activities.find((a) => a.id === viewId);

    // If this activity maps to a canvas, switch to that canvas
    if (activity?.canvasId) {
      activeCanvasStore.setActiveCanvas(activity.canvasId);
      dispatch("canvasChange", { canvas: activity.canvasId });
      return;
    }

    if (activity) {
      // If it has a route, navigate to it
      if (activity.route) {
        window.location.href = activity.route;
        return;
      }
      navigationStore.navigateToView(viewId, activity.label);
      dispatch("viewChange", { view: viewId });
    }
  }

  // Handle keyboard shortcuts for canvas switching (1-9)
  function handleKeydown(event: KeyboardEvent) {
    if (event.ctrlKey || event.metaKey || event.altKey) return;

    const key = event.key;
    const canvasId = CANVAS_SHORTCUT[key];
    if (canvasId) {
      event.preventDefault();
      activeCanvasStore.setActiveCanvas(canvasId);
      dispatch("canvasChange", { canvas: canvasId });
    }
  }

  // Add keyboard listener for canvas shortcuts
  $effect(() => {
    if (typeof window !== 'undefined') {
      window.addEventListener('keydown', handleKeydown);
      return () => window.removeEventListener('keydown', handleKeydown);
    }
  });
</script>

<aside class="activity-bar">
  <div class="top-icons">
    {#each activities as activity}
      <button
        class="activity-icon"
        class:active={activity.canvasId ? activeCanvas === activity.canvasId : activeView === activity.id}
        onclick={() => selectView(activity.id)}
        title={activity.label}
      >
        <activity.icon size={22} />
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
    background: var(--bg-input); /* Darker than sidebar */
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 12px 0;
    border-right: 1px solid var(--border-subtle);
    z-index: 10;
  }

  .activity-icon {
    width: 48px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: transparent;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    position: relative;
    transition: color 0.1s ease;
  }

  .activity-icon:hover {
    color: var(--text-primary);
  }

  .activity-icon.active {
    color: var(--text-primary);
  }

  .activity-icon.active::before {
    content: "";
    position: absolute;
    left: 0;
    top: 12px;
    bottom: 12px;
    width: 2px;
    background: var(--accent-primary);
  }

  .top-icons {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .shortcut-hint {
    margin-top: auto;
    padding: 8px;
  }

  .hint {
    font-size: 10px;
    color: var(--text-disabled);
    opacity: 0.5;
  }
</style>

<script lang="ts">
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
  } from "lucide-svelte";
  import { navigationStore } from "../stores/navigationStore";

  const dispatch = createEventDispatcher();
  let unsubscribe: (() => void) | null = null;

  // Type definition for activity items
  interface ActivityItem {
    id: string;
    icon: any;
    label: string;
    route?: string;
  }

  // Left sidebar icons - clicking opens view in MAIN EDITOR
  // Removed duplicates: backtest-results, shared-assets, kill-switch, database-view, news
  // These are now accessible as sub-tabs within their parent sections
  const activities: ActivityItem[] = [
    { id: "workshop", icon: Wrench, label: "Workshop" },
    { id: "knowledge", icon: BookOpen, label: "Knowledge Hub" },
    { id: "assets", icon: Boxes, label: "Shared Assets & Database" },
    { id: "ea", icon: Bot, label: "EA Management" },
    { id: "backtest", icon: TestTube, label: "Backtests" },
    { id: "paper-trading", icon: MonitorPlay, label: "Paper Trading" },
    { id: "live", icon: PlayCircle, label: "Live Trading" },
    { id: "router", icon: Server, label: "Strategy Router" },
    { id: "hmm", icon: Activity, label: "HMM Dashboard" },
    { id: "journal", icon: FileText, label: "Trade Journal" },
    { id: "github-ea", icon: Github, label: "GitHub EA Library" },
    { id: "batch", icon: Layers, label: "Batch Processing" },
    { id: "editor", icon: Edit3, label: "Editor Workspace" },
    { id: "workflow", icon: Activity, label: "Workflows" },
    { id: "workflow-builder", icon: Workflow, label: "Workflow Builder" },
  ];

  export let activeView = "live";

  // Subscribe to navigation store for view changes
  $: activeView = $navigationStore.currentView;

  function selectView(viewId: string) {
    const activity = activities.find((a) => a.id === viewId);
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
</style>

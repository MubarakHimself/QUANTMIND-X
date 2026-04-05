<script lang="ts">
  import TopBar from "$lib/components/TopBar.svelte";
  import ActivityBar from "$lib/components/ActivityBar.svelte";
  import StatusBand from "$lib/components/StatusBand.svelte";
  import MainContent from "$lib/components/MainContent.svelte";
  import AgentPanel from "$lib/components/shell/AgentPanel.svelte";
  import { customWallpaper, wallpaperEnabled, loadSavedWallpaperEnabled, loadSavedWallpaper, setCustomWallpaper, toggleWallpaper, getActiveWallpaper } from "$lib/stores/themeStore";
  import { activeCanvasStore } from "$lib/stores/canvasStore";
  import { navigationStore } from "$lib/stores/navigationStore";
  import { onMount } from "svelte";
  import { getCanvasCollapsed, setCanvasCollapsed } from "$lib/components/shell/agentPanelState";

  // Initialize wallpaper
  onMount(async () => {
    const enabled = loadSavedWallpaperEnabled();
    toggleWallpaper(enabled);

    // First load local wallpaper
    const wallpaper = loadSavedWallpaper();
    if (wallpaper) {
      setCustomWallpaper(wallpaper);
    }

    // Then check for active custom wallpaper from backend
    const activeWallpaper = await getActiveWallpaper();
    if (activeWallpaper?.urlPath) {
      setCustomWallpaper(activeWallpaper.urlPath);
    }
  });

  let agentPanelCollapsed = $state(false);
  let agentPanelCollapsedByCanvas = $state<Record<string, boolean>>({});
  let agentPanelWidth = $state(380);
  let currentCanvas = $derived($activeCanvasStore);
  let agentPanelCollapseCanvasKey = $state($activeCanvasStore);
  let mainContentRef: MainContent;

  // Agent panel only shown on dept canvases that have active sub-agents
  const AGENT_PANEL_CANVASES = new Set(['research', 'development', 'risk', 'trading', 'portfolio', 'flowforge']);
  let showAgentPanel = $derived(AGENT_PANEL_CANVASES.has(currentCanvas));

  $effect(() => {
    if (agentPanelCollapseCanvasKey !== currentCanvas) {
      return;
    }

    const currentStoredValue = getCanvasCollapsed(agentPanelCollapsedByCanvas, currentCanvas);
    if (currentStoredValue === agentPanelCollapsed) {
      return;
    }

    agentPanelCollapsedByCanvas = setCanvasCollapsed(
      agentPanelCollapsedByCanvas,
      currentCanvas,
      agentPanelCollapsed
    );
  });

  $effect(() => {
    if (agentPanelCollapseCanvasKey === currentCanvas) {
      return;
    }

    const previousStoredValue = getCanvasCollapsed(agentPanelCollapsedByCanvas, agentPanelCollapseCanvasKey);
    if (previousStoredValue !== agentPanelCollapsed) {
      agentPanelCollapsedByCanvas = setCanvasCollapsed(
        agentPanelCollapsedByCanvas,
        agentPanelCollapseCanvasKey,
        agentPanelCollapsed
      );
    }

    const nextCollapsed = getCanvasCollapsed(agentPanelCollapsedByCanvas, currentCanvas);
    if (nextCollapsed !== agentPanelCollapsed) {
      agentPanelCollapsed = nextCollapsed;
    }

    agentPanelCollapseCanvasKey = currentCanvas;
  });

  let openFiles: Array<{
    id: string;
    name: string;
    content?: string;
    type?: string;
  }> = $state([]);
  let activeTabId = $state("");

  function handleOpenSettings() {
    mainContentRef?.onSettingsNavigationRequest();
  }

  function handleOpenFile(event: CustomEvent) {
    const file = event.detail;
    if (!openFiles.find((f) => f.id === file.id)) {
      openFiles = [
        ...openFiles,
        {
          ...file,
          content: `// Content of ${file.name}\n// Would be loaded from API`,
        },
      ];
    }
    activeTabId = file.id;
  }

  function handleCloseTab(event: CustomEvent) {
    const id = event.detail;
    openFiles = openFiles.filter((f) => f.id !== id);
    if (activeTabId === id)
      activeTabId =
        openFiles.length > 0 ? openFiles[openFiles.length - 1].id : "";
  }
</script>

<div
  class="ide-layout"
  class:agent-panel-collapsed={agentPanelCollapsed || !showAgentPanel}
  style={`--agent-panel-width: ${agentPanelWidth}px;`}
>
  <!-- Wallpaper Background -->
  <div
    class="wallpaper-layer"
    class:is-pattern={$customWallpaper && !$customWallpaper.startsWith('url') && !$customWallpaper.includes('gradient')}
    style="background: {$customWallpaper.startsWith('url') ? `url(${$customWallpaper})` : $customWallpaper}; opacity: var(--wallpaper-visible, 1);"
  ></div>

  <TopBar on:openSettings={handleOpenSettings} />
  <StatusBand />
  <ActivityBar />
  <MainContent
    bind:this={mainContentRef}
    {openFiles}
    {activeTabId}
    on:openFile={handleOpenFile}
    on:closeTab={handleCloseTab}
  />
  {#if showAgentPanel}
    <AgentPanel
      activeCanvas={currentCanvas}
      bind:collapsed={agentPanelCollapsed}
      bind:panelWidth={agentPanelWidth}
      hidden={!showAgentPanel}
    />
  {/if}
</div>

<style>
  .ide-layout {
    display: grid;
    grid-template-areas:
      "topbar topbar topbar"
      "statusband statusband statusband"
      "activity main agent";
    grid-template-columns: var(--sidebar-width) 1fr var(--agent-panel-width, 320px);
    grid-template-rows: var(--header-height) auto 1fr;
    height: 100vh;
    width: 100vw;
    background: transparent;
    overflow: hidden;
    gap: 0;
    position: relative;
  }

  .ide-layout.agent-panel-collapsed {
    grid-template-columns: var(--sidebar-width) 1fr 0px;
    overflow: hidden;
  }

  /* Wallpaper background */
  .wallpaper-layer {
    position: fixed;
    inset: 0;
    z-index: -2;
    background: var(--wallpaper, none);
    background-position: center;
    background-repeat: no-repeat;
    background-size: cover;
  }

  .wallpaper-layer.is-pattern {
    background-size: 20px 20px;
  }

  /* Wallpaper overlay - only shown when wallpaper is active */
  .ide-layout::before {
    content: '';
    position: absolute;
    inset: 0;
    background: var(--color-bg-base);
    opacity: calc(var(--wallpaper-visible, 0) * 0.7);
    z-index: -1;
    pointer-events: none;
  }

  :global(.status-band) {
    grid-area: statusband;
  }
</style>

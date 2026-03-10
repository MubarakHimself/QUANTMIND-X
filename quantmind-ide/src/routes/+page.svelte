<script lang="ts">
  import TopBar from "$lib/components/TopBar.svelte";
  import ActivityBar from "$lib/components/ActivityBar.svelte";
  import StatusBand from "$lib/components/StatusBand.svelte";
  import MainContent from "$lib/components/MainContent.svelte";
  import BottomPanel from "$lib/components/BottomPanel.svelte";
  import { customWallpaper, wallpaperEnabled, loadSavedWallpaperEnabled, loadSavedWallpaper, setCustomWallpaper, toggleWallpaper } from "$lib/stores/themeStore";
  import { onMount } from "svelte";

  // Initialize wallpaper
  onMount(() => {
    const enabled = loadSavedWallpaperEnabled();
    toggleWallpaper(enabled);

    const wallpaper = loadSavedWallpaper();
    if (wallpaper) {
      setCustomWallpaper(wallpaper);
    }
  });

  let activeView = "live";
  let openFiles: Array<{
    id: string;
    name: string;
    content?: string;
    type?: string;
  }> = [];
  let activeTabId = "";

  function handleViewChange(event: CustomEvent) {
    const newView = event.detail.view;
    activeView = newView;
    if (newView !== "file") {
      activeTabId = ""; // Clear tab ID when switching views
    }
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

  function handleOpenSettings() {
    activeView = "settings";
    activeTabId = "";
  }
</script>

<div class="ide-layout">
  <!-- Wallpaper Background -->
  <div
    class="wallpaper-layer"
    class:is-pattern={$customWallpaper && !$customWallpaper.startsWith('url') && !$customWallpaper.includes('gradient')}
    style="background: {$customWallpaper.startsWith('url') ? `url(${$customWallpaper})` : $customWallpaper}; opacity: var(--wallpaper-visible, 1);"
  ></div>

  <TopBar on:openSettings={handleOpenSettings} />
  <StatusBand />
  <ActivityBar
    bind:activeView
    on:viewChange={handleViewChange}
  />
  <MainContent
    {activeView}
    {openFiles}
    {activeTabId}
    on:openFile={handleOpenFile}
    on:closeTab={handleCloseTab}
    on:viewChange={handleViewChange}
  />
  <BottomPanel />
</div>

<style>
  .ide-layout {
    display: grid;
    grid-template-areas:
      "topbar topbar"
      "statusband statusband"
      "activity main"
      "activity bottom";
    grid-template-columns: var(--sidebar-width) 1fr;
    grid-template-rows: var(--header-height) auto 1fr auto;
    height: 100vh;
    width: 100vw;
    background: transparent;
    overflow: hidden;
    gap: 0;
    position: relative;
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

  /* Wallpaper overlay - makes content readable */
  .ide-layout::before {
    content: '';
    position: absolute;
    inset: 0;
    background: var(--bg-primary);
    opacity: calc(0.85 - (var(--wallpaper-visible, 1) * 0.5));
    z-index: -1;
    pointer-events: none;
  }

  :global(.status-band) {
    grid-area: statusband;
  }
</style>

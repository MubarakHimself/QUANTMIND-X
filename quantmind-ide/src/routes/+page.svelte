<script lang="ts">
  import TopBar from "$lib/components/TopBar.svelte";
  import ActivityBar from "$lib/components/ActivityBar.svelte";
  import MainContent from "$lib/components/MainContent.svelte";
  import BottomPanel from "$lib/components/BottomPanel.svelte";

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
  <TopBar on:openSettings={handleOpenSettings} />
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
      "activity main"
      "activity bottom";
    grid-template-columns: var(--sidebar-width) 1fr;
    grid-template-rows: var(--header-height) 1fr auto;
    height: 100vh;
    width: 100vw;
    background: var(--bg-primary);
    overflow: hidden;
    gap: 0;
  }
</style>

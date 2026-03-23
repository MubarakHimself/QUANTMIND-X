<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import { File } from "lucide-svelte";


  interface Props {
    openFiles?: Array<{
    id: string;
    name: string;
    content?: string;
    type?: string;
    path?: string;
  }>;
    activeTabId?: string;
    activeView: string;
    viewConfig?: Record<string, any>;
  }

  let {
    openFiles = [],
    activeTabId = $bindable(""),
    activeView,
    viewConfig = {}
  }: Props = $props();

  const dispatch = createEventDispatcher();

  function handleTabClick(fileId: string) {
    activeTabId = fileId;
    dispatch("tabClick", { fileId });
  }

  function handleCloseTab(e: Event, fileId: string) {
    e.stopPropagation();
    dispatch("closeTab", { fileId });
  }

  const SvelteComponent = $derived(viewConfig[activeView]?.icon || File);
</script>

<div class="tab-bar">
  <button
    class="tab"
    class:active={!activeTabId || activeTabId === ""}
    onclick={() => {
      activeTabId = "";
      dispatch("resetNavigation");
    }}
  >
    <SvelteComponent
      size={14}
    />
    <span>{viewConfig[activeView]?.title || "Explorer"}</span>
  </button>

  {#each openFiles as file}
    <div
      class="tab"
      class:active={activeTabId === file.id}
      role="button"
      tabindex="0"
      onclick={() => handleTabClick(file.id)}
      onkeydown={(e) => e.key === 'Enter' && handleTabClick(file.id)}
    >
      <File size={14} />
      <span>{file.name}</span>
      <button
        class="tab-close"
        onclick={(e) => handleCloseTab(e, file.id)}
      >×</button
      >
    </div>
  {/each}
</div>

<style>
  .tab-bar {
    display: flex;
    align-items: center;
    height: 35px;
    background: var(--color-bg-surface);
    border-bottom: 1px solid var(--color-border-subtle);
    overflow-x: auto;
  }

  .tab {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 0 12px;
    height: 100%;
    background: transparent;
    border: none;
    border-right: 1px solid var(--color-border-subtle);
    color: var(--color-text-secondary);
    font-size: 12px;
    cursor: pointer;
    white-space: nowrap;
  }

  .tab:hover {
    background: var(--color-bg-elevated);
  }

  .tab.active {
    background: var(--color-bg-base);
    color: var(--color-text-primary);
  }

  .tab-close {
    margin-left: 4px;
    padding: 0 4px;
    background: transparent;
    border: none;
    color: var(--color-text-muted);
    cursor: pointer;
    border-radius: 4px;
  }
</style>

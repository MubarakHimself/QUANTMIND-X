<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import { Folder, ArrowLeft, Database } from "lucide-svelte";

  const dispatch = createEventDispatcher();

  export let currentFolder: string | null = null;
  export let assetTabs: Array<{
    id: string;
    name: string;
    icon?: any;
  }> = [];

  function openSubPage(page: string) {
    dispatch("openSubPage", page);
  }
</script>

<div class="assets-view">
  <div class="tab-nav">
    {#each assetTabs as tab}
      <button
        class="tab-item"
        class:active={currentFolder === tab.id ||
          (!currentFolder && tab.id === "indicators")}
        on:click={() =>
          tab.id === "database"
            ? openSubPage("database")
            : (currentFolder = tab.id)}
      >
        {#if tab.icon}
          <svelte:component this={tab.icon} size={14} />
        {/if}
        {tab.name}
      </button>
    {/each}
  </div>

  <div class="asset-view">
    {#if !currentFolder}
      <div class="asset-grid">
        {#each assetTabs as tab}
          <div
            class="folder-item"
            role="button"
            tabindex="0"
            on:click={() => (currentFolder = tab.id)}
            on:keydown={(e) =>
              e.key === "Enter" && (currentFolder = tab.id)}
          >
            <Folder size={40} class="text-accent" />
            <span>{tab.name}</span>
          </div>
        {/each}
        <div
          class="folder-item"
          role="button"
          tabindex="0"
          on:click={() => openSubPage("database")}
          on:keydown={(e) =>
            e.key === "Enter" && openSubPage("database")}
        >
          <Database size={40} class="text-accent" />
          <span>Database</span>
        </div>
      </div>
    {:else}
      <!-- File List for Selected Folder -->
      <div class="file-list">
        <div class="list-header">
          <button
            class="back-link"
            on:click={() => (currentFolder = null)}
          >
            <ArrowLeft size={14} /> Back to Assets
          </button>
          <h4>
            {currentFolder}
          </h4>
        </div>

        <div class="folder-grid">
          <p class="empty-msg">No files found in {currentFolder}</p>
        </div>
      </div>
    {/if}
  </div>
</div>

<style>
  .assets-view {
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  .tab-nav {
    display: flex;
    gap: 4px;
    padding: 12px 16px;
    background: var(--bg-secondary, #f9fafb);
    border-bottom: 1px solid var(--border-subtle, #e5e7eb);
  }

  .tab-item {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--text-secondary, #6b7280);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .tab-item:hover {
    background: var(--bg-tertiary, #f3f4f6);
    color: var(--text-primary, #111827);
  }

  .tab-item.active {
    background: var(--accent-primary, #3b82f6);
    color: white;
  }

  .asset-view {
    flex: 1;
    padding: 20px;
    overflow-y: auto;
  }

  .asset-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 20px;
  }

  .folder-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 32px 20px;
    background: var(--bg-secondary, #f9fafb);
    border: 1px solid var(--border-subtle, #e5e7eb);
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .folder-item:hover {
    background: var(--bg-tertiary, #f3f4f6);
    border-color: var(--accent-primary, #3b82f6);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }

  .folder-item :global(.text-accent) {
    color: var(--accent-primary, #3b82f6);
  }

  .folder-item span {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary, #4b5563);
    text-align: center;
  }

  .file-list {
    height: 100%;
  }

  .list-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 20px;
  }

  .back-link {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: transparent;
    border: none;
    color: var(--accent-primary, #3b82f6);
    font-size: 13px;
    cursor: pointer;
    border-radius: 4px;
  }

  .back-link:hover {
    background: var(--bg-tertiary, #f3f4f6);
  }

  .list-header h4 {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary, #111827);
    margin: 0;
  }

  .folder-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 16px;
  }

  .empty-msg {
    grid-column: 1 / -1;
    text-align: center;
    color: var(--text-muted, #9ca3af);
    padding: 40px;
    font-size: 14px;
  }
</style>

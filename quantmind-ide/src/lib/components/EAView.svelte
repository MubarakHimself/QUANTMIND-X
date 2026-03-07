<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import {
    Plus,
    Folder,
    Home,
    ChevronRight,
  } from "lucide-svelte";

  const dispatch = createEventDispatcher();

  export let videoIngestQueue: Array<{
    id: string;
    name: string;
    status: string;
    progress: number;
  }> = [];

  export let currentFolder = "";
  export let viewMode: "grid" | "list" = "grid";

  function openSubPage(page: string) {
    dispatch("openSubPage", page);
  }
</script>

<div class="ea-view">
  {#if videoIngestQueue.length > 0}
    <div class="queue-banner">
      <span class="queue-icon">⏱</span>
      {videoIngestQueue.length} Video Ingest job(s) in progress
    </div>
  {/if}

  <div class="ea-cards" class:list-view={viewMode === "list"}>
    <p class="empty-msg">No strategies available</p>
    <div
      class="ea-card add-new"
      role="button"
      tabindex="0"
      on:click={() => openSubPage("video-ingest-modal")}
      on:keydown={(e) => e.key === "Enter" && openSubPage("video-ingest-modal")}
    >
      <Plus size={32} />
      <span>Video Ingest</span>
    </div>
  </div>

  {#if currentFolder}
    <!-- Inside a strategy folder - Windows Explorer style -->
    <div class="folder-contents">
      <!-- Breadcrumb navigation -->
      <div class="folder-breadcrumb">
        <button
          class="breadcrumb-btn"
          on:click={() => (currentFolder = "")}
        >
          <Home size={14} />
          <span>EA Management</span>
        </button>
        <ChevronRight size={12} class="breadcrumb-separator" />
        <span class="breadcrumb-current">{currentFolder}</span>
      </div>

      <div class="folder-grid">
        {#if currentFolder === "video_ingest"}
          <!-- Video Ingest Output files -->
          <p class="empty-msg">No VideoIngest files found</p>
        {:else if currentFolder === "trd"}
          <!-- TRD files -->
          <p class="empty-msg">No TRD files found</p>
        {:else if currentFolder === "ea"}
          <!-- EA Code files -->
          <p class="empty-msg">No EA files found</p>
        {:else if currentFolder === "backtest"}
          <!-- Backtest Report files -->
          <p class="empty-msg">No backtest files found</p>
        {:else}
          <!-- Default subfolders -->
          <div
            class="folder-item"
            role="button"
            tabindex="0"
            on:click={() => (currentFolder = "video_ingest")}
            on:keydown={(e) =>
              e.key === "Enter" && (currentFolder = "video_ingest")}
          >
            <Folder size={40} color="#f59e0b" />
            <span>Video Ingest Output</span>
          </div>
          <div
            class="folder-item"
            role="button"
            tabindex="0"
            on:click={() => (currentFolder = "trd")}
            on:keydown={(e) =>
              e.key === "Enter" && (currentFolder = "trd")}
          >
            <Folder size={40} color="#3b82f6" />
            <span>TRD</span>
          </div>
          <div
            class="folder-item"
            role="button"
            tabindex="0"
            on:click={() => (currentFolder = "ea")}
            on:keydown={(e) =>
              e.key === "Enter" && (currentFolder = "ea")}
          >
            <Folder size={40} color="#10b981" />
            <span>EA Code</span>
          </div>
          <div
            class="folder-item"
            role="button"
            tabindex="0"
            on:click={() => (currentFolder = "backtest")}
            on:keydown={(e) =>
              e.key === "Enter" && (currentFolder = "backtest")}
          >
            <Folder size={40} color="#8b5cf6" />
            <span>Backtest Reports</span>
          </div>
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
  .ea-view {
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  .queue-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 6px;
    color: #f59e0b;
    font-size: 13px;
    margin-bottom: 16px;
  }

  .queue-icon {
    font-size: 14px;
  }

  .ea-cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 16px;
    padding: 16px 0;
  }

  .ea-cards.list-view {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .empty-msg {
    grid-column: 1 / -1;
    text-align: center;
    color: var(--text-muted, #6b7280);
    padding: 40px;
    font-size: 14px;
  }

  .ea-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 24px;
    background: var(--bg-secondary, #f3f4f6);
    border: 2px dashed var(--border-subtle, #d1d5db);
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
    color: var(--text-secondary, #6b7280);
  }

  .ea-card:hover {
    border-color: var(--accent-primary, #3b82f6);
    background: var(--bg-tertiary, #e5e7eb);
  }

  .ea-card.add-new {
    border-style: dashed;
    min-height: 140px;
  }

  .ea-card span {
    font-size: 13px;
    font-weight: 500;
  }

  .folder-contents {
    margin-top: 20px;
    padding-top: 20px;
    border-top: 1px solid var(--border-subtle, #e5e7eb);
  }

  .folder-breadcrumb {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
    padding: 8px 12px;
    background: var(--bg-glass, rgba(255, 255, 255, 0.05));
    border: 1px solid var(--border-subtle, #e5e7eb);
    border-radius: 8px;
  }

  .breadcrumb-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 8px;
    background: transparent;
    border: none;
    color: var(--text-secondary, #6b7280);
    cursor: pointer;
    font-size: 13px;
    border-radius: 4px;
  }

  .breadcrumb-btn:hover {
    background: var(--bg-tertiary, #e5e7eb);
    color: var(--text-primary, #111827);
  }

  .breadcrumb-current {
    color: var(--text-primary, #111827);
    font-weight: 500;
    font-size: 13px;
  }

  .folder-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 16px;
  }

  .folder-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 20px;
    background: var(--bg-secondary, #f9fafb);
    border: 1px solid var(--border-subtle, #e5e7eb);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .folder-item:hover {
    background: var(--bg-tertiary, #f3f4f6);
    border-color: var(--accent-primary, #3b82f6);
    transform: translateY(-2px);
  }

  .folder-item span {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary, #4b5563);
    text-align: center;
  }
</style>

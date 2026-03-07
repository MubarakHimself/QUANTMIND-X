<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { X } from 'lucide-svelte';

  export let isOpen = false;
  export let logs: string[] = [];
  export let isLoading = false;

  const dispatch = createEventDispatcher();
</script>

{#if isOpen}
  <div class="modal-overlay" on:click={() => dispatch('close')} on:keydown={(e) => e.key === 'Escape' && dispatch('close')} role="dialog" aria-modal="true">
    <div class="modal-content" on:click|stopPropagation role="presentation">
      <div class="modal-header">
        <h3>Agent Logs</h3>
        <button class="close-btn" on:click={() => dispatch('close')}>
          <X size={18} />
        </button>
      </div>
      <div class="modal-body">
        {#if isLoading}
          <div class="loading-state">Loading logs...</div>
        {:else if logs.length === 0}
          <div class="empty-state">
            <div class="empty-icon">📋</div>
            <p>No logs available for this agent</p>
          </div>
        {:else}
          <div class="logs-container">
            {#each logs as log}
              <div class="log-line">{log}</div>
            {/each}
          </div>
        {/if}
      </div>
    </div>
  </div>
{/if}

<style>
  .modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .modal-content {
    background: #1e293b;
    border-radius: 12px;
    width: 90%;
    max-width: 800px;
    max-height: 80vh;
    display: flex;
    flex-direction: column;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid #334155;
  }

  .modal-header h3 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
    color: #e2e8f0;
  }

  .close-btn {
    background: none;
    border: none;
    color: #94a3b8;
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
  }

  .close-btn:hover {
    background: #334155;
    color: #e2e8f0;
  }

  .modal-body {
    padding: 16px 20px;
    overflow-y: auto;
    flex: 1;
  }

  .loading-state,
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
    color: #64748b;
  }

  .empty-icon {
    font-size: 48px;
    margin-bottom: 12px;
  }

  .logs-container {
    background: #0f172a;
    border-radius: 8px;
    padding: 12px;
    max-height: 400px;
    overflow-y: auto;
    font-family: monospace;
    font-size: 12px;
    line-height: 1.6;
  }

  .log-line {
    color: #94a3b8;
    padding: 2px 0;
    white-space: pre-wrap;
    word-break: break-all;
  }
</style>

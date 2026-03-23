<script lang="ts">
  import { X, Plus } from "lucide-svelte";


  interface Props {
    open?: boolean;
    selectedTable?: any;
    newRowData?: Record<string, any>;
    getColumnTypeColor: (type: string) => string;
    insertRow: () => Promise<void>;
  }

  let {
    open = $bindable(false),
    selectedTable = null,
    newRowData = $bindable({}),
    getColumnTypeColor,
    insertRow
  }: Props = $props();

  function close() {
    open = false;
  }

  function handleOverlayClick(e: MouseEvent) {
    if (e.target === e.currentTarget) {
      close();
    }
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === "Escape") {
      close();
    }
  }
</script>

{#if open && selectedTable}
  <div
    class="modal-overlay"
    onclick={handleOverlayClick}
    onkeydown={handleKeyDown}
    role="button"
    tabindex="0"
  >
    <div class="modal">
      <div class="modal-header">
        <div>
          <h3>Insert Row</h3>
          <p class="modal-subtitle">{selectedTable.name}</p>
        </div>
        <button class="icon-btn" onclick={close}>
          <X size={18} />
        </button>
      </div>

      <div class="modal-content">
        {#each selectedTable.columns as column}
          {#if !column.primary_key}
            <div class="form-group">
              <label for="insert-{column.name}">
                {column.name}
                <span
                  class="type-badge"
                  style="color: {getColumnTypeColor(column.type)}"
                  >{column.type}</span
                >
              </label>
              {#if column.type.includes("INT") || column.type.includes("REAL")}
                <input
                  type="number"
                  id="insert-{column.name}"
                  bind:value={newRowData[column.name]}
                  placeholder={column.name}
                />
              {:else if column.type.includes("TEXT")}
                <textarea
                  id="insert-{column.name}"
                  bind:value={newRowData[column.name]}
                  placeholder={column.name}
                  rows="2"
                ></textarea>
              {:else}
                <input
                  type="text"
                  id="insert-{column.name}"
                  bind:value={newRowData[column.name]}
                  placeholder={column.name}
                />
              {/if}
            </div>
          {/if}
        {/each}

        <div class="modal-actions">
          <button class="btn" onclick={close}>Cancel</button>
          <button class="btn primary" onclick={insertRow}>
            <Plus size={14} />
            <span>Insert Row</span>
          </button>
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .modal {
    background: var(--surface-1);
    border-radius: 8px;
    width: 90%;
    max-width: 500px;
    max-height: 80vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem;
    border-bottom: 1px solid var(--border-color);
  }

  .modal-header h3 {
    margin: 0;
    font-size: 1.1rem;
  }

  .modal-subtitle {
    margin: 0.25rem 0 0;
    font-size: 0.8rem;
    color: var(--color-text-muted);
  }

  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: transparent;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    color: var(--color-text-muted);
  }

  .icon-btn:hover {
    background: var(--surface-3);
    color: var(--text-color);
  }

  .modal-content {
    padding: 1rem;
    overflow-y: auto;
  }

  .form-group {
    margin-bottom: 1rem;
  }

  .form-group label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 0.35rem;
  }

  .type-badge {
    font-size: 0.7rem;
    font-family: monospace;
  }

  .form-group input,
  .form-group textarea {
    width: 100%;
    padding: 0.5rem;
    background: var(--surface-2);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    font-size: 0.85rem;
  }

  .form-group input:focus,
  .form-group textarea:focus {
    outline: none;
    border-color: var(--primary-color);
  }

  .modal-actions {
    display: flex;
    justify-content: flex-end;
    gap: 0.5rem;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border-color);
  }

  .btn {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.5rem 1rem;
    font-size: 0.85rem;
    background: var(--surface-3);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .btn.primary {
    background: var(--primary-color);
    color: white;
    border-color: var(--primary-color);
  }

  .btn:hover {
    background: var(--surface-4);
  }
</style>

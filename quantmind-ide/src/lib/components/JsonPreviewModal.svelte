<script lang="ts">
  import { X } from "lucide-svelte";

  interface Props {
    open?: boolean;
    jsonData?: any;
  }

  let { open = $bindable(false), jsonData = null }: Props = $props();

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

{#if open}
  <div
    class="modal-overlay"
    onclick={handleOverlayClick}
    onkeydown={handleKeyDown}
    role="button"
    tabindex="0"
  >
    <div class="modal large">
      <div class="modal-header">
        <div>
          <h3>JSON Preview</h3>
          <p class="modal-subtitle">View complex data structure</p>
        </div>
        <button class="icon-btn" onclick={close}>
          <X size={18} />
        </button>
      </div>

      <div class="modal-content">
        <pre class="json-preview">{JSON.stringify(
            jsonData,
            null,
            2,
          )}</pre>
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

  .modal.large {
    max-width: 800px;
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
    color: var(--text-muted);
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
    color: var(--text-muted);
  }

  .icon-btn:hover {
    background: var(--surface-3);
    color: var(--text-color);
  }

  .modal-content {
    padding: 1rem;
    overflow: auto;
  }

  .json-preview {
    margin: 0;
    padding: 1rem;
    background: var(--surface-2);
    border-radius: 4px;
    font-family: monospace;
    font-size: 0.8rem;
    white-space: pre-wrap;
    word-break: break-all;
    max-height: 60vh;
    overflow: auto;
  }
</style>

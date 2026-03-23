<script lang="ts">
  /**
   * FilePreviewOverlay — overlay for agent-surfaced file references.
   * Dismiss on Escape or click-outside.
   * Story 12-3
   */
  import { FileText, X } from 'lucide-svelte';

  interface Props {
    open?: boolean;
    fileName?: string;
    content?: string;
    onClose?: () => void;
  }

  let { open = false, fileName = '', content = '', onClose }: Props = $props();

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') onClose?.();
  }
</script>

{#if open}
  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <div
    class="overlay-backdrop"
    onclick={onClose}
    onkeydown={handleKeydown}
    role="presentation"
  >
    <!-- svelte-ignore a11y-no-noninteractive-element-interactions -->
    <div
      class="overlay-panel"
      role="dialog"
      aria-modal="true"
      aria-label="File preview"
      onclick={(e) => e.stopPropagation()}
      onkeydown={(e) => e.stopPropagation()}
    >
      <div class="overlay-header">
        <div class="file-label">
          <FileText size={14} />
          <span class="file-name">{fileName}</span>
        </div>
        <button class="overlay-close" onclick={onClose} aria-label="Close preview">
          <X size={16} />
        </button>
      </div>
      <div class="overlay-body">
        <pre class="file-content">{content}</pre>
      </div>
    </div>
  </div>
{/if}

<style>
  .overlay-backdrop {
    position: fixed;
    inset: 0;
    z-index: 2000;
    background: rgba(0, 0, 0, 0.7);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: var(--space-6);
  }

  .overlay-panel {
    background: var(--glass-content-bg);
    backdrop-filter: var(--glass-blur);
    -webkit-backdrop-filter: var(--glass-blur);
    border: 1px solid var(--color-border-subtle);
    border-radius: 10px;
    width: 100%;
    max-width: 700px;
    max-height: 80vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .overlay-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-3) var(--space-4);
    border-bottom: 1px solid var(--color-border-subtle);
    flex-shrink: 0;
  }

  .file-label {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    color: var(--color-text-secondary);
  }

  .file-name {
    font-family: var(--font-data);
    font-size: var(--text-sm);
    color: var(--color-text-primary);
  }

  .overlay-close {
    background: none;
    border: none;
    cursor: pointer;
    color: var(--color-text-muted);
    display: flex;
    align-items: center;
    transition: color 0.15s ease;
  }

  .overlay-close:hover {
    color: var(--color-text-primary);
  }

  .overlay-body {
    overflow-y: auto;
    flex: 1;
    padding: var(--space-4);
  }

  .file-content {
    font-family: var(--font-data);
    font-size: var(--text-xs);
    color: var(--color-text-secondary);
    white-space: pre-wrap;
    word-break: break-word;
    margin: 0;
    line-height: 1.6;
  }
</style>

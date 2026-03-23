<script lang="ts">
  /**
   * ConfirmModal — destructive action confirmation modal.
   * Uses CheckCircle + X Lucide icons; glass tier 2 background.
   * Story 12-3
   */
  import { CheckCircle, X } from 'lucide-svelte';

  interface Props {
    open?: boolean;
    title?: string;
    message?: string;
    confirmLabel?: string;
    cancelLabel?: string;
    onConfirm?: () => void;
    onCancel?: () => void;
  }

  let {
    open = false,
    title = 'Confirm Action',
    message = 'Are you sure you want to proceed?',
    confirmLabel = 'Confirm',
    cancelLabel = 'Cancel',
    onConfirm,
    onCancel,
  }: Props = $props();

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') onCancel?.();
  }
</script>

{#if open}
  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <div
    class="modal-backdrop"
    onclick={onCancel}
    onkeydown={handleKeydown}
    role="presentation"
  >
    <!-- svelte-ignore a11y-no-noninteractive-element-interactions -->
    <div
      class="modal-panel"
      role="dialog"
      aria-modal="true"
      aria-labelledby="confirm-modal-title"
      onclick={(e) => e.stopPropagation()}
      onkeydown={(e) => e.stopPropagation()}
    >
      <div class="modal-header">
        <h2 id="confirm-modal-title" class="modal-title">{title}</h2>
        <button class="modal-close" onclick={onCancel} aria-label="Close modal">
          <X size={16} />
        </button>
      </div>
      <div class="modal-body">
        <p class="modal-message">{message}</p>
      </div>
      <div class="modal-footer">
        <button class="btn-cancel" onclick={onCancel}>
          {cancelLabel}
        </button>
        <button class="btn-confirm" onclick={onConfirm}>
          <CheckCircle size={14} />
          <span>{confirmLabel}</span>
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .modal-backdrop {
    position: fixed;
    inset: 0;
    z-index: 2000;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .modal-panel {
    background: var(--glass-content-bg);
    backdrop-filter: var(--glass-blur);
    -webkit-backdrop-filter: var(--glass-blur);
    border: 1px solid var(--color-border-subtle);
    border-radius: 10px;
    padding: var(--space-5);
    width: 400px;
    max-width: calc(100vw - var(--space-8));
  }

  .modal-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: var(--space-4);
  }

  .modal-title {
    font-family: var(--font-heading);
    font-weight: 700;
    font-size: var(--text-lg);
    color: var(--color-text-primary);
    margin: 0;
  }

  .modal-close {
    background: none;
    border: none;
    cursor: pointer;
    color: var(--color-text-muted);
    display: flex;
    align-items: center;
    transition: color 0.15s ease;
  }

  .modal-close:hover {
    color: var(--color-text-primary);
  }

  .modal-body {
    margin-bottom: var(--space-5);
  }

  .modal-message {
    font-family: var(--font-body);
    font-size: var(--text-sm);
    color: var(--color-text-secondary);
    margin: 0;
    line-height: 1.5;
  }

  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: var(--space-3);
  }

  .btn-cancel {
    background: none;
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    padding: var(--space-2) var(--space-4);
    cursor: pointer;
    color: var(--color-text-secondary);
    font-family: var(--font-body);
    font-size: var(--text-sm);
    transition: border-color 0.15s ease, color 0.15s ease;
  }

  .btn-cancel:hover {
    border-color: rgba(255, 255, 255, 0.2);
    color: var(--color-text-primary);
  }

  .btn-confirm {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    background: var(--color-accent-cyan);
    border: none;
    border-radius: 6px;
    padding: var(--space-2) var(--space-4);
    cursor: pointer;
    color: var(--color-bg-base);
    font-family: var(--font-body);
    font-size: var(--text-sm);
    font-weight: 600;
    transition: opacity 0.15s ease;
  }

  .btn-confirm:hover {
    opacity: 0.85;
  }
</style>

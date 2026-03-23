<script lang="ts">
  /**
   * WorkflowKillSwitchModal Component
   *
   * Two-step confirmation modal for workflow cancellation.
   * Only cancels the specified workflow - all other workflows and live trading unaffected.
   */

  import type { PrefectWorkflow } from '$lib/stores/flowforge';
  import { Square, AlertTriangle, X } from 'lucide-svelte';

  interface Props {
    workflow: PrefectWorkflow;
    onConfirm: () => void;
    onCancel: () => void;
  }

  let { workflow, onConfirm, onCancel }: Props = $props();

  let step = $state<1 | 2>(1);
  let confirmedName = $state('');

  // Reset state when workflow changes
  $effect(() => {
    step = 1;
    confirmedName = '';
  });

  function handleStep1() {
    step = 2;
  }

  function handleStep2() {
    if (confirmedName.toLowerCase() === workflow.name.toLowerCase()) {
      onConfirm();
    }
  }

  function handleClose() {
    step = 1;
    confirmedName = '';
    onCancel();
  }

  // Check if name matches
  const isNameConfirmed = $derived(
    confirmedName.toLowerCase() === workflow.name.toLowerCase()
  );
</script>

<div class="modal-overlay" onclick={handleClose} role="dialog" aria-modal="true">
  <div class="modal-content" onclick={(e) => e.stopPropagation()}>
    <!-- Header -->
    <div class="modal-header">
      <div class="header-icon">
        {#if step === 1}
          <AlertTriangle size={24} />
        {:else}
          <Square size={24} />
        {/if}
      </div>
      <div class="header-text">
        {#if step === 1}
          <h3>Stop Workflow?</h3>
        {:else}
          <h3>Confirm Cancellation</h3>
        {/if}
      </div>
      <button class="close-btn" onclick={handleClose} aria-label="Close">
        <X size={20} />
      </button>
    </div>

    <!-- Body -->
    <div class="modal-body">
      {#if step === 1}
        <!-- Step 1: Warning -->
        <div class="warning-content">
          <p class="workflow-name">{workflow.name}</p>
          <p class="warning-text">
            This will immediately stop the workflow and all its running tasks.
            Other workflows and live trading will continue unaffected.
          </p>
          <div class="workflow-info">
            <div class="info-row">
              <span class="label">Department:</span>
              <span class="value">{workflow.department}</span>
            </div>
            <div class="info-row">
              <span class="label">Progress:</span>
              <span class="value">{workflow.completed_steps}/{workflow.total_steps} steps</span>
            </div>
            <div class="info-row">
              <span class="label">Next Step:</span>
              <span class="value">{workflow.next_step}</span>
            </div>
          </div>
        </div>
      {:else}
        <!-- Step 2: Type confirmation -->
        <div class="confirmation-content">
          <p class="instruction">
            Type <strong>{workflow.name}</strong> to confirm cancellation:
          </p>
          <input
            type="text"
            class="confirmation-input"
            placeholder="Enter workflow name"
            bind:value={confirmedName}
            onkeydown={(e) => e.key === 'Enter' && isNameConfirmed && handleStep2()}
          />
          <p class="warning-small">
            This action cannot be undone. Use /resume-workflow to restart from last completed step.
          </p>
        </div>
      {/if}
    </div>

    <!-- Footer -->
    <div class="modal-footer">
      {#if step === 1}
        <button class="btn btn-secondary" onclick={handleClose}>Cancel</button>
        <button class="btn btn-danger" onclick={handleStep1}>Stop Workflow</button>
      {:else}
        <button class="btn btn-secondary" onclick={() => (step = 1)}>Back</button>
        <button
          class="btn btn-danger"
          onclick={handleStep2}
          disabled={!isNameConfirmed}
        >
          Confirm Stop
        </button>
      {/if}
    </div>
  </div>
</div>

<style>
  .modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .modal-content {
    background: rgba(30, 32, 40, 0.95);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    width: 100%;
    max-width: 440px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
  }

  .modal-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 20px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  }

  .header-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    background: rgba(239, 68, 68, 0.15);
    border-radius: 8px;
    color: #ef4444;
  }

  .header-text h3 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
    color: #f1f5f9;
  }

  .close-btn {
    margin-left: auto;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: transparent;
    border: none;
    color: #94a3b8;
    cursor: pointer;
    border-radius: 6px;
    transition: all 0.2s;
  }

  .close-btn:hover {
    background: rgba(255, 255, 255, 0.1);
    color: #f1f5f9;
  }

  .modal-body {
    padding: 20px;
  }

  .warning-content .workflow-name {
    font-size: 16px;
    font-weight: 600;
    color: #f1f5f9;
    margin: 0 0 12px 0;
  }

  .warning-text {
    color: #cbd5e1;
    font-size: 14px;
    line-height: 1.5;
    margin: 0 0 16px 0;
  }

  .workflow-info {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 8px;
    padding: 12px;
  }

  .info-row {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    padding: 4px 0;
  }

  .info-row .label {
    color: #94a3b8;
  }

  .info-row .value {
    color: #cbd5e1;
  }

  .confirmation-content .instruction {
    color: #cbd5e1;
    font-size: 14px;
    margin: 0 0 12px 0;
  }

  .confirmation-content strong {
    color: #f1f5f9;
  }

  .confirmation-input {
    width: 100%;
    padding: 10px 12px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 6px;
    color: #f1f5f9;
    font-size: 14px;
    outline: none;
    transition: all 0.2s;
  }

  .confirmation-input:focus {
    border-color: #ef4444;
    box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.2);
  }

  .confirmation-input::placeholder {
    color: #64748b;
  }

  .warning-small {
    color: #64748b;
    font-size: 12px;
    margin: 12px 0 0 0;
  }

  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    padding: 16px 20px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
  }

  .btn {
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    border: none;
  }

  .btn-secondary {
    background: rgba(255, 255, 255, 0.08);
    color: #cbd5e1;
  }

  .btn-secondary:hover {
    background: rgba(255, 255, 255, 0.12);
  }

  .btn-danger {
    background: #ef4444;
    color: white;
  }

  .btn-danger:hover:not(:disabled) {
    background: #dc2626;
  }

  .btn-danger:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
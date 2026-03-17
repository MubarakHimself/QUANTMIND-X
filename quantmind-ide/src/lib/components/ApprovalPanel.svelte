<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Check, X, Clock, AlertCircle, Workflow, ChevronDown, ChevronUp } from 'lucide-svelte';
  import {
    pendingApprovals,
    approvalHistory,
    selectedApproval,
    hasPendingApprovals,
    pendingCount,
    approvalLoading,
    approvalError,
    approvalStore,
    type ApprovalGate,
    type ApprovalActionRequest,
  } from '$lib/stores/approvalStore';

  // Local state
  let showPanel = $state(false);
  let showHistory = $state(false);
  let approverName = $state('User');
  let notes = $state('');
  let actionInProgress = $state(false);
  let expandedGates: Set<string> = $state(new Set());

  // Initialize
  onMount(() => {
    approvalStore.startPolling(10000);
  });

  onDestroy(() => {
    approvalStore.stopPolling();
  });

  function togglePanel() {
    showPanel = !showPanel;
    if (showPanel) {
      showHistory = false;
    }
  }

  function toggleHistory() {
    showHistory = !showHistory;
  }

  function toggleGateExpand(gateId: string) {
    if (expandedGates.has(gateId)) {
      expandedGates.delete(gateId);
    } else {
      expandedGates.add(gateId);
    }
    expandedGates = expandedGates;
  }

  async function handleApprove(gate: ApprovalGate) {
    actionInProgress = true;
    const request: ApprovalActionRequest = {
      approver: approverName,
      notes: notes || 'Approved',
    };

    const result = await approvalStore.approveGate(gate.gate_id, request);
    if (result) {
      notes = '';
    }
    actionInProgress = false;
  }

  async function handleReject(gate: ApprovalGate) {
    if (!notes.trim()) {
      alert('Please provide a reason for rejection');
      return;
    }

    actionInProgress = true;
    const request: ApprovalActionRequest = {
      approver: approverName,
      notes: notes,
    };

    const result = await approvalStore.rejectGate(gate.gate_id, request);
    if (result) {
      notes = '';
    }
    actionInProgress = false;
  }

  function formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    return date.toLocaleString();
  }

  function getGateTypeLabel(gateType: string): string {
    const labels: Record<string, string> = {
      stage_transition: 'Stage Transition',
      deployment: 'Deployment',
      risk_check: 'Risk Check',
      manual_review: 'Manual Review',
    };
    return labels[gateType] || gateType;
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'pending':
        return 'text-yellow-500';
      case 'approved':
        return 'text-green-500';
      case 'rejected':
        return 'text-red-500';
      default:
        return 'text-gray-500';
    }
  }
</script>

<div class="approval-panel">
  <!-- Toggle Button -->
  <button
    class="approval-toggle"
    class:has-approvals={$hasPendingApprovals}
    onclick={togglePanel}
    title="Workflow Approvals"
  >
    <Workflow size={18} />
    {#if $pendingCount > 0}
      <span class="badge">{$pendingCount}</span>
    {/if}
  </button>

  <!-- Panel -->
  {#if showPanel}
    <div class="panel">
      <div class="panel-header">
        <h3>Approval Gates</h3>
        <button class="close-btn" onclick={togglePanel}>
          <X size={16} />
        </button>
      </div>

      {#if $approvalError}
        <div class="error-message">
          <AlertCircle size={16} />
          <span>{$approvalError}</span>
          <button onclick={() => approvalStore.clearError()}>Dismiss</button>
        </div>
      {/if}

      <!-- Approver Input -->
      <div class="approver-input">
        <label for="approver">Approver Name:</label>
        <input
          id="approver"
          type="text"
          bind:value={approverName}
          placeholder="Enter your name"
        />
      </div>

      <!-- Pending Approvals -->
      {#if $pendingApprovals.length > 0}
        <div class="section">
          <h4 class="section-title">
            <Clock size={14} />
            Pending Approvals ({$pendingApprovals.length})
          </h4>

          <div class="gates-list">
            {#each $pendingApprovals as gate (gate.gate_id)}
              <div class="gate-card pending">
                <button
                  class="gate-header"
                  onclick={() => toggleGateExpand(gate.gate_id)}
                >
                  <div class="gate-info">
                    <span class="gate-type">{getGateTypeLabel(gate.gate_type)}</span>
                    <span class="gate-stages">
                      {gate.from_stage} → {gate.to_stage}
                    </span>
                  </div>
                  {#if expandedGates.has(gate.gate_id)}
                    <ChevronUp size={16} />
                  {:else}
                    <ChevronDown size={16} />
                  {/if}
                </button>

                {#if expandedGates.has(gate.gate_id)}
                  <div class="gate-details">
                    <div class="detail-row">
                      <span class="label">Workflow ID:</span>
                      <span class="value code">{gate.workflow_id}</span>
                    </div>
                    <div class="detail-row">
                      <span class="label">Requested:</span>
                      <span class="value">{formatDate(gate.created_at)}</span>
                    </div>
                    <div class="detail-row">
                      <span class="label">Requester:</span>
                      <span class="value">{gate.requester || 'System'}</span>
                    </div>
                    {#if gate.reason}
                      <div class="detail-row">
                        <span class="label">Reason:</span>
                        <span class="value">{gate.reason}</span>
                      </div>
                    {/if}
                    {#if gate.extra_data}
                      <div class="detail-row">
                        <span class="label">Input File:</span>
                        <span class="value code">{gate.extra_data?.input_file || 'N/A'}</span>
                      </div>
                    {/if}

                    <!-- Notes Input -->
                    <div class="notes-input">
                      <label for="notes-{gate.gate_id}">Notes:</label>
                      <textarea
                        id="notes-{gate.gate_id}"
                        bind:value={notes}
                        placeholder="Add approval notes (required for rejection)"
                        rows="2"
                      ></textarea>
                    </div>

                    <!-- Action Buttons -->
                    <div class="action-buttons">
                      <button
                        class="btn approve"
                        onclick={() => handleApprove(gate)}
                        disabled={actionInProgress}
                      >
                        <Check size={14} />
                        Approve
                      </button>
                      <button
                        class="btn reject"
                        onclick={() => handleReject(gate)}
                        disabled={actionInProgress || !notes.trim()}
                      >
                        <X size={14} />
                        Reject
                      </button>
                    </div>
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        </div>
      {:else if $approvalLoading}
        <div class="loading">
          <div class="spinner"></div>
          <span>Loading approvals...</span>
        </div>
      {:else}
        <div class="empty-state">
          <Check size={32} />
          <p>No pending approvals</p>
        </div>
      {/if}

      <!-- History Toggle -->
      <button class="history-toggle" onclick={toggleHistory}>
        {showHistory ? 'Hide History' : 'Show History'}
        {#if $approvalHistory.length > 0}
          <span class="history-count">{$approvalHistory.length}</span>
        {/if}
      </button>

      <!-- History -->
      {#if showHistory && $approvalHistory.length > 0}
        <div class="section history">
          <h4 class="section-title">Approval History</h4>
          <div class="gates-list">
            {#each $approvalHistory as gate (gate.gate_id)}
              <div class="gate-card {gate.status}">
                <div class="gate-info">
                  <span class="gate-type">{getGateTypeLabel(gate.gate_type)}</span>
                  <span class="gate-stages">
                    {gate.from_stage} → {gate.to_stage}
                  </span>
                  <span class="gate-status {getStatusColor(gate.status)}">
                    {gate.status}
                  </span>
                </div>
                <div class="gate-meta">
                  <span>{formatDate(gate.updated_at)}</span>
                  <span>by {gate.approver || 'Unknown'}</span>
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .approval-panel {
    position: relative;
    z-index: 100;
  }

  .approval-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    color: #e2e8f0;
    cursor: pointer;
    font-size: 14px;
    transition: all 0.2s;
  }

  .approval-toggle:hover {
    background: #334155;
  }

  .approval-toggle.has-approvals {
    border-color: #f59e0b;
    background: #451a03;
  }

  .badge {
    background: #f59e0b;
    color: #000;
    padding: 2px 6px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 600;
  }

  .panel {
    position: absolute;
    top: 100%;
    right: 0;
    margin-top: 8px;
    width: 380px;
    max-height: 70vh;
    overflow-y: auto;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid #334155;
  }

  .panel-header h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    color: #f1f5f9;
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
    color: #f1f5f9;
  }

  .error-message {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    background: #450a0a;
    border-bottom: 1px solid #7f1d1d;
    color: #fecaca;
    font-size: 13px;
  }

  .error-message button {
    margin-left: auto;
    background: none;
    border: none;
    color: #fecaca;
    text-decoration: underline;
    cursor: pointer;
  }

  .approver-input {
    padding: 12px 16px;
    border-bottom: 1px solid #1e293b;
  }

  .approver-input label {
    display: block;
    font-size: 12px;
    color: #94a3b8;
    margin-bottom: 4px;
  }

  .approver-input input {
    width: 100%;
    padding: 8px 10px;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 4px;
    color: #f1f5f9;
    font-size: 13px;
  }

  .section {
    padding: 12px 16px;
  }

  .section-title {
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 0 0 12px 0;
    font-size: 13px;
    font-weight: 600;
    color: #94a3b8;
  }

  .gates-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .gate-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    overflow: hidden;
  }

  .gate-card.pending {
    border-color: #f59e0b;
  }

  .gate-card.approved {
    border-color: #22c55e;
  }

  .gate-card.rejected {
    border-color: #ef4444;
  }

  .gate-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
    padding: 10px 12px;
    background: none;
    border: none;
    color: #f1f5f9;
    cursor: pointer;
    text-align: left;
  }

  .gate-header:hover {
    background: #334155;
  }

  .gate-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .gate-type {
    font-size: 12px;
    font-weight: 600;
    color: #e2e8f0;
  }

  .gate-stages {
    font-size: 11px;
    color: #94a3b8;
  }

  .gate-status {
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
  }

  .gate-details {
    padding: 12px;
    border-top: 1px solid #334155;
    background: #0f172a;
  }

  .detail-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    font-size: 12px;
  }

  .detail-row .label {
    color: #94a3b8;
  }

  .detail-row .value {
    color: #e2e8f0;
  }

  .detail-row .value.code {
    font-family: monospace;
    font-size: 11px;
    color: #a5b4fc;
    max-width: 180px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .notes-input {
    margin: 12px 0;
  }

  .notes-input label {
    display: block;
    font-size: 12px;
    color: #94a3b8;
    margin-bottom: 4px;
  }

  .notes-input textarea {
    width: 100%;
    padding: 8px 10px;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 4px;
    color: #f1f5f9;
    font-size: 13px;
    resize: none;
  }

  .action-buttons {
    display: flex;
    gap: 8px;
    margin-top: 12px;
  }

  .btn {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 8px 12px;
    border: none;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
  }

  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn.approve {
    background: #22c55e;
    color: #fff;
  }

  .btn.approve:hover:not(:disabled) {
    background: #16a34a;
  }

  .btn.reject {
    background: #ef4444;
    color: #fff;
  }

  .btn.reject:hover:not(:disabled) {
    background: #dc2626;
  }

  .loading {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    padding: 40px 20px;
    color: #94a3b8;
  }

  .spinner {
    width: 24px;
    height: 24px;
    border: 2px solid #334155;
    border-top-color: #f59e0b;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    padding: 40px 20px;
    color: #64748b;
  }

  .empty-state p {
    margin: 0;
    font-size: 14px;
  }

  .history-toggle {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    width: 100%;
    padding: 10px;
    background: #1e293b;
    border: none;
    border-top: 1px solid #334155;
    color: #94a3b8;
    font-size: 13px;
    cursor: pointer;
  }

  .history-toggle:hover {
    background: #334155;
    color: #e2e8f0;
  }

  .history-count {
    background: #334155;
    padding: 2px 6px;
    border-radius: 10px;
    font-size: 11px;
  }

  .history .gate-card {
    padding: 10px 12px;
  }

  .gate-meta {
    display: flex;
    justify-content: space-between;
    margin-top: 6px;
    font-size: 11px;
    color: #64748b;
  }
</style>

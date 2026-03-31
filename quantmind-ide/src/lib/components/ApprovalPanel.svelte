<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Check, X, Clock, AlertCircle, Workflow, ChevronDown, ChevronUp, Shield, Zap } from 'lucide-svelte';
  import {
    pendingApprovals,
    approvalHistory,
    selectedApproval,
    hasPendingApprovals,
    pendingCount,
    approvalLoading,
    approvalError,
    approvalStore,
    type UnifiedApproval,
    type ApprovalActionRequest,
  } from '$lib/stores/approvalStore';

  // Local state
  let showPanel = $state(false);
  let showHistory = $state(false);
  let approverName = $state('User');
  let notes = $state('');
  let actionInProgress = $state(false);
  let expandedItems: Set<string> = $state(new Set());

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

  function toggleExpand(id: string) {
    if (expandedItems.has(id)) {
      expandedItems.delete(id);
    } else {
      expandedItems.add(id);
    }
    expandedItems = expandedItems;
  }

  async function handleApprove(item: UnifiedApproval) {
    actionInProgress = true;
    const request: ApprovalActionRequest = {
      approver: approverName,
      notes: notes || 'Approved',
    };

    const result = await approvalStore.approveItem(item, request);
    if (result) {
      notes = '';
    }
    actionInProgress = false;
  }

  async function handleReject(item: UnifiedApproval) {
    if (!notes.trim()) {
      alert('Please provide a reason for rejection');
      return;
    }

    actionInProgress = true;
    const request: ApprovalActionRequest = {
      approver: approverName,
      notes: notes,
    };

    const result = await approvalStore.rejectItem(item, request);
    if (result) {
      notes = '';
    }
    actionInProgress = false;
  }

  function formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    return date.toLocaleString();
  }

  function getTypeLabel(item: UnifiedApproval): string {
    const labels: Record<string, string> = {
      stage_transition: 'Stage Transition',
      deployment: 'Deployment',
      risk_check: 'Risk Check',
      manual_review: 'Manual Review',
      workflow_gate: 'Workflow Gate',
      tool_execution: 'Tool Approval',
      agent_action: 'Agent Action',
      ea_promotion: 'EA Promotion',
      trade_execution: 'Trade Execution',
    };
    return labels[item.gate_type] || item.gate_type?.replace(/_/g, ' ') || 'Approval';
  }

  function getUrgencyClass(item: UnifiedApproval): string {
    if (item.urgency === 'critical') return 'urgency-critical';
    if (item.urgency === 'high') return 'urgency-high';
    return '';
  }

  function getSourceIcon(source: string): typeof Shield | typeof Zap {
    return source === 'hitl' ? Zap : Shield;
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'pending':
        return 'text-yellow-500';
      case 'approved':
        return 'text-green-500';
      case 'rejected':
        return 'text-red-500';
      case 'expired':
      case 'cancelled':
        return 'text-gray-500';
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
        <h3>Approvals</h3>
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
            {#each $pendingApprovals as item (item.id)}
              <div class="gate-card pending {getUrgencyClass(item)}">
                <button
                  class="gate-header"
                  onclick={() => toggleExpand(item.id)}
                >
                  <div class="gate-info">
                    <div class="gate-type-row">
                      <svelte:component this={getSourceIcon(item.source)} size={12} />
                      <span class="gate-type">{getTypeLabel(item)}</span>
                      {#if item.urgency === 'critical' || item.urgency === 'high'}
                        <span class="urgency-badge {item.urgency}">{item.urgency}</span>
                      {/if}
                    </div>
                    <span class="gate-title">{item.title}</span>
                    {#if item.department}
                      <span class="gate-dept">{item.department}</span>
                    {/if}
                  </div>
                  {#if expandedItems.has(item.id)}
                    <ChevronUp size={16} />
                  {:else}
                    <ChevronDown size={16} />
                  {/if}
                </button>

                {#if expandedItems.has(item.id)}
                  <div class="gate-details">
                    <div class="detail-row">
                      <span class="label">Description:</span>
                      <span class="value">{item.description}</span>
                    </div>
                    {#if item.workflow_id}
                      <div class="detail-row">
                        <span class="label">Workflow ID:</span>
                        <span class="value code">{item.workflow_id}</span>
                      </div>
                    {/if}
                    {#if item.from_stage}
                      <div class="detail-row">
                        <span class="label">Stage:</span>
                        <span class="value">{item.from_stage}{item.to_stage ? ` → ${item.to_stage}` : ''}</span>
                      </div>
                    {/if}
                    <div class="detail-row">
                      <span class="label">Requested:</span>
                      <span class="value">{formatDate(item.created_at)}</span>
                    </div>
                    {#if item.agent_id}
                      <div class="detail-row">
                        <span class="label">Agent:</span>
                        <span class="value code">{item.agent_id}</span>
                      </div>
                    {/if}
                    <div class="detail-row">
                      <span class="label">Source:</span>
                      <span class="value">{item.source === 'hitl' ? 'Agent HITL' : 'Workflow Gate'}</span>
                    </div>

                    <!-- Notes Input -->
                    <div class="notes-input">
                      <label for="notes-{item.id}">Notes:</label>
                      <textarea
                        id="notes-{item.id}"
                        bind:value={notes}
                        placeholder="Add approval notes (required for rejection)"
                        rows="2"
                      ></textarea>
                    </div>

                    <!-- Action Buttons -->
                    <div class="action-buttons">
                      <button
                        class="btn approve"
                        onclick={() => handleApprove(item)}
                        disabled={actionInProgress}
                      >
                        <Check size={14} />
                        Approve
                      </button>
                      <button
                        class="btn reject"
                        onclick={() => handleReject(item)}
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
            {#each $approvalHistory as item (item.id)}
              <div class="gate-card {item.status}">
                <div class="gate-info" style="padding: 10px 12px;">
                  <div class="gate-type-row">
                    <svelte:component this={getSourceIcon(item.source)} size={12} />
                    <span class="gate-type">{getTypeLabel(item)}</span>
                    <span class="gate-status {getStatusColor(item.status)}">
                      {item.status}
                    </span>
                  </div>
                  <span class="gate-title">{item.title}</span>
                </div>
                <div class="gate-meta">
                  <span>{formatDate(item.resolved_at || item.updated_at || item.created_at)}</span>
                  <span>by {item.approver || 'Unknown'}</span>
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
    width: 400px;
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

  .gate-card.urgency-critical {
    border-color: #ef4444;
    box-shadow: 0 0 8px rgba(239, 68, 68, 0.3);
  }

  .gate-card.urgency-high {
    border-color: #f97316;
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

  .gate-type-row {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .gate-type {
    font-size: 12px;
    font-weight: 600;
    color: #e2e8f0;
  }

  .gate-title {
    font-size: 13px;
    color: #cbd5e1;
    line-height: 1.3;
  }

  .gate-dept {
    font-size: 11px;
    color: #64748b;
    text-transform: capitalize;
  }

  .urgency-badge {
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    padding: 1px 5px;
    border-radius: 3px;
    letter-spacing: 0.5px;
  }

  .urgency-badge.critical {
    background: #ef4444;
    color: #fff;
  }

  .urgency-badge.high {
    background: #f97316;
    color: #fff;
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
    flex-shrink: 0;
    margin-right: 8px;
  }

  .detail-row .value {
    color: #e2e8f0;
    text-align: right;
  }

  .detail-row .value.code {
    font-family: monospace;
    font-size: 11px;
    color: #a5b4fc;
    max-width: 200px;
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
    padding: 0;
  }

  .gate-meta {
    display: flex;
    justify-content: space-between;
    padding: 0 12px 10px 12px;
    font-size: 11px;
    color: #64748b;
  }
</style>

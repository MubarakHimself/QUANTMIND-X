/**
 * Unified Approval Store
 *
 * Merges two approval sources into a single unified stream:
 *   1. Legacy approval gates  → /api/approval-gates/pending  (SQLAlchemy DB)
 *   2. New HITL approvals     → /api/approvals/pending       (ApprovalManager in-memory)
 *
 * Also routes every new/resolved approval through the notification system
 * so the NotificationTray stays in sync.
 */

import { writable, derived, get } from 'svelte/store';
import type { Readable } from 'svelte/store';
import { addNotification } from '$lib/stores/notifications';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ApprovalStatus = 'pending' | 'approved' | 'rejected' | 'expired' | 'cancelled';
export type GateType =
  | 'stage_transition'
  | 'deployment'
  | 'risk_check'
  | 'manual_review'
  | 'workflow_gate'
  | 'tool_execution'
  | 'agent_action'
  | 'ea_promotion'
  | 'trade_execution';

/** Source system that originated the approval */
export type ApprovalSource = 'gate' | 'hitl';

/**
 * Unified approval item — normalises both legacy gate rows and new HITL
 * approval requests into a single shape the UI can render.
 */
export interface UnifiedApproval {
  /** Canonical ID used for approve/reject calls */
  id: string;
  source: ApprovalSource;
  workflow_id?: string;
  workflow_type?: string;
  title: string;
  description: string;
  from_stage?: string;
  to_stage?: string;
  gate_type: string;
  status: ApprovalStatus;
  urgency?: string;
  department?: string;
  agent_id?: string;
  requester?: string;
  approver?: string;
  reason?: string;
  notes?: string;
  extra_data?: Record<string, any>;
  strategy_id?: string;
  created_at: string;
  updated_at?: string;
  resolved_at?: string;
  rejection_reason?: string;
}

export interface ApprovalActionRequest {
  approver: string;
  notes?: string;
}

export interface ApprovalActionResponse {
  success: boolean;
  gate_id?: string;
  approval_id?: string;
  status: ApprovalStatus;
  message: string;
  approver?: string;
  timestamp?: string;
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

interface ApprovalState {
  pendingItems: UnifiedApproval[];
  historyItems: UnifiedApproval[];
  selectedItem: UnifiedApproval | null;
  loading: boolean;
  error: string | null;
  pollingEnabled: boolean;
}

const initialState: ApprovalState = {
  pendingItems: [],
  historyItems: [],
  selectedItem: null,
  loading: false,
  error: null,
  pollingEnabled: true,
};

const approvalState = writable<ApprovalState>(initialState);

/** Set of IDs we already pushed a "new approval" notification for */
let _notifiedNewIds = new Set<string>();

/** Set of IDs we already pushed a "resolved" notification for */
let _notifiedResolvedIds = new Set<string>();

// ---------------------------------------------------------------------------
// Derived stores (public)
// ---------------------------------------------------------------------------

export const pendingApprovals: Readable<UnifiedApproval[]> = derived(
  approvalState,
  ($s) => $s.pendingItems,
);

export const approvalHistory: Readable<UnifiedApproval[]> = derived(
  approvalState,
  ($s) => $s.historyItems,
);

export const selectedApproval: Readable<UnifiedApproval | null> = derived(
  approvalState,
  ($s) => $s.selectedItem,
);

export const hasPendingApprovals: Readable<boolean> = derived(
  approvalState,
  ($s) => $s.pendingItems.length > 0,
);

export const pendingCount: Readable<number> = derived(
  approvalState,
  ($s) => $s.pendingItems.length,
);

export const approvalLoading: Readable<boolean> = derived(
  approvalState,
  ($s) => $s.loading,
);

export const approvalError: Readable<string | null> = derived(
  approvalState,
  ($s) => $s.error,
);

// ---------------------------------------------------------------------------
// Normalisation helpers
// ---------------------------------------------------------------------------

/** Convert a legacy gate row to UnifiedApproval */
function normaliseGate(g: any): UnifiedApproval {
  return {
    id: g.gate_id,
    source: 'gate',
    workflow_id: g.workflow_id,
    workflow_type: g.workflow_type,
    title: `${g.gate_type?.replace(/_/g, ' ') ?? 'Gate'}: ${g.from_stage} → ${g.to_stage}`,
    description: g.reason || `Stage transition from ${g.from_stage} to ${g.to_stage}`,
    from_stage: g.from_stage,
    to_stage: g.to_stage,
    gate_type: g.gate_type ?? 'stage_transition',
    status: g.status ?? 'pending',
    department: g.assigned_to,
    requester: g.requester,
    approver: g.approver,
    reason: g.reason,
    notes: g.notes,
    extra_data: g.extra_data,
    strategy_id: g.strategy_id,
    created_at: g.created_at,
    updated_at: g.updated_at,
    resolved_at: g.approved_at || g.rejected_at,
  };
}

/** Convert a new HITL approval to UnifiedApproval */
function normaliseHitl(a: any): UnifiedApproval {
  return {
    id: a.id,
    source: 'hitl',
    workflow_id: a.workflow_id,
    title: a.title,
    description: a.description,
    gate_type: a.approval_type ?? 'agent_action',
    status: a.status ?? 'pending',
    urgency: a.urgency,
    department: a.department,
    agent_id: a.agent_id,
    strategy_id: a.strategy_id,
    workflow_type: a.workflow_stage,
    from_stage: a.workflow_stage,
    to_stage: a.context?.to_stage,
    extra_data: a.context,
    created_at: a.created_at,
    resolved_at: a.resolved_at,
    approver: a.resolved_by,
    rejection_reason: a.rejection_reason,
  };
}

// ---------------------------------------------------------------------------
// Notification bridge
// ---------------------------------------------------------------------------

function _urgencyToNotificationType(urgency?: string): 'info' | 'warning' | 'error' | 'agent' {
  switch (urgency) {
    case 'critical':
      return 'error';
    case 'high':
      return 'warning';
    default:
      return 'agent';
  }
}

/** Push a notification for every genuinely new approval */
function _pushNewApprovalNotifications(items: UnifiedApproval[]) {
  for (const item of items) {
    if (_notifiedNewIds.has(item.id)) continue;
    _notifiedNewIds.add(item.id);

    addNotification({
      type: _urgencyToNotificationType(item.urgency),
      title: 'Approval Required',
      body: item.title,
      canvasLink: item.department ?? undefined,
    });
  }
}

/** Push a notification when an approval gets resolved */
function _pushResolvedNotification(item: UnifiedApproval) {
  if (_notifiedResolvedIds.has(item.id)) return;
  _notifiedResolvedIds.add(item.id);

  const action = item.status === 'approved' ? 'Approved' : 'Rejected';
  addNotification({
    type: item.status === 'approved' ? 'success' : 'warning',
    title: `Approval ${action}`,
    body: item.title,
    canvasLink: item.department ?? undefined,
  });
}

// ---------------------------------------------------------------------------
// SSE connection for real-time HITL events
// ---------------------------------------------------------------------------

let _eventSource: EventSource | null = null;

function _connectSSE() {
  if (_eventSource) return;

  try {
    _eventSource = new EventSource('/api/approvals/stream');

    _eventSource.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);

        if (payload.type === 'approval_event' && payload.approval) {
          const item = normaliseHitl(payload.approval);

          if (item.status === 'pending') {
            // New pending approval — merge into state + notify
            approvalState.update((s) => {
              const exists = s.pendingItems.some((p) => p.id === item.id);
              if (exists) return s;
              const newPending = [item, ...s.pendingItems];
              _pushNewApprovalNotifications([item]);
              return { ...s, pendingItems: newPending };
            });
          } else {
            // Resolved — remove from pending, add to history, notify
            approvalState.update((s) => ({
              ...s,
              pendingItems: s.pendingItems.filter((p) => p.id !== item.id),
              historyItems: [item, ...s.historyItems].slice(0, 100),
            }));
            _pushResolvedNotification(item);
          }
        }

        if (payload.type === 'init' && Array.isArray(payload.pending)) {
          const hitlItems = payload.pending.map(normaliseHitl);
          approvalState.update((s) => {
            // Merge HITL items that aren't already present
            const existingIds = new Set(s.pendingItems.map((p) => p.id));
            const newItems = hitlItems.filter((i: UnifiedApproval) => !existingIds.has(i.id));
            if (newItems.length === 0) return s;
            _pushNewApprovalNotifications(newItems);
            return { ...s, pendingItems: [...newItems, ...s.pendingItems] };
          });
        }
      } catch {
        // Ignore parse errors on heartbeat etc
      }
    };

    _eventSource.onerror = () => {
      // Auto-reconnect is handled by EventSource; just log
      console.debug('[approvalStore] SSE connection error, will reconnect');
    };
  } catch {
    // SSE not available — fall back to polling only
  }
}

function _disconnectSSE() {
  if (_eventSource) {
    _eventSource.close();
    _eventSource = null;
  }
}

// ---------------------------------------------------------------------------
// Polling
// ---------------------------------------------------------------------------

let pollingInterval: ReturnType<typeof setInterval> | null = null;

// ---------------------------------------------------------------------------
// Public store actions
// ---------------------------------------------------------------------------

export const approvalStore = {
  subscribe: approvalState.subscribe,

  /**
   * Start polling + SSE for pending approvals from both sources
   */
  startPolling(intervalMs: number = 10000): void {
    if (pollingInterval) clearInterval(pollingInterval);

    approvalState.update((s) => ({ ...s, pollingEnabled: true }));

    // Connect SSE for real-time HITL events
    _connectSSE();

    // Initial fetch
    this.fetchAllPending();

    // Set up polling for both sources
    pollingInterval = setInterval(() => {
      const state = get(approvalState);
      if (state.pollingEnabled) {
        this.fetchAllPending();
      }
    }, intervalMs);
  },

  /**
   * Stop polling and disconnect SSE
   */
  stopPolling(): void {
    if (pollingInterval) {
      clearInterval(pollingInterval);
      pollingInterval = null;
    }
    _disconnectSSE();
    approvalState.update((s) => ({ ...s, pollingEnabled: false }));
  },

  /**
   * Fetch pending approvals from BOTH sources and merge
   */
  async fetchAllPending(): Promise<void> {
    try {
      // Fetch both sources in parallel
      const [gatesRes, hitlRes] = await Promise.allSettled([
        fetch('/api/approval-gates/pending?limit=50'),
        fetch('/api/approvals/pending'),
      ]);

      let gateItems: UnifiedApproval[] = [];
      let hitlItems: UnifiedApproval[] = [];

      if (gatesRes.status === 'fulfilled' && gatesRes.value.ok) {
        const data = await gatesRes.value.json();
        const gates = data.gates || data || [];
        gateItems = (Array.isArray(gates) ? gates : []).map(normaliseGate);
      }

      if (hitlRes.status === 'fulfilled' && hitlRes.value.ok) {
        const data = await hitlRes.value.json();
        hitlItems = (Array.isArray(data) ? data : []).map(normaliseHitl);
      }

      // Merge (deduplicate by id)
      const seen = new Set<string>();
      const merged: UnifiedApproval[] = [];
      for (const item of [...hitlItems, ...gateItems]) {
        if (!seen.has(item.id)) {
          seen.add(item.id);
          merged.push(item);
        }
      }

      // Sort newest-first
      merged.sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''));

      // Push notifications for any new items
      _pushNewApprovalNotifications(merged);

      approvalState.update((s) => ({
        ...s,
        pendingItems: merged,
        loading: false,
      }));
    } catch (err) {
      approvalState.update((s) => ({
        ...s,
        error: err instanceof Error ? err.message : 'Failed to fetch pending approvals',
        loading: false,
      }));
    }
  },

  // -- Legacy convenience aliases (kept for ApprovalPanel backward compat) --

  async fetchPendingGates(workflowId?: string): Promise<void> {
    return this.fetchAllPending();
  },

  /**
   * Fetch all gates for a specific workflow
   */
  async fetchWorkflowGates(workflowId: string): Promise<UnifiedApproval[]> {
    approvalState.update((s) => ({ ...s, loading: true }));
    try {
      const response = await fetch(`/api/approval-gates/workflow/${workflowId}`);
      if (!response.ok) throw new Error('Failed to fetch workflow approvals');

      const data = await response.json();
      const gates = (data.gates || []).map(normaliseGate);

      approvalState.update((s) => ({
        ...s,
        historyItems: gates,
        loading: false,
      }));

      return gates;
    } catch (err) {
      approvalState.update((s) => ({
        ...s,
        error: err instanceof Error ? err.message : 'Failed to fetch workflow approvals',
        loading: false,
      }));
      return [];
    }
  },

  /**
   * Approve a unified approval item — routes to correct backend
   */
  async approveItem(
    item: UnifiedApproval,
    request: ApprovalActionRequest,
  ): Promise<ApprovalActionResponse | null> {
    approvalState.update((s) => ({ ...s, loading: true }));

    try {
      let response: Response;

      if (item.source === 'hitl') {
        // New ApprovalManager endpoint
        response = await fetch(`/api/approvals/${item.id}/approve`, {
          method: 'POST',
        });
      } else {
        // Legacy gate endpoint
        response = await fetch(`/api/approval-gates/${item.id}/approve`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(request),
        });
      }

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to approve');
      }

      const result = await response.json();

      // Remove from pending, add to history
      approvalState.update((s) => ({
        ...s,
        loading: false,
        pendingItems: s.pendingItems.filter((p) => p.id !== item.id),
        historyItems: [
          { ...item, status: 'approved' as ApprovalStatus, approver: request.approver },
          ...s.historyItems,
        ].slice(0, 100),
      }));

      // Push resolved notification
      _pushResolvedNotification({ ...item, status: 'approved' });

      return {
        success: true,
        gate_id: item.source === 'gate' ? item.id : undefined,
        approval_id: item.source === 'hitl' ? item.id : undefined,
        status: 'approved',
        message: result.message || 'Approved',
        approver: request.approver,
        timestamp: new Date().toISOString(),
      };
    } catch (err) {
      approvalState.update((s) => ({
        ...s,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to approve',
      }));
      return null;
    }
  },

  /**
   * Reject a unified approval item — routes to correct backend
   */
  async rejectItem(
    item: UnifiedApproval,
    request: ApprovalActionRequest,
  ): Promise<ApprovalActionResponse | null> {
    approvalState.update((s) => ({ ...s, loading: true }));

    try {
      let response: Response;

      if (item.source === 'hitl') {
        // New ApprovalManager endpoint
        response = await fetch(`/api/approvals/${item.id}/reject`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ reason: request.notes || '' }),
        });
      } else {
        // Legacy gate endpoint
        response = await fetch(`/api/approval-gates/${item.id}/reject`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(request),
        });
      }

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to reject');
      }

      const result = await response.json();

      // Remove from pending, add to history
      approvalState.update((s) => ({
        ...s,
        loading: false,
        pendingItems: s.pendingItems.filter((p) => p.id !== item.id),
        historyItems: [
          { ...item, status: 'rejected' as ApprovalStatus, approver: request.approver },
          ...s.historyItems,
        ].slice(0, 100),
      }));

      // Push resolved notification
      _pushResolvedNotification({ ...item, status: 'rejected' });

      return {
        success: true,
        gate_id: item.source === 'gate' ? item.id : undefined,
        approval_id: item.source === 'hitl' ? item.id : undefined,
        status: 'rejected',
        message: result.message || 'Rejected',
        approver: request.approver,
        timestamp: new Date().toISOString(),
      };
    } catch (err) {
      approvalState.update((s) => ({
        ...s,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to reject',
      }));
      return null;
    }
  },

  // -- Legacy gate actions (delegate to unified methods) --

  async approveGate(
    gateId: string,
    request: ApprovalActionRequest,
  ): Promise<ApprovalActionResponse | null> {
    const state = get(approvalState);
    const item = state.pendingItems.find((p) => p.id === gateId);
    if (!item) return null;
    return this.approveItem(item, request);
  },

  async rejectGate(
    gateId: string,
    request: ApprovalActionRequest,
  ): Promise<ApprovalActionResponse | null> {
    const state = get(approvalState);
    const item = state.pendingItems.find((p) => p.id === gateId);
    if (!item) return null;
    return this.rejectItem(item, request);
  },

  /**
   * Select an approval item for detail view
   */
  selectItem(item: UnifiedApproval | null): void {
    approvalState.update((s) => ({ ...s, selectedItem: item }));
  },

  /** Legacy alias */
  selectGate(gate: any | null): void {
    if (!gate) return this.selectItem(null);
    this.selectItem(normaliseGate(gate));
  },

  clearError(): void {
    approvalState.update((s) => ({ ...s, error: null }));
  },

  reset(): void {
    this.stopPolling();
    _notifiedNewIds.clear();
    _notifiedResolvedIds.clear();
    approvalState.set(initialState);
  },
};

export default approvalStore;

<script lang="ts">
  /**
   * LiveTradingMailPage — Department Mail sub-page for the Live Trading canvas.
   *
   * Shows the TRADING department inbox with inline message expand.
   * Compose overlay allows sending to any department.
   * Uses /api/departments/mail endpoints directly (no shared store).
   */
  import { onMount } from 'svelte';
  import {
    Mail,
    Inbox,
    Send,
    RefreshCw,
    ChevronDown,
    ChevronUp,
    X,
    Edit,
    AlertTriangle,
    CheckCircle,
    Info,
    Loader
  } from 'lucide-svelte';

  // ─── Types ────────────────────────────────────────────────────────────────
  type Priority = 'low' | 'normal' | 'high' | 'urgent';
  type MessageType = 'dispatch' | 'result' | 'question' | 'status' | 'error' | 'approval_request' | 'approval_approved' | 'approval_rejected' | 'escalation' | 'health_check';

  interface MailMessage {
    id: string;
    from_dept: string;
    to_dept: string;
    type: MessageType;
    subject: string;
    body: string;
    priority: Priority;
    timestamp: string;
    read: boolean;
    gate_id?: string;
    workflow_id?: string;
  }

  // ─── Constants ────────────────────────────────────────────────────────────
  const API_BASE = '/api/departments/mail';
  const TRADING_DEPT = 'trading';

  const DEPT_LABELS: Record<string, string> = {
    development: 'Development',
    research: 'Research',
    risk: 'Risk',
    trading: 'Trading',
    portfolio: 'Portfolio',
    floor_manager: 'Floor Manager',
  };

  const PRIORITY_STYLES: Record<Priority, { label: string; color: string; bg: string; border: string }> = {
    low:    { label: 'LOW',    color: '#6b7280', bg: 'rgba(107,114,128,0.12)', border: 'rgba(107,114,128,0.3)' },
    normal: { label: 'NORMAL', color: '#00d4ff', bg: 'rgba(0,212,255,0.10)',   border: 'rgba(0,212,255,0.25)' },
    high:   { label: 'HIGH',   color: '#f0a500', bg: 'rgba(240,165,0,0.12)',   border: 'rgba(240,165,0,0.3)' },
    urgent: { label: 'URGENT', color: '#ff3b3b', bg: 'rgba(255,59,59,0.12)',   border: 'rgba(255,59,59,0.3)' },
  };

  const MSG_TYPE_LABELS: Record<string, string> = {
    dispatch: 'Dispatch',
    result: 'Result',
    question: 'Question',
    status: 'Status',
    error: 'Error',
    approval_request: 'Approval',
    approval_approved: 'Approved',
    approval_rejected: 'Rejected',
    escalation: 'Escalation',
    health_check: 'Health',
  };

  const DEPARTMENTS = ['development', 'research', 'risk', 'trading', 'portfolio'];

  // ─── State ────────────────────────────────────────────────────────────────
  let messages = $state<MailMessage[]>([]);
  let loading = $state(false);
  let loadError = $state<string | null>(null);
  let expandedId = $state<string | null>(null);
  let refreshing = $state(false);

  // Compose form state
  let showCompose = $state(false);
  let composeTo = $state('');
  let composeSubject = $state('');
  let composeBody = $state('');
  let composePriority = $state<Priority>('normal');
  let composeType = $state<string>('status');
  let composeSending = $state(false);
  let composeSendError = $state<string | null>(null);
  let composeSendSuccess = $state(false);

  // ─── Derived ──────────────────────────────────────────────────────────────
  let unreadCount = $derived(messages.filter(m => !m.read).length);

  // ─── Helpers ──────────────────────────────────────────────────────────────
  function formatTimestamp(ts: string): string {
    const date = new Date(ts);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60_000);
    const diffHours = Math.floor(diffMs / 3_600_000);
    const diffDays = Math.floor(diffMs / 86_400_000);
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  }

  function deptLabel(id: string): string {
    return DEPT_LABELS[id] ?? (id.charAt(0).toUpperCase() + id.slice(1));
  }

  function deptInitial(id: string): string {
    return (DEPT_LABELS[id] ?? id).charAt(0).toUpperCase();
  }

  function priorityStyle(p: Priority) {
    return PRIORITY_STYLES[p] ?? PRIORITY_STYLES.normal;
  }

  // ─── Data fetching ────────────────────────────────────────────────────────
  async function fetchInbox(silent = false) {
    if (!silent) loading = true;
    else refreshing = true;
    loadError = null;

    try {
      const res = await fetch(`${API_BASE}/inbox/${TRADING_DEPT}?limit=50`);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const data = await res.json();
      messages = data.messages ?? [];
    } catch (e) {
      loadError = e instanceof Error ? e.message : 'Failed to load inbox';
    } finally {
      loading = false;
      refreshing = false;
    }
  }

  async function markRead(id: string) {
    try {
      await fetch(`${API_BASE}/${id}/read`, { method: 'PATCH' });
      messages = messages.map(m => m.id === id ? { ...m, read: true } : m);
    } catch {
      // non-critical — message still expands
    }
  }

  function toggleExpand(msg: MailMessage) {
    if (expandedId === msg.id) {
      expandedId = null;
    } else {
      expandedId = msg.id;
      if (!msg.read) markRead(msg.id);
    }
  }

  // ─── Compose ──────────────────────────────────────────────────────────────
  function openCompose() {
    composeTo = '';
    composeSubject = '';
    composeBody = '';
    composePriority = 'normal';
    composeType = 'status';
    composeSendError = null;
    composeSendSuccess = false;
    showCompose = true;
  }

  function closeCompose() {
    showCompose = false;
  }

  async function sendMessage() {
    if (!composeTo || !composeSubject || !composeBody) {
      composeSendError = 'All fields are required.';
      return;
    }
    composeSending = true;
    composeSendError = null;
    composeSendSuccess = false;

    try {
      const res = await fetch(`${API_BASE}/send`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          from_dept: TRADING_DEPT,
          to_dept: composeTo,
          subject: composeSubject,
          body: composeBody,
          priority: composePriority,
          type: composeType,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail ?? `${res.status} ${res.statusText}`);
      }
      composeSendSuccess = true;
      setTimeout(() => {
        closeCompose();
        fetchInbox(true);
      }, 800);
    } catch (e) {
      composeSendError = e instanceof Error ? e.message : 'Failed to send message';
    } finally {
      composeSending = false;
    }
  }

  // ─── Lifecycle ────────────────────────────────────────────────────────────
  onMount(() => {
    fetchInbox();
  });
</script>

<!-- ─────────────────────────────────────────────────────────────────────── -->
<div class="mail-page">

  <!-- Sub-header -->
  <div class="mail-header">
    <div class="mail-title-row">
      <Inbox size={15} />
      <span class="mail-title">Trading Department — Inbox</span>
      {#if unreadCount > 0}
        <span class="unread-badge">{unreadCount}</span>
      {/if}
    </div>
    <div class="mail-actions">
      <button class="mail-action-btn compose-btn" onclick={openCompose} title="Compose">
        <Edit size={13} />
        <span>Compose</span>
      </button>
      <button
        class="mail-action-btn"
        onclick={() => fetchInbox(true)}
        disabled={refreshing}
        title="Refresh inbox"
      >
        <span class:spinning={refreshing}><RefreshCw size={13} /></span>
      </button>
    </div>
  </div>

  <!-- Error state -->
  {#if loadError}
    <div class="mail-error">
      <AlertTriangle size={14} />
      <span>{loadError}</span>
      <button class="retry-btn" onclick={() => fetchInbox()}>Retry</button>
    </div>
  {/if}

  <!-- Loading state -->
  {#if loading}
    <div class="mail-loading">
      <span class="spinning"><Loader size={16} /></span>
      <span>Loading inbox…</span>
    </div>

  <!-- Empty state -->
  {:else if messages.length === 0 && !loadError}
    <div class="mail-empty">
      <Mail size={32} strokeWidth={1} />
      <p>No messages in Trading inbox.</p>
    </div>

  <!-- Message list -->
  {:else}
    <div class="message-list">
      {#each messages as msg (msg.id)}
        {@const ps = priorityStyle(msg.priority)}
        {@const isExpanded = expandedId === msg.id}

        <div
          class="message-row"
          class:unread={!msg.read}
          class:expanded={isExpanded}
        >
          <!-- Row header (always visible) -->
          <button class="message-row-header" onclick={() => toggleExpand(msg)}>
            <!-- Dept avatar -->
            <div
              class="dept-avatar"
              style="background: rgba(0,212,255,0.08); border-color: rgba(0,212,255,0.2);"
            >
              {deptInitial(msg.from_dept)}
            </div>

            <!-- Message meta -->
            <div class="message-meta">
              <div class="meta-top">
                <span class="from-dept">{deptLabel(msg.from_dept)}</span>
                <span class="msg-type-chip">{MSG_TYPE_LABELS[msg.type] ?? msg.type}</span>
                <span
                  class="priority-badge"
                  style="color:{ps.color}; background:{ps.bg}; border-color:{ps.border};"
                >
                  {ps.label}
                </span>
                <span class="msg-timestamp">{formatTimestamp(msg.timestamp)}</span>
              </div>
              <div class="msg-subject" class:unread-subject={!msg.read}>
                {msg.subject}
              </div>
            </div>

            <!-- Expand chevron -->
            <div class="expand-icon">
              {#if isExpanded}
                <ChevronUp size={14} />
              {:else}
                <ChevronDown size={14} />
              {/if}
            </div>
          </button>

          <!-- Expanded body -->
          {#if isExpanded}
            <div class="message-body">
              <pre class="body-text">{msg.body}</pre>
              {#if msg.gate_id}
                <div class="gate-chip">
                  <Info size={11} />
                  Gate: {msg.gate_id}
                </div>
              {/if}
            </div>
          {/if}
        </div>
      {/each}
    </div>
  {/if}

</div>

<!-- ── Compose Overlay ──────────────────────────────────────────────────── -->
{#if showCompose}
  <div class="compose-overlay" role="dialog" aria-modal="true">
    <div class="compose-modal">
      <div class="compose-header">
        <div class="compose-title">
          <Send size={14} />
          <span>New Message — from Trading</span>
        </div>
        <button class="close-compose-btn" onclick={closeCompose} title="Close">
          <X size={14} />
        </button>
      </div>

      <div class="compose-fields">
        <!-- To -->
        <div class="field-group">
          <label class="field-label" for="compose-to">To</label>
          <select id="compose-to" class="field-select" bind:value={composeTo}>
            <option value="" disabled>Select department…</option>
            {#each DEPARTMENTS.filter(d => d !== TRADING_DEPT) as dept}
              <option value={dept}>{DEPT_LABELS[dept] ?? dept}</option>
            {/each}
          </select>
        </div>

        <!-- Subject -->
        <div class="field-group">
          <label class="field-label" for="compose-subject">Subject</label>
          <input
            id="compose-subject"
            class="field-input"
            type="text"
            placeholder="Message subject…"
            bind:value={composeSubject}
          />
        </div>

        <!-- Priority + Type row -->
        <div class="field-row">
          <div class="field-group field-half">
            <label class="field-label" for="compose-priority">Priority</label>
            <select id="compose-priority" class="field-select" bind:value={composePriority}>
              <option value="low">Low</option>
              <option value="normal">Normal</option>
              <option value="high">High</option>
              <option value="urgent">Urgent</option>
            </select>
          </div>
          <div class="field-group field-half">
            <label class="field-label" for="compose-type">Type</label>
            <select id="compose-type" class="field-select" bind:value={composeType}>
              <option value="status">Status</option>
              <option value="question">Question</option>
              <option value="result">Result</option>
              <option value="dispatch">Dispatch</option>
              <option value="error">Error</option>
              <option value="escalation">Escalation</option>
            </select>
          </div>
        </div>

        <!-- Body -->
        <div class="field-group">
          <label class="field-label" for="compose-body">Message</label>
          <textarea
            id="compose-body"
            class="field-textarea"
            placeholder="Write your message…"
            rows={5}
            bind:value={composeBody}
          ></textarea>
        </div>
      </div>

      <!-- Feedback -->
      {#if composeSendError}
        <div class="compose-error">
          <AlertTriangle size={13} />
          {composeSendError}
        </div>
      {/if}
      {#if composeSendSuccess}
        <div class="compose-success">
          <CheckCircle size={13} />
          Message sent!
        </div>
      {/if}

      <!-- Footer actions -->
      <div class="compose-footer">
        <button class="cancel-btn" onclick={closeCompose} disabled={composeSending}>Cancel</button>
        <button
          class="send-btn"
          onclick={sendMessage}
          disabled={composeSending || !composeTo || !composeSubject || !composeBody}
        >
          {#if composeSending}
            <span class="spinning"><Loader size={13} /></span>
            Sending…
          {:else}
            <Send size={13} />
            Send
          {/if}
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  /* ── Page shell ──────────────────────────────────────────────────────── */
  .mail-page {
    display: flex;
    flex-direction: column;
    gap: 12px;
    max-width: 860px;
    margin: 0 auto;
    width: 100%;
  }

  /* ── Mail sub-header ─────────────────────────────────────────────────── */
  .mail-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    background: rgba(8, 13, 20, 0.35);
    border: 1px solid rgba(0, 212, 255, 0.12);
    border-radius: 8px;
    gap: 10px;
  }

  .mail-title-row {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #5a6a80;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .mail-title {
    color: #e8edf5;
  }

  .unread-badge {
    background: rgba(0, 212, 255, 0.15);
    color: #00d4ff;
    border: 1px solid rgba(0, 212, 255, 0.3);
    border-radius: 10px;
    padding: 1px 7px;
    font-size: 10px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0;
    text-transform: none;
  }

  .mail-actions {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .mail-action-btn {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    background: rgba(0, 212, 255, 0.07);
    border: 1px solid rgba(0, 212, 255, 0.18);
    border-radius: 5px;
    color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
  }

  .mail-action-btn:hover:not(:disabled) {
    background: rgba(0, 212, 255, 0.14);
    border-color: rgba(0, 212, 255, 0.35);
  }

  .mail-action-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .compose-btn {
    background: rgba(0, 212, 255, 0.12);
  }

  /* ── Loading / Error / Empty ─────────────────────────────────────────── */
  .mail-loading,
  .mail-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 48px 0;
    color: #5a6a80;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }

  .mail-error {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    background: rgba(255, 59, 59, 0.08);
    border: 1px solid rgba(255, 59, 59, 0.2);
    border-radius: 6px;
    color: #ff3b3b;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }

  .retry-btn {
    margin-left: auto;
    background: none;
    border: 1px solid rgba(255, 59, 59, 0.35);
    border-radius: 4px;
    color: #ff3b3b;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    padding: 2px 8px;
    cursor: pointer;
  }

  /* ── Message list ────────────────────────────────────────────────────── */
  .message-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .message-row {
    background: rgba(8, 13, 20, 0.30);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 7px;
    overflow: hidden;
    transition: border-color 0.15s;
  }

  .message-row.unread {
    border-left: 3px solid rgba(0, 212, 255, 0.6);
    background: rgba(0, 212, 255, 0.03);
  }

  .message-row.expanded {
    border-color: rgba(0, 212, 255, 0.2);
  }

  .message-row-header {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    padding: 10px 12px;
    background: none;
    border: none;
    cursor: pointer;
    color: inherit;
    text-align: left;
    font-family: inherit;
    transition: background 0.12s;
  }

  .message-row-header:hover {
    background: rgba(255, 255, 255, 0.03);
  }

  /* ── Dept avatar ─────────────────────────────────────────────────────── */
  .dept-avatar {
    flex-shrink: 0;
    width: 28px;
    height: 28px;
    border-radius: 6px;
    border: 1px solid;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    color: #00d4ff;
  }

  /* ── Message meta row ────────────────────────────────────────────────── */
  .message-meta {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }

  .meta-top {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }

  .from-dept {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    color: #e8edf5;
  }

  .msg-type-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #5a6a80;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 4px;
    padding: 1px 5px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .priority-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    border: 1px solid;
    border-radius: 4px;
    padding: 1px 5px;
    letter-spacing: 0.03em;
  }

  .msg-timestamp {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #3d4f63;
    margin-left: auto;
    flex-shrink: 0;
  }

  .msg-subject {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #8a9bb0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
  }

  .msg-subject.unread-subject {
    color: #c8d8e8;
    font-weight: 500;
  }

  .expand-icon {
    flex-shrink: 0;
    color: #3d4f63;
  }

  /* ── Expanded body ───────────────────────────────────────────────────── */
  .message-body {
    padding: 0 12px 12px 50px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .body-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #a8bccf;
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 5px;
    padding: 10px 12px;
    white-space: pre-wrap;
    word-break: break-word;
    margin: 0;
    line-height: 1.55;
  }

  .gate-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #f0a500;
    background: rgba(240, 165, 0, 0.08);
    border: 1px solid rgba(240, 165, 0, 0.2);
    border-radius: 4px;
    padding: 2px 8px;
    align-self: flex-start;
  }

  /* ── Spinning animation ──────────────────────────────────────────────── */
  :global(.spinning) {
    display: inline-flex;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
  }

  /* ── Compose overlay ─────────────────────────────────────────────────── */
  .compose-overlay {
    position: fixed;
    inset: 0;
    z-index: 200;
    background: rgba(0, 0, 0, 0.55);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 16px;
  }

  .compose-modal {
    background: rgba(8, 16, 28, 0.92);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 10px;
    width: 100%;
    max-width: 520px;
    display: flex;
    flex-direction: column;
    gap: 0;
    box-shadow: 0 24px 64px rgba(0, 0, 0, 0.6);
    overflow: hidden;
  }

  .compose-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    border-bottom: 1px solid rgba(0, 212, 255, 0.1);
    background: rgba(0, 212, 255, 0.04);
  }

  .compose-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    color: #00d4ff;
    letter-spacing: 0.02em;
  }

  .close-compose-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    color: #5a6a80;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
  }

  .close-compose-btn:hover {
    background: rgba(255, 59, 59, 0.12);
    color: #ff3b3b;
    border-color: rgba(255, 59, 59, 0.25);
  }

  .compose-fields {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 14px 16px;
  }

  .field-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .field-row {
    display: flex;
    gap: 10px;
  }

  .field-half {
    flex: 1;
  }

  .field-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #5a6a80;
  }

  .field-input,
  .field-select,
  .field-textarea {
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 5px;
    color: #e8edf5;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    padding: 7px 10px;
    transition: border-color 0.15s;
    outline: none;
    width: 100%;
    box-sizing: border-box;
  }

  .field-input:focus,
  .field-select:focus,
  .field-textarea:focus {
    border-color: rgba(0, 212, 255, 0.4);
  }

  .field-select option {
    background: #0d1520;
    color: #e8edf5;
  }

  .field-textarea {
    resize: vertical;
    min-height: 90px;
    line-height: 1.5;
  }

  /* ── Compose feedback ────────────────────────────────────────────────── */
  .compose-error {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 8px 16px;
    background: rgba(255, 59, 59, 0.08);
    color: #ff3b3b;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    border-top: 1px solid rgba(255, 59, 59, 0.15);
  }

  .compose-success {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 8px 16px;
    background: rgba(0, 200, 150, 0.08);
    color: #00c896;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    border-top: 1px solid rgba(0, 200, 150, 0.15);
  }

  /* ── Compose footer ──────────────────────────────────────────────────── */
  .compose-footer {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 8px;
    padding: 10px 16px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    background: rgba(0, 0, 0, 0.2);
  }

  .cancel-btn {
    padding: 6px 14px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 5px;
    color: #5a6a80;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: background 0.15s;
  }

  .cancel-btn:hover:not(:disabled) {
    background: rgba(255, 255, 255, 0.08);
  }

  .send-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 16px;
    background: rgba(0, 212, 255, 0.12);
    border: 1px solid rgba(0, 212, 255, 0.3);
    border-radius: 5px;
    color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
  }

  .send-btn:hover:not(:disabled) {
    background: rgba(0, 212, 255, 0.22);
    border-color: rgba(0, 212, 255, 0.5);
  }

  .send-btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
</style>

<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Mail, Send, Inbox, CheckCircle, AlertCircle, Clock, Filter, RefreshCw, X, ChevronRight, User } from 'lucide-svelte';
  import {
    departmentMailStore,
    filteredInbox,
    sentMessages,
    mailStats,
    mailLoading,
    mailError,
    selectedDepartment,
    unreadCount,
    DEPARTMENT_COLORS,
    PRIORITY_COLORS,
    fetchAllInbox,
    fetchSent,
    fetchStats,
    fetchMessage,
    markAsRead,
    setSelectedDepartment,
    setSelectedMessage,
    clearMail,
    type DepartmentMailMessage,
    type MessagePriority,
    type MessageType,
  } from '$lib/stores/departmentMailStore';

  // Local state
  let viewMode: 'inbox' | 'sent' | 'stats' = 'inbox';
  let refreshInterval: number | null = null;

  // Reactive state from store
  $: inbox = $filteredInbox;
  $: sent = $sentMessages;
  $: stats = $mailStats;
  $: loading = $mailLoading;
  $: error = $mailError;
  $: selectedDept = $selectedDepartment;
  $: unread = $unreadCount;

  // Get store directly for selected message
  $: state = $departmentMailStore;
  $: selectedMsg = state.selectedMessage;

  // Departments for filtering
  const departments = ['analysis', 'research', 'risk', 'execution', 'portfolio'];

  onMount(() => {
    loadMailData();
    // Set up polling for real-time updates
    refreshInterval = window.setInterval(() => {
      fetchStats();
    }, 30000); // Refresh stats every 30 seconds
  });

  onDestroy(() => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }
  });

  async function loadMailData() {
    await Promise.all([fetchAllInbox(), fetchSent(), fetchStats()]);
  }

  async function handleRefresh() {
    await loadMailData();
  }

  function handleSelectDepartment(dept: string | null) {
    setSelectedDepartment(dept);
  }

  async function handleSelectMessage(message: DepartmentMailMessage) {
    setSelectedMessage(message);
    if (!message.is_read) {
      await markAsRead(message.id);
    }
  }

  function handleCloseMessage() {
    setSelectedMessage(null);
  }

  function formatTimestamp(timestamp: string): string {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  }

  function getPriorityBadgeClass(priority: MessagePriority): string {
    const classes: Record<MessagePriority, string> = {
      low: 'priority-low',
      normal: 'priority-normal',
      high: 'priority-high',
      urgent: 'priority-urgent',
    };
    return classes[priority];
  }

  function getTypeIcon(type: MessageType): typeof Mail {
    switch (type) {
      case 'dispatch': return Send;
      case 'result': return CheckCircle;
      case 'question': return AlertCircle;
      case 'escalation': return AlertCircle;
      case 'health_check': return Clock;
      default: return Mail;
    }
  }
</script>

<div class="mail-panel">
  <!-- Header -->
  <div class="mail-header">
    <div class="header-title">
      <Mail size={18} />
      <h3>Department Mail</h3>
      {#if unread > 0}
        <span class="unread-badge">{unread}</span>
      {/if}
    </div>
    <div class="header-actions">
      <button class="refresh-btn" on:click={handleRefresh} disabled={loading}>
        <RefreshCw size={14} class={loading ? 'spinning' : ''} />
      </button>
    </div>
  </div>

  <!-- View Tabs -->
  <div class="view-tabs">
    <button
      class="tab-btn"
      class:active={viewMode === 'inbox'}
      on:click={() => viewMode = 'inbox'}
    >
      <Inbox size={14} />
      Inbox
      {#if stats.total_unread > 0}
        <span class="mini-badge">{stats.total_unread}</span>
      {/if}
    </button>
    <button
      class="tab-btn"
      class:active={viewMode === 'sent'}
      on:click={() => viewMode = 'sent'}
    >
      <Send size={14} />
      Sent
    </button>
    <button
      class="tab-btn"
      class:active={viewMode === 'stats'}
      on:click={() => viewMode = 'stats'}
    >
      <Clock size={14} />
      Stats
    </button>
  </div>

  <!-- Department Filter (only in inbox view) -->
  {#if viewMode === 'inbox'}
  <div class="department-filter">
    <div class="filter-label">
      <Filter size={12} />
      Filter by department
    </div>
    <div class="filter-chips">
      <button
        class="chip"
        class:active={!selectedDept}
        on:click={() => handleSelectDepartment(null)}
      >
        All
      </button>
      {#each departments as dept}
        <button
          class="chip"
          class:active={selectedDept === dept}
          style="--dept-color: {DEPARTMENT_COLORS[dept]}"
          on:click={() => handleSelectDepartment(dept)}
        >
          <span class="chip-dot"></span>
          {dept}
          {#if stats.unread_by_department[dept] > 0}
            <span class="chip-count">{stats.unread_by_department[dept]}</span>
          {/if}
        </button>
      {/each}
    </div>
  </div>
  {/if}

  <!-- Error Banner -->
  {#if error}
  <div class="error-banner">
    <AlertCircle size={14} />
    {error}
    <button class="dismiss-btn" on:click={() => ({})}>
      <X size={12} />
    </button>
  </div>
  {/if}

  <!-- Main Content -->
  <div class="mail-content">
    {#if loading && inbox.length === 0}
      <div class="loading-state">
        <RefreshCw size={24} class="spinning" />
        <span>Loading messages...</span>
      </div>
    {:else if selectedMsg}
      <!-- Message Detail View -->
      <div class="message-detail">
        <div class="detail-header">
          <button class="back-btn" on:click={handleCloseMessage}>
            <ChevronRight size={16} />
            Back
          </button>
          <div class="detail-meta">
            <span class="priority-badge {getPriorityBadgeClass(selectedMsg.priority)}">
              {selectedMsg.priority}
            </span>
            <span class="type-badge">
              {selectedMsg.message_type}
            </span>
          </div>
        </div>

        <div class="detail-subject">{selectedMsg.subject}</div>

        <div class="detail-parties">
          <div class="party from">
            <span class="label">From:</span>
            <span class="dept-badge" style="--dept-color: {DEPARTMENT_COLORS[selectedMsg.from_department]}">
              {selectedMsg.from_department}
            </span>
            {#if selectedMsg.from_agent}
              <span class="agent-name">
                <User size={10} />
                {selectedMsg.from_agent}
              </span>
            {/if}
          </div>
          <div class="party to">
            <span class="label">To:</span>
            <span class="dept-badge" style="--dept-color: {DEPARTMENT_COLORS[selectedMsg.to_department]}">
              {selectedMsg.to_department}
            </span>
            {#if selectedMsg.to_agent}
              <span class="agent-name">
                <User size={10} />
                {selectedMsg.to_agent}
              </span>
            {/if}
          </div>
        </div>

        <div class="detail-body">
          {selectedMsg.body}
        </div>

        <div class="detail-footer">
          <span class="timestamp">
            <Clock size={12} />
            {formatTimestamp(selectedMsg.created_at)}
          </span>
          {#if selectedMsg.is_read && selectedMsg.read_at}
            <span class="read-status">
              <CheckCircle size={12} />
              Read {formatTimestamp(selectedMsg.read_at)}
            </span>
          {/if}
        </div>
      </div>
    {:else if viewMode === 'inbox'}
      <!-- Inbox List -->
      <div class="message-list">
        {#if inbox.length === 0}
          <div class="empty-state">
            <Inbox size={32} />
            <p>No messages in inbox</p>
          </div>
        {:else}
          {#each inbox as message (message.id)}
            <button
              class="message-item"
              class:unread={!message.is_read}
              on:click={() => handleSelectMessage(message)}
            >
              <div class="item-indicator" style="background-color: {DEPARTMENT_COLORS[message.from_department]}"></div>
              <div class="item-content">
                <div class="item-header">
                  <span class="from-dept" style="color: {DEPARTMENT_COLORS[message.from_department]}">
                    {message.from_department}
                  </span>
                  <span class="to-arrow">→</span>
                  <span class="to-dept" style="color: {DEPARTMENT_COLORS[message.to_department]}">
                    {message.to_department}
                  </span>
                  <span class="timestamp">{formatTimestamp(message.created_at)}</span>
                </div>
                <div class="item-subject">{message.subject}</div>
                <div class="item-footer">
                  <span class="priority-dot {getPriorityBadgeClass(message.priority)}"></span>
                  <span class="message-type">{message.message_type}</span>
                </div>
              </div>
              {#if !message.is_read}
                <div class="unread-dot"></div>
              {/if}
            </button>
          {/each}
        {/if}
      </div>
    {:else if viewMode === 'sent'}
      <!-- Sent List -->
      <div class="message-list">
        {#if sent.length === 0}
          <div class="empty-state">
            <Send size={32} />
            <p>No sent messages</p>
          </div>
        {:else}
          {#each sent as message (message.id)}
            <div class="message-item sent-item">
              <div class="item-indicator" style="background-color: {DEPARTMENT_COLORS[message.to_department]}"></div>
              <div class="item-content">
                <div class="item-header">
                  <span class="to-dept" style="color: {DEPARTMENT_COLORS[message.to_department]}">
                    To: {message.to_department}
                  </span>
                  <span class="timestamp">{formatTimestamp(message.created_at)}</span>
                </div>
                <div class="item-subject">{message.subject}</div>
                <div class="item-footer">
                  <span class="priority-dot {getPriorityBadgeClass(message.priority)}"></span>
                  <span class="message-type">{message.message_type}</span>
                </div>
              </div>
            </div>
          {/each}
        {/if}
      </div>
    {:else if viewMode === 'stats'}
      <!-- Stats View -->
      <div class="stats-view">
        <div class="stat-cards">
          <div class="stat-card">
            <span class="stat-value">{stats.total_inbox}</span>
            <span class="stat-label">Total Inbox</span>
          </div>
          <div class="stat-card">
            <span class="stat-value">{stats.total_unread}</span>
            <span class="stat-label">Unread</span>
          </div>
          <div class="stat-card">
            <span class="stat-value">{stats.total_sent}</span>
            <span class="stat-label">Sent</span>
          </div>
        </div>

        <div class="stats-section">
          <h4>Unread by Department</h4>
          <div class="dept-stats">
            {#each departments as dept}
              <div class="dept-stat-row">
                <div class="dept-info">
                  <span class="dept-dot" style="background-color: {DEPARTMENT_COLORS[dept]}"></span>
                  <span class="dept-name">{dept}</span>
                </div>
                <span class="dept-count">{stats.unread_by_department[dept] || 0}</span>
              </div>
            {/each}
          </div>
        </div>

        <div class="stats-section">
          <h4>By Priority</h4>
          <div class="priority-stats">
            {#each Object.entries(stats.by_priority) as [priority, count]}
              <div class="priority-stat-row">
                <span class="priority-label {`priority-${priority}`}">{priority}</span>
                <div class="priority-bar-container">
                  <div
                    class="priority-bar"
                    style="width: {Math.min((count / (stats.total_inbox || 1)) * 100, 100)}%; background-color: {PRIORITY_COLORS[priority]}"
                  ></div>
                </div>
                <span class="priority-count">{count}</span>
              </div>
            {/each}
          </div>
        </div>
      </div>
    {/if}
  </div>
</div>

<style>
  .mail-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary, #0a0f1a);
    color: var(--text-primary, #e2e8f0);
    border-radius: 0.5rem;
    overflow: hidden;
  }

  .mail-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1rem;
    background: var(--bg-secondary, #111827);
    border-bottom: 1px solid var(--border-color, #1e293b);
  }

  .header-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--accent-primary, #3b82f6);
  }

  .header-title h3 {
    margin: 0;
    font-size: 0.875rem;
    font-weight: 600;
  }

  .unread-badge {
    background: #ef4444;
    color: white;
    font-size: 0.6875rem;
    font-weight: 600;
    padding: 0.125rem 0.375rem;
    border-radius: 0.5rem;
    min-width: 18px;
    text-align: center;
  }

  .header-actions {
    display: flex;
    gap: 0.25rem;
  }

  .refresh-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0.375rem;
    background: transparent;
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.25rem;
    color: var(--text-secondary, #94a3b8);
    cursor: pointer;
    transition: all 0.15s;
  }

  .refresh-btn:hover:not(:disabled) {
    background: var(--bg-tertiary, #1e293b);
    color: var(--text-primary, #e2e8f0);
  }

  .refresh-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .view-tabs {
    display: flex;
    gap: 0.25rem;
    padding: 0.5rem 1rem;
    background: var(--bg-secondary, #111827);
    border-bottom: 1px solid var(--border-color, #1e293b);
  }

  .tab-btn {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.375rem 0.75rem;
    background: transparent;
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.375rem;
    color: var(--text-secondary, #94a3b8);
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  .tab-btn:hover {
    background: var(--bg-tertiary, #1e293b);
    color: var(--text-primary, #e2e8f0);
  }

  .tab-btn.active {
    background: var(--accent-primary, #3b82f6);
    border-color: var(--accent-primary, #3b82f6);
    color: white;
  }

  .mini-badge {
    background: rgba(255, 255, 255, 0.2);
    padding: 0.125rem 0.25rem;
    border-radius: 0.25rem;
    font-size: 0.625rem;
    font-weight: 600;
  }

  .department-filter {
    padding: 0.75rem 1rem;
    background: var(--bg-secondary, #111827);
    border-bottom: 1px solid var(--border-color, #1e293b);
  }

  .filter-label {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    font-size: 0.6875rem;
    color: var(--text-muted, #64748b);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
  }

  .filter-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.375rem;
  }

  .chip {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.25rem 0.5rem;
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 1rem;
    color: var(--text-secondary, #94a3b8);
    font-size: 0.6875rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  .chip:hover {
    background: var(--bg-hover, #334155);
    border-color: var(--dept-color, var(--accent-primary, #3b82f6));
  }

  .chip.active {
    background: color-mix(in srgb, var(--dept-color, #3b82f6) 20%, transparent);
    border-color: var(--dept-color, #3b82f6);
    color: var(--text-primary, #e2e8f0);
  }

  .chip-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: var(--dept-color, #6b7280);
  }

  .chip-count {
    background: #ef4444;
    color: white;
    padding: 0.0625rem 0.25rem;
    border-radius: 0.25rem;
    font-size: 0.5625rem;
    font-weight: 600;
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: rgba(239, 68, 68, 0.1);
    border-bottom: 1px solid rgba(239, 68, 68, 0.3);
    color: #fca5a5;
    font-size: 0.75rem;
  }

  .dismiss-btn {
    margin-left: auto;
    background: transparent;
    border: none;
    color: inherit;
    cursor: pointer;
    padding: 0.25rem;
  }

  .mail-content {
    flex: 1;
    overflow-y: auto;
    padding: 0.75rem;
  }

  .loading-state,
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem;
    color: var(--text-muted, #64748b);
    gap: 0.75rem;
  }

  .message-list {
    display: flex;
    flex-direction: column;
    gap: 0.375rem;
  }

  .message-item {
    display: flex;
    align-items: stretch;
    padding: 0.625rem;
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.5rem;
    cursor: pointer;
    transition: all 0.15s;
    text-align: left;
    width: 100%;
  }

  .message-item:hover {
    background: var(--bg-hover, #334155);
    border-color: var(--accent-primary, #3b82f6);
  }

  .message-item.unread {
    background: color-mix(in srgb, var(--accent-primary, #3b82f6) 5%, var(--bg-tertiary, #1e293b));
    border-left: 3px solid var(--accent-primary, #3b82f6);
  }

  .item-indicator {
    width: 4px;
    border-radius: 2px;
    margin-right: 0.625rem;
    flex-shrink: 0;
  }

  .item-content {
    flex: 1;
    min-width: 0;
  }

  .item-header {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    margin-bottom: 0.25rem;
    font-size: 0.6875rem;
  }

  .from-dept,
  .to-dept {
    font-weight: 500;
    text-transform: capitalize;
  }

  .to-arrow {
    color: var(--text-muted, #64748b);
  }

  .timestamp {
    margin-left: auto;
    color: var(--text-muted, #64748b);
    font-size: 0.625rem;
  }

  .item-subject {
    font-size: 0.8125rem;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 0.25rem;
  }

  .item-footer {
    display: flex;
    align-items: center;
    gap: 0.375rem;
  }

  .priority-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
  }

  .priority-dot.priority-low { background-color: #6b7280; }
  .priority-dot.priority-normal { background-color: #3b82f6; }
  .priority-dot.priority-high { background-color: #f59e0b; }
  .priority-dot.priority-urgent { background-color: #ef4444; }

  .message-type {
    font-size: 0.625rem;
    color: var(--text-muted, #64748b);
    text-transform: uppercase;
  }

  .unread-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: var(--accent-primary, #3b82f6);
    flex-shrink: 0;
    margin-left: 0.5rem;
  }

  /* Message Detail */
  .message-detail {
    padding: 0.75rem;
    background: var(--bg-tertiary, #1e293b);
    border-radius: 0.5rem;
    border: 1px solid var(--border-color, #334155);
  }

  .detail-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
  }

  .back-btn {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    background: transparent;
    border: none;
    color: var(--text-secondary, #94a3b8);
    font-size: 0.75rem;
    cursor: pointer;
    padding: 0.25rem;
  }

  .back-btn:hover {
    color: var(--text-primary, #e2e8f0);
  }

  .back-btn svg {
    transform: rotate(180deg);
  }

  .detail-meta {
    display: flex;
    gap: 0.375rem;
  }

  .priority-badge,
  .type-badge {
    font-size: 0.625rem;
    padding: 0.125rem 0.375rem;
    border-radius: 0.25rem;
    text-transform: uppercase;
    font-weight: 600;
  }

  .priority-badge.priority-low { background: rgba(107, 114, 128, 0.2); color: #9ca3af; }
  .priority-badge.priority-normal { background: rgba(59, 130, 246, 0.2); color: #60a5fa; }
  .priority-badge.priority-high { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
  .priority-badge.priority-urgent { background: rgba(239, 68, 68, 0.2); color: #f87171; }

  .type-badge {
    background: var(--bg-input, #0f172a);
    color: var(--text-secondary, #94a3b8);
  }

  .detail-subject {
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
    line-height: 1.4;
  }

  .detail-parties {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin-bottom: 1rem;
    padding: 0.75rem;
    background: var(--bg-input, #0f172a);
    border-radius: 0.375rem;
  }

  .party {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.75rem;
  }

  .party .label {
    color: var(--text-muted, #64748b);
    min-width: 40px;
  }

  .dept-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.125rem 0.5rem;
    border-radius: 0.25rem;
    background: color-mix(in srgb, var(--dept-color, #3b82f6) 20%, transparent);
    color: var(--dept-color, #3b82f6);
    font-weight: 500;
    text-transform: capitalize;
  }

  .agent-name {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    color: var(--text-muted, #64748b);
    font-size: 0.6875rem;
  }

  .detail-body {
    font-size: 0.875rem;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
    padding: 0.75rem;
    background: var(--bg-input, #0f172a);
    border-radius: 0.375rem;
    margin-bottom: 0.75rem;
  }

  .detail-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.6875rem;
    color: var(--text-muted, #64748b);
  }

  .timestamp,
  .read-status {
    display: flex;
    align-items: center;
    gap: 0.25rem;
  }

  .read-status {
    color: #22c55e;
  }

  /* Stats View */
  .stats-view {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .stat-cards {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.5rem;
  }

  .stat-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 0.75rem;
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.5rem;
  }

  .stat-value {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--accent-primary, #3b82f6);
  }

  .stat-label {
    font-size: 0.625rem;
    color: var(--text-muted, #64748b);
    text-transform: uppercase;
  }

  .stats-section {
    padding: 0.75rem;
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.5rem;
  }

  .stats-section h4 {
    margin: 0 0 0.75rem 0;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary, #94a3b8);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .dept-stats {
    display: flex;
    flex-direction: column;
    gap: 0.375rem;
  }

  .dept-stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.375rem 0;
  }

  .dept-info {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .dept-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }

  .dept-name {
    font-size: 0.75rem;
    text-transform: capitalize;
  }

  .dept-count {
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--accent-primary, #3b82f6);
  }

  .priority-stats {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .priority-stat-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .priority-label {
    font-size: 0.6875rem;
    text-transform: capitalize;
    min-width: 50px;
  }

  .priority-label.priority-low { color: #9ca3af; }
  .priority-label.priority-normal { color: #60a5fa; }
  .priority-label.priority-high { color: #fbbf24; }
  .priority-label.priority-urgent { color: #f87171; }

  .priority-bar-container {
    flex: 1;
    height: 6px;
    background: var(--bg-input, #0f172a);
    border-radius: 3px;
    overflow: hidden;
  }

  .priority-bar {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
  }

  .priority-count {
    font-size: 0.6875rem;
    color: var(--text-muted, #64748b);
    min-width: 20px;
    text-align: right;
  }

  .spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>

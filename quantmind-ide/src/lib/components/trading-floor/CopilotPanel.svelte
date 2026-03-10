<script lang="ts">
  import { createEventDispatcher, onMount, tick } from "svelte";
  import { fly, fade, slide } from "svelte/transition";
  import {
    Send,
    Loader,
    X,
    Bot,
    Plus,
    ChevronRight,
    ArrowRightCircle,
    CheckCircle2,
    Clock,
    AlertCircle,
    Users,
    Building2,
    Mail,
    MailPlus,
    MessageSquarePlus,
  } from "lucide-svelte";
  import AgentModelSelector from "$lib/components/AgentModelSelector.svelte";
  import {
    departmentChatStore,
    activeDelegatedTasks,
    DEPARTMENTS,
    type DepartmentId,
    type DepartmentInfo,
    type DelegatedTask,
  } from "$lib/stores/departmentChatStore";

  export let isCopilot: boolean = false;

  const dispatch = createEventDispatcher();

  // API base URL
  const API_BASE = "http://localhost:8000/api";

  // Floor Manager state
  let message = "";
  let textareaElement: HTMLTextAreaElement;
  let messagesContainer: HTMLDivElement;
  let settingsOpen = false;

  // Mail compose state
  let showMailCompose = false;
  let mailTo: DepartmentId = "development";
  let mailSubject = "";
  let mailBody = "";
  let mailPriority: "low" | "normal" | "high" | "urgent" = "normal";
  let mailType: "status" | "question" | "result" | "error" | "dispatch" = "dispatch";
  let sendingMail = false;

  // Messages with Floor Manager
  interface FloorManagerMessage {
    id: string;
    role: "user" | "floor_manager" | "system";
    content: string;
    timestamp: Date;
    delegation?: {
      departmentId: DepartmentId;
      taskId: string;
      status: DelegatedTask["status"];
    };
  }

  // Dynamic greeting based on agent type
  const copilotGreeting = "Hello! I'm QuantMind Copilot, your AI trading assistant. I can help with market analysis, strategy questions, or delegate tasks to the Floor Manager for trading operations. How can I assist you?";
  const floorManagerGreeting = "Hello! I'm the Floor Manager. I coordinate tasks across all departments - Analysis, Research, Risk, Execution, and Portfolio. I can delegate tasks, check status, or answer questions about the trading floor. How can I help?";

  $: greeting = isCopilot ? copilotGreeting : floorManagerGreeting;
  $: agentName = isCopilot ? "QuantMind Copilot" : "Floor Manager";
  $: placeholderText = isCopilot ? "Ask QuantMind Copilot..." : "Ask the Floor Manager...";
  $: apiEndpoint = isCopilot
    ? `${API_BASE}/chat/workshop/message`
    : `${API_BASE}/chat/floor-manager/message`;
  $: messages = [
    {
      id: "fm_welcome",
      role: isCopilot ? "copilot" : "floor_manager",
      content: greeting,
      timestamp: new Date(),
    },
  ];

  // Department info for display
  const departmentList = Object.values(DEPARTMENTS);

  // Subscribed values
  $: activeTasks = $activeDelegatedTasks;
  $: isLoading = $departmentChatStore.isLoading;

  // Generate message ID
  function generateId(): string {
    return `fm_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Auto-resize textarea
  function autoResize() {
    if (textareaElement) {
      textareaElement.style.height = "auto";
      const newHeight = Math.min(Math.max(textareaElement.scrollHeight, 60), 150);
      textareaElement.style.height = newHeight + "px";
    }
  }

  // Send message to Floor Manager
  async function sendMessage() {
    if (!message.trim() || isLoading) return;

    const userContent = message.trim();
    message = "";

    // Add user message
    messages = [
      ...messages,
      {
        id: generateId(),
        role: "user",
        content: userContent,
        timestamp: new Date(),
      },
    ];

    await tick();
    scrollToBottom();

    try {
      const response = await fetch(apiEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userContent,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();

      // Add Floor Manager response
      messages = [
        ...messages,
        {
          id: data.message_id || generateId(),
          role: "floor_manager",
          content: data.reply || "I've processed your request.",
          timestamp: new Date(),
          delegation: data.delegation
            ? {
                departmentId: data.delegation.department,
                taskId: data.delegation.task_id,
                status: data.delegation.status || "pending",
              }
            : undefined,
        },
      ];

      await tick();
      scrollToBottom();
    } catch (error) {
      messages = [
        ...messages,
        {
          id: generateId(),
          role: "system",
          content: `Error: ${error instanceof Error ? error.message : "Failed to send message"}`,
          timestamp: new Date(),
        },
      ];
    }
  }

  // Delegate to specific department
  async function delegateToDepartment(deptId: DepartmentId, task: string) {
    if (isLoading) return;

    // Add delegation message
    messages = [
      ...messages,
      {
        id: generateId(),
        role: "user",
        content: `Delegate to ${DEPARTMENTS[deptId].name}: ${task}`,
        timestamp: new Date(),
      },
    ];

    await tick();
    scrollToBottom();

    try {
      const result = await departmentChatStore.delegateTask(deptId, task);

      if (result) {
        messages = [
          ...messages,
          {
            id: generateId(),
            role: "floor_manager",
            content: `Task delegated to ${DEPARTMENTS[deptId].name}. ${result.result || "Processing..."}`,
            timestamp: new Date(),
            delegation: {
              departmentId: deptId,
              taskId: result.id,
              status: result.status,
            },
          },
        ];
      }
    } catch (error) {
      messages = [
        ...messages,
        {
          id: generateId(),
          role: "system",
          content: `Delegation failed: ${error instanceof Error ? error.message : "Unknown error"}`,
          timestamp: new Date(),
        },
      ];
    }

    await tick();
    scrollToBottom();
  }

  // Quick delegation actions
  const quickDelegations: Array<{ label: string; dept: DepartmentId; task: string }> = [
    { label: "Analyze Market", dept: "development", task: "Perform market analysis on current conditions" },
    { label: "Run Backtest", dept: "research", task: "Run backtest on active strategy" },
    { label: "Check Risk", dept: "risk", task: "Generate risk exposure report" },
    { label: "Order Status", dept: "trading", task: "Check status of all orders" },
    { label: "Portfolio Review", dept: "portfolio", task: "Generate portfolio review" },
  ];

  // Handle keyboard
  function handleKeydown(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  // Scroll to bottom
  async function scrollToBottom() {
    await tick();
    if (messagesContainer) {
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
  }

  // Format timestamp
  function formatTime(date: Date): string {
    return new Date(date).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  // Get status icon
  function getStatusIcon(status: DelegatedTask["status"]): typeof CheckCircle2 {
    switch (status) {
      case "completed":
        return CheckCircle2;
      case "error":
        return AlertCircle;
      default:
        return Clock;
    }
  }

  // Get status color
  function getStatusColor(status: DelegatedTask["status"]): string {
    switch (status) {
      case "completed":
        return "#22c55e";
      case "error":
        return "#ef4444";
      default:
        return "#f59e0b";
    }
  }

  // Clear chat
  function clearChat() {
    messages = [
      {
        id: "fm_welcome",
        role: "floor_manager",
        content:
          "Chat cleared. I'm ready to help coordinate tasks across departments. What would you like to do?",
        timestamp: new Date(),
      },
    ];
  }

  // Open department chat
  function openDepartmentChat(deptId: DepartmentId) {
    dispatch("openDepartment", { departmentId: deptId });
  }

  // Close panel
  function closePanel() {
    dispatch("close");
  }

  // Send mail to department
  async function sendMailToDepartment() {
    if (!mailSubject.trim() || !mailBody.trim() || sendingMail) return;

    sendingMail = true;

    try {
      const response = await fetch(`${API_BASE}/departments/mail/send`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          from_dept: "floor_manager",
          to_dept: mailTo,
          type: mailType,
          subject: mailSubject.trim(),
          body: mailBody.trim(),
          priority: mailPriority,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();

      // Add confirmation message
      messages = [
        ...messages,
        {
          id: generateId(),
          role: "floor_manager",
          content: `Mail sent to ${DEPARTMENTS[mailTo].name}: "${mailSubject}"`,
          timestamp: new Date(),
        },
      ];

      // Reset form
      mailSubject = "";
      mailBody = "";
      showMailCompose = false;

      await tick();
      scrollToBottom();
    } catch (error) {
      messages = [
        ...messages,
        {
          id: generateId(),
          role: "system",
          content: `Failed to send mail: ${error instanceof Error ? error.message : "Unknown error"}`,
          timestamp: new Date(),
        },
      ];
    } finally {
      sendingMail = false;
    }
  }

  // Quick mail to department
  async function quickMailToDepartment(deptId: DepartmentId, subject: string, body: string) {
    try {
      const response = await fetch(`${API_BASE}/departments/mail/send`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          from_dept: "floor_manager",
          to_dept: deptId,
          type: "dispatch",
          subject,
          body,
          priority: "normal",
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      messages = [
        ...messages,
        {
          id: generateId(),
          role: "floor_manager",
          content: `Task dispatched to ${DEPARTMENTS[deptId].name}: "${subject}"`,
          timestamp: new Date(),
        },
      ];

      await tick();
      scrollToBottom();
    } catch (error) {
      messages = [
        ...messages,
        {
          id: generateId(),
          role: "system",
          content: `Failed to dispatch: ${error instanceof Error ? error.message : "Unknown error"}`,
          timestamp: new Date(),
        },
      ];
    }
  }
</script>

<div class="floor-manager-panel">
  <!-- Header -->
  <div class="panel-header">
    <div class="header-left">
      <div class="manager-icon">
        <Building2 size={18} />
      </div>
      <div class="manager-info">
        <span class="manager-name">{agentName}</span>
        <span class="manager-role">Trading Floor Coordinator</span>
      </div>
    </div>

    <div class="header-actions">
      <button class="icon-btn mail-btn" title="Send Mail to Department" on:click={() => showMailCompose = !showMailCompose}>
        <MailPlus size={14} />
      </button>
      <button class="icon-btn" title="New Chat" on:click={clearChat}>
        <MessageSquarePlus size={14} />
      </button>
      <button class="icon-btn" title="Close" on:click={closePanel}>
        <X size={14} />
      </button>
    </div>
  </div>

  <!-- Mail Compose Modal -->
  {#if showMailCompose}
    <div class="mail-compose-modal" transition:slide={{ y: -10 }}>
      <div class="mail-compose-header">
        <Mail size={14} />
        <span>Send Mail to Department</span>
        <button class="close-modal-btn" on:click={() => showMailCompose = false}>
          <X size={12} />
        </button>
      </div>

      <div class="mail-compose-body">
        <div class="form-row">
          <label>To Department:</label>
          <div class="dept-selector">
            {#each departmentList as dept}
              <button
                class="dept-option"
                class:selected={mailTo === dept.id}
                style="--dept-color: {dept.color}"
                on:click={() => mailTo = dept.id}
              >
                <span class="dept-dot"></span>
                {dept.name}
              </button>
            {/each}
          </div>
        </div>

        <div class="form-row">
          <label>Priority:</label>
          <div class="priority-selector">
            {#each [{v: 'low', c: '#6b7280'}, {v: 'normal', c: '#3b82f6'}, {v: 'high', c: '#f59e0b'}, {v: 'urgent', c: '#ef4444'}] as p}
              <button
                class="priority-option"
                class:selected={mailPriority === p.v}
                style="--priority-color: {p.c}"
                on:click={() => mailPriority = p.v}
              >
                {p.v}
              </button>
            {/each}
          </div>
        </div>

        <div class="form-row">
          <label>Type:</label>
          <select bind:value={mailType}>
            <option value="status">Status Update</option>
            <option value="question">Question</option>
            <option value="result">Result</option>
            <option value="dispatch" selected>Task Dispatch</option>
            <option value="error">Error Report</option>
          </select>
        </div>

        <div class="form-row">
          <label>Subject:</label>
          <input type="text" bind:value={mailSubject} placeholder="Enter subject..." />
        </div>

        <div class="form-row">
          <label>Message:</label>
          <textarea bind:value={mailBody} placeholder="Enter your message to the department head..." rows="4"></textarea>
        </div>
      </div>

      <div class="mail-compose-footer">
        <button class="cancel-btn" on:click={() => showMailCompose = false}>Cancel</button>
        <button
          class="send-mail-btn"
          on:click={sendMailToDepartment}
          disabled={!mailSubject.trim() || !mailBody.trim() || sendingMail}
        >
          {#if sendingMail}
            <Loader size={12} class="spinning" />
          {:else}
            <Send size={12} />
          {/if}
          Send Mail
        </button>
      </div>
    </div>
  {/if}

  <!-- Active Tasks Banner -->
  {#if activeTasks.length > 0}
    <div class="active-tasks-banner" transition:slide={{ y: -10 }}>
      <div class="tasks-header">
        <Clock size={12} />
        <span>{activeTasks.length} Active Task{activeTasks.length !== 1 ? 's' : ''}</span>
      </div>
      <div class="tasks-list">
        {#each activeTasks.slice(0, 3) as task}
          <div class="task-item">
            <div class="task-dept" style="background: {DEPARTMENTS[task.departmentId].color}20; color: {DEPARTMENTS[task.departmentId].color}">
              {DEPARTMENTS[task.departmentId].name}
            </div>
            <div class="task-status" style="color: {getStatusColor(task.status)}">
              <svelte:component this={getStatusIcon(task.status)} size={10} />
            </div>
          </div>
        {/each}
        {#if activeTasks.length > 3}
          <div class="task-more">+{activeTasks.length - 3} more</div>
        {/if}
      </div>
    </div>
  {/if}

  <!-- Messages -->
  <div class="messages" bind:this={messagesContainer}>
    {#each messages as msg}
      <div class="message {msg.role}">
        <div class="message-avatar">
          {#if msg.role === "user"}
            <span>Y</span>
          {:else if msg.role === "floor_manager"}
            <Building2 size={14} />
          {:else}
            <AlertCircle size={14} />
          {/if}
        </div>
        <div class="message-body">
          {#if msg.role === "floor_manager"}
            <div class="message-label">{agentName}</div>
          {:else if msg.role === "system"}
            <div class="message-label system">System</div>
          {/if}
          <div class="message-text">{msg.content}</div>
          {#if msg.delegation}
            <div class="delegation-info">
              <ArrowRightCircle size={12} />
              <span>
                Delegated to <strong>{DEPARTMENTS[msg.delegation.departmentId].name}</strong>
              </span>
              <span class="delegation-status" style="color: {getStatusColor(msg.delegation.status)}">
                <svelte:component this={getStatusIcon(msg.delegation.status)} size={10} />
                {msg.delegation.status}
              </span>
            </div>
          {/if}
          <div class="message-time">{formatTime(msg.timestamp)}</div>
        </div>
      </div>
    {/each}

    {#if isLoading}
      <div class="message floor_manager">
        <div class="message-avatar">
          <Building2 size={14} />
        </div>
        <div class="message-body">
          <div class="typing-indicator">
            <Loader size={12} class="spinning" />
            <span>Coordinating...</span>
          </div>
        </div>
      </div>
    {/if}
  </div>

  <!-- Department Shortcuts -->
  <div class="department-shortcuts">
    <span class="shortcuts-label">Quick Delegate:</span>
    <div class="shortcuts-list">
      {#each departmentList as dept}
        <button
          class="dept-shortcut"
          style="--dept-color: {dept.color}"
          on:click={() => openDepartmentChat(dept.id)}
          title="{dept.name}: {dept.description}"
        >
          <span class="shortcut-dot"></span>
          {dept.name}
        </button>
      {/each}
    </div>
  </div>

  <!-- Quick Actions -->
  <div class="quick-actions">
    {#each quickDelegations.slice(0, 4) as action}
      <button
        class="quick-action-btn"
        on:click={() => delegateToDepartment(action.dept, action.task)}
        disabled={isLoading}
      >
        {action.label}
      </button>
    {/each}
  </div>

  <!-- Input -->
  <div class="input-area">
    <div class="input-wrapper">
      <textarea
        bind:this={textareaElement}
        bind:value={message}
        on:keydown={handleKeydown}
        on:input={autoResize}
        placeholder={placeholderText}
        rows="1"
        disabled={isLoading}
      ></textarea>
    </div>
    <div class="input-footer">
      <div class="model-selector-container">
        <AgentModelSelector
          agentId={isCopilot ? 'copilot' : 'floor_manager'}
          currentModel="opus"
        />
      </div>
      <div class="char-count">{message.length} / 4000</div>
      <button
        class="send-btn"
        on:click={sendMessage}
        disabled={!message.trim() || isLoading}
      >
        {#if isLoading}
          <Loader size={16} class="spinning" />
        {:else}
          <Send size={16} />
        {/if}
      </button>
    </div>
  </div>
</div>

<style>
  .floor-manager-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary, #0a0f1a);
    color: var(--text-primary, #e2e8f0);
    font-family: "Inter", sans-serif;
  }

  /* Header */
  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.625rem 0.75rem;
    background: var(--bg-secondary, #111827);
    border-bottom: 1px solid var(--border-color, #1e293b);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .manager-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 8px;
    color: white;
  }

  .manager-info {
    display: flex;
    flex-direction: column;
    gap: 0.125rem;
  }

  .manager-name {
    font-size: 0.875rem;
    font-weight: 600;
  }

  .manager-role {
    font-size: 0.625rem;
    color: var(--text-muted, #64748b);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .header-actions {
    display: flex;
    gap: 0.25rem;
  }

  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: transparent;
    border: none;
    border-radius: 4px;
    color: var(--text-muted, #64748b);
    cursor: pointer;
    transition: all 0.15s;
  }

  .icon-btn:hover {
    background: var(--bg-tertiary, #1e293b);
    color: var(--text-primary, #e2e8f0);
  }

  /* Active Tasks Banner */
  .active-tasks-banner {
    padding: 0.5rem 0.75rem;
    background: rgba(245, 158, 11, 0.1);
    border-bottom: 1px solid rgba(245, 158, 11, 0.2);
  }

  .tasks-header {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    font-size: 0.6875rem;
    font-weight: 600;
    color: #f59e0b;
    margin-bottom: 0.375rem;
  }

  .tasks-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.375rem;
  }

  .task-item {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.25rem 0.5rem;
    background: var(--bg-tertiary, #1e293b);
    border-radius: 0.25rem;
  }

  .task-dept {
    font-size: 0.625rem;
    font-weight: 500;
    padding: 0.125rem 0.375rem;
    border-radius: 0.125rem;
  }

  .task-status {
    display: flex;
    align-items: center;
  }

  .task-more {
    font-size: 0.625rem;
    color: var(--text-muted, #64748b);
    padding: 0.25rem;
  }

  /* Messages */
  .messages {
    flex: 1;
    overflow-y: auto;
    padding: 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .message {
    display: flex;
    gap: 0.5rem;
    max-width: 90%;
  }

  .message.user {
    align-self: flex-end;
    flex-direction: row-reverse;
  }

  .message.floor_manager,
  .message.system {
    align-self: flex-start;
  }

  .message-avatar {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-tertiary, #1e293b);
    flex-shrink: 0;
    font-size: 0.75rem;
    font-weight: 600;
  }

  .message.user .message-avatar {
    background: var(--accent-primary, #3b82f6);
    color: white;
  }

  .message.floor_manager .message-avatar {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
  }

  .message.system .message-avatar {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .message-body {
    flex: 1;
    min-width: 0;
  }

  .message-label {
    font-size: 0.625rem;
    font-weight: 600;
    color: var(--accent-primary, #6366f1);
    text-transform: uppercase;
    margin-bottom: 0.25rem;
  }

  .message-label.system {
    color: #ef4444;
  }

  .message-text {
    padding: 0.5rem 0.75rem;
    border-radius: 0.75rem;
    font-size: 0.8125rem;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .message.user .message-text {
    background: var(--accent-primary, #3b82f6);
    color: white;
    border-bottom-right-radius: 0.25rem;
  }

  .message.floor_manager .message-text {
    background: var(--bg-tertiary, #1e293b);
    border-bottom-left-radius: 0.25rem;
  }

  .message.system .message-text {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 0.5rem;
    font-size: 0.75rem;
  }

  .delegation-info {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    margin-top: 0.375rem;
    padding: 0.375rem 0.5rem;
    background: var(--bg-secondary, #111827);
    border-radius: 0.25rem;
    font-size: 0.6875rem;
  }

  .delegation-info strong {
    color: var(--accent-primary, #3b82f6);
  }

  .delegation-status {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    margin-left: auto;
    font-weight: 500;
  }

  .message-time {
    font-size: 0.625rem;
    color: var(--text-muted, #64748b);
    margin-top: 0.25rem;
  }

  .typing-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    color: var(--text-muted, #64748b);
    font-size: 0.75rem;
  }

  .spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  /* Department Shortcuts */
  .department-shortcuts {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    background: var(--bg-secondary, #111827);
    border-top: 1px solid var(--border-color, #1e293b);
    overflow-x: auto;
  }

  .shortcuts-label {
    font-size: 0.625rem;
    color: var(--text-muted, #64748b);
    flex-shrink: 0;
  }

  .shortcuts-list {
    display: flex;
    gap: 0.25rem;
  }

  .dept-shortcut {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.25rem 0.5rem;
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.25rem;
    color: var(--text-secondary, #94a3b8);
    font-size: 0.6875rem;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }

  .dept-shortcut:hover {
    border-color: var(--dept-color, #3b82f6);
    color: var(--text-primary, #e2e8f0);
  }

  .shortcut-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--dept-color, #3b82f6);
  }

  /* Quick Actions */
  .quick-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.375rem;
    padding: 0.5rem 0.75rem;
    border-top: 1px solid var(--border-color, #1e293b);
  }

  .quick-action-btn {
    padding: 0.375rem 0.625rem;
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.25rem;
    color: var(--text-secondary, #94a3b8);
    font-size: 0.6875rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  .quick-action-btn:hover:not(:disabled) {
    background: var(--bg-hover, #334155);
    border-color: var(--accent-primary, #3b82f6);
    color: var(--text-primary, #e2e8f0);
  }

  .quick-action-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  /* Input */
  .input-area {
    padding: 0.625rem 0.75rem;
    background: var(--bg-secondary, #111827);
    border-top: 1px solid var(--border-color, #1e293b);
  }

  .input-wrapper textarea {
    width: 100%;
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.5rem;
    padding: 0.5rem 0.75rem;
    color: var(--text-primary, #e2e8f0);
    font-size: 0.8125rem;
    font-family: inherit;
    resize: none;
    min-height: 40px;
    max-height: 120px;
    line-height: 1.4;
  }

  .input-wrapper textarea:focus {
    outline: none;
    border-color: var(--accent-primary, #3b82f6);
  }

  .input-wrapper textarea:disabled {
    opacity: 0.6;
  }

  .input-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 0.5rem;
  }

  .char-count {
    font-size: 0.625rem;
    color: var(--text-muted, #64748b);
  }

  .send-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: var(--accent-primary, #3b82f6);
    border: none;
    border-radius: 0.375rem;
    color: white;
    cursor: pointer;
    transition: all 0.15s;
  }

  .send-btn:hover:not(:disabled) {
    background: var(--accent-hover, #2563eb);
  }

  .send-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  /* Mail Compose Modal */
  .mail-compose-modal {
    background: var(--bg-secondary, #111827);
    border-bottom: 1px solid var(--border-color, #1e293b);
  }

  .mail-compose-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    background: var(--bg-tertiary, #1e293b);
    border-bottom: 1px solid var(--border-color, #334155);
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--accent-primary, #3b82f6);
  }

  .close-modal-btn {
    margin-left: auto;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    background: transparent;
    border: none;
    border-radius: 4px;
    color: var(--text-muted, #64748b);
    cursor: pointer;
  }

  .close-modal-btn:hover {
    background: var(--bg-hover, #334155);
    color: var(--text-primary, #e2e8f0);
  }

  .mail-compose-body {
    padding: 0.625rem 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.625rem;
  }

  .form-row {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .form-row label {
    font-size: 0.625rem;
    font-weight: 600;
    color: var(--text-muted, #64748b);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .dept-selector {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
  }

  .dept-option {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.25rem 0.5rem;
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.25rem;
    color: var(--text-secondary, #94a3b8);
    font-size: 0.6875rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  .dept-option:hover {
    border-color: var(--dept-color, #3b82f6);
  }

  .dept-option.selected {
    border-color: var(--dept-color, #3b82f6);
    background: color-mix(in srgb, var(--dept-color) 20%, var(--bg-tertiary));
    color: var(--text-primary, #e2e8f0);
  }

  .dept-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--dept-color, #3b82f6);
  }

  .priority-selector {
    display: flex;
    gap: 0.25rem;
  }

  .priority-option {
    padding: 0.25rem 0.625rem;
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.25rem;
    color: var(--text-secondary, #94a3b8);
    font-size: 0.625rem;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.15s;
  }

  .priority-option:hover {
    border-color: var(--priority-color, #3b82f6);
  }

  .priority-option.selected {
    border-color: var(--priority-color, #3b82f6);
    background: color-mix(in srgb, var(--priority-color) 20%, var(--bg-tertiary));
    color: var(--priority-color);
  }

  .form-row select,
  .form-row input,
  .form-row textarea {
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.25rem;
    padding: 0.375rem 0.5rem;
    color: var(--text-primary, #e2e8f0);
    font-size: 0.75rem;
    font-family: inherit;
  }

  .form-row select:focus,
  .form-row input:focus,
  .form-row textarea:focus {
    outline: none;
    border-color: var(--accent-primary, #3b82f6);
  }

  .form-row textarea {
    resize: vertical;
    min-height: 80px;
  }

  .mail-compose-footer {
    display: flex;
    justify-content: flex-end;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    background: var(--bg-tertiary, #1e293b);
    border-top: 1px solid var(--border-color, #334155);
  }

  .cancel-btn {
    padding: 0.375rem 0.75rem;
    background: var(--bg-secondary, #111827);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.25rem;
    color: var(--text-secondary, #94a3b8);
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  .cancel-btn:hover {
    background: var(--bg-hover, #334155);
  }

  .send-mail-btn {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.375rem 0.75rem;
    background: var(--accent-primary, #3b82f6);
    border: none;
    border-radius: 0.25rem;
    color: white;
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  .send-mail-btn:hover:not(:disabled) {
    background: var(--accent-hover, #2563eb);
  }

  .send-mail-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .mail-btn {
    position: relative;
  }
</style>

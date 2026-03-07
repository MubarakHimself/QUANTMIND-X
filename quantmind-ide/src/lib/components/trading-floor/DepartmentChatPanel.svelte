<script lang="ts">
  import { createEventDispatcher, onMount, tick } from "svelte";
  import { fly, fade } from "svelte/transition";
  import {
    Send,
    Loader,
    X,
    BarChart2,
    FlaskConical,
    Shield,
    Zap,
    Briefcase,
    ChevronRight,
    Bot,
    AlertCircle,
    CheckCircle2,
    Clock,
    ArrowRight,
  } from "lucide-svelte";
  import {
    departmentChatStore,
    activeDepartmentChat,
    activeDepartmentMessages,
    DEPARTMENTS,
    type DepartmentId,
    type DepartmentMessage,
    type DepartmentInfo,
  } from "$lib/stores/departmentChatStore";

  const dispatch = createEventDispatcher();

  // Props
  export let initialDepartment: DepartmentId | null = null;

  // Local state
  let message = "";
  let textareaElement: HTMLTextAreaElement;
  let messagesContainer: HTMLDivElement;
  let showDepartmentSelector = false;

  // Department icons mapping
  const departmentIcons: Record<DepartmentId, typeof BarChart2> = {
    development: BarChart2,
    research: FlaskConical,
    risk: Shield,
    trading: Zap,
    portfolio: Briefcase,
  };

  // Quick actions for each department
  const quickActions: Record<DepartmentId, Array<{ label: string; prompt: string }>> = {
    development: [
      { label: "Analyze Market", prompt: "Analyze the current market conditions for EURUSD" },
      { label: "Signal Check", prompt: "Check for any active trading signals" },
      { label: "Trend Analysis", prompt: "Perform trend analysis on major pairs" },
    ],
    research: [
      { label: "Run Backtest", prompt: "Run a backtest on the current strategy" },
      { label: "Optimize Params", prompt: "Optimize strategy parameters" },
      { label: "Compare Strategies", prompt: "Compare performance of active strategies" },
    ],
    risk: [
      { label: "Risk Report", prompt: "Generate current risk exposure report" },
      { label: "Position Check", prompt: "Check all open positions and risk levels" },
      { label: "Kelly Sizing", prompt: "Calculate Kelly-optimal position sizes" },
    ],
    trading: [
      { label: "Order Status", prompt: "Check status of pending orders" },
      { label: "Routing Check", prompt: "Verify broker connections and routing" },
      { label: "Execution Report", prompt: "Generate execution quality report" },
    ],
    portfolio: [
      { label: "Balance Report", prompt: "Generate portfolio balance report" },
      { label: "Rebalance", prompt: "Check if portfolio needs rebalancing" },
      { label: "Performance", prompt: "Show portfolio performance metrics" },
    ],
  };

  // Subscribe to store
  $: activeDept = $departmentChatStore.activeDepartment;
  $: activeChat = $activeDepartmentChat;
  $: messages = $activeDepartmentMessages;
  $: isLoading = $departmentChatStore.isLoading;
  $: isTyping = activeChat?.isTyping || false;
  $: activeDeptInfo = activeDept ? DEPARTMENTS[activeDept] : null;

  onMount(() => {
    if (initialDepartment) {
      departmentChatStore.setActiveDepartment(initialDepartment);
    }
  });

  // Auto-resize textarea
  function autoResize() {
    if (textareaElement) {
      textareaElement.style.height = "auto";
      const newHeight = Math.min(Math.max(textareaElement.scrollHeight, 60), 150);
      textareaElement.style.height = newHeight + "px";
    }
  }

  // Select department
  function selectDepartment(deptId: DepartmentId) {
    departmentChatStore.setActiveDepartment(deptId);
    showDepartmentSelector = false;
  }

  // Send message
  async function sendMessage() {
    if (!message.trim() || !activeDept || isLoading) return;

    const content = message.trim();
    message = "";

    await tick();
    scrollToBottom();

    await departmentChatStore.sendMessage(activeDept, content);

    await tick();
    scrollToBottom();
  }

  // Execute quick action
  async function executeQuickAction(prompt: string) {
    if (!activeDept || isLoading) return;

    message = "";
    await departmentChatStore.sendMessage(activeDept, prompt);

    await tick();
    scrollToBottom();
  }

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
  function formatTime(date: Date | null | undefined): string {
    if (!date) return "";
    const d = new Date(date);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  // Get status icon
  function getStatusIcon(status?: string): typeof CheckCircle2 {
    switch (status) {
      case "completed":
        return CheckCircle2;
      case "error":
        return AlertCircle;
      case "pending":
      case "in_progress":
        return Clock;
      default:
        return CheckCircle2;
    }
  }

  // Get status color
  function getStatusColor(status?: string): string {
    switch (status) {
      case "completed":
        return "#22c55e";
      case "error":
        return "#ef4444";
      case "pending":
      case "in_progress":
        return "#f59e0b";
      default:
        return "#64748b";
    }
  }

  // Close panel
  function closePanel() {
    dispatch("close");
  }
</script>

<div class="department-chat-panel">
  <!-- Header -->
  <div class="panel-header">
    <div class="header-left">
      {#if activeDeptInfo}
        <button
          class="department-selector-btn"
          on:click={() => (showDepartmentSelector = !showDepartmentSelector)}
        >
          <div
            class="dept-icon"
            style="background: {activeDeptInfo.color}20; color: {activeDeptInfo.color}"
          >
            <svelte:component this={departmentIcons[activeDeptInfo.id]} size={16} />
          </div>
          <span class="dept-name">{activeDeptInfo.name}</span>
          <span class="chevron" class:rotated={showDepartmentSelector}>
            <ChevronRight size={14} />
          </span>
        </button>
      {:else}
        <button
          class="department-selector-btn placeholder"
          on:click={() => (showDepartmentSelector = !showDepartmentSelector)}
        >
          <Bot size={16} />
          <span>Select Department</span>
          <span class="chevron" class:rotated={showDepartmentSelector}>
            <ChevronRight size={14} />
          </span>
        </button>
      {/if}
    </div>

    <div class="header-actions">
      <button class="icon-btn" title="Close" on:click={closePanel}>
        <X size={16} />
      </button>
    </div>
  </div>

  <!-- Department Selector Dropdown -->
  {#if showDepartmentSelector}
    <div class="department-selector" transition:fly={{ y: -10, duration: 150 }}>
      {#each Object.values(DEPARTMENTS) as dept}
        <button
          class="dept-option"
          class:active={activeDept === dept.id}
          on:click={() => selectDepartment(dept.id)}
        >
          <div class="dept-icon-small" style="background: {dept.color}20; color: {dept.color}">
            <svelte:component this={departmentIcons[dept.id]} size={14} />
          </div>
          <div class="dept-info">
            <span class="dept-label">{dept.name}</span>
            <span class="dept-desc">{dept.description}</span>
          </div>
        </button>
      {/each}
    </div>
  {/if}

  <!-- Messages Area -->
  <div class="messages" bind:this={messagesContainer}>
    {#if !activeDept}
      <div class="empty-state">
        <Bot size={32} />
        <p>Select a department to start chatting</p>
      </div>
    {:else if messages.length === 0}
      <div class="empty-state">
        <div
          class="dept-icon-large"
          style="background: {activeDeptInfo?.color}20; color: {activeDeptInfo?.color}"
        >
          <svelte:component this={departmentIcons[activeDept]} size={32} />
        </div>
        <h4>{activeDeptInfo?.name} Department</h4>
        <p>{activeDeptInfo?.description}</p>
      </div>
    {:else}
      {#each messages as msg}
        <div class="message {msg.role}">
          <div class="message-content">
            {#if msg.role === "department"}
              <div class="message-header">
                <span
                  class="status-indicator"
                  style="color: {getStatusColor(msg.metadata?.status)}"
                >
                  <svelte:component this={getStatusIcon(msg.metadata?.status)} size={12} />
                </span>
                <span class="dept-badge" style="background: {activeDeptInfo?.color}20; color: {activeDeptInfo?.color}">
                  {activeDeptInfo?.name}
                </span>
              </div>
            {:else if msg.role === "system"}
              <div class="message-header">
                <span class="system-badge">
                  <ArrowRight size={10} />
                  Delegated
                </span>
              </div>
            {/if}
            <div class="message-text">
              {msg.content}
            </div>
            <div class="message-time">
              {formatTime(msg.timestamp)}
            </div>
          </div>
        </div>
      {/each}

      {#if isTyping}
        <div class="message department">
          <div class="message-content">
            <div class="typing-indicator">
              <Loader size={14} class="spinning" />
              <span>{activeDeptInfo?.name} is responding...</span>
            </div>
          </div>
        </div>
      {/if}
    {/if}
  </div>

  <!-- Quick Actions -->
  {#if activeDept && !isLoading}
    <div class="quick-actions">
      {#each quickActions[activeDept] as action}
        <button
          class="quick-action-btn"
          on:click={() => executeQuickAction(action.prompt)}
        >
          {action.label}
        </button>
      {/each}
    </div>
  {/if}

  <!-- Input Area -->
  <div class="input-area">
    <div class="input-wrapper">
      <textarea
        bind:this={textareaElement}
        bind:value={message}
        on:keydown={handleKeydown}
        on:input={autoResize}
        placeholder={activeDept ? `Message ${activeDeptInfo?.name}...` : "Select a department first..."}
        rows="1"
        disabled={!activeDept || isLoading}
      ></textarea>
    </div>

    <div class="input-footer">
      <div class="char-count">
        {message.length} / 2000
      </div>
      <button
        class="send-btn"
        on:click={sendMessage}
        disabled={!message.trim() || !activeDept || isLoading}
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
  .department-chat-panel {
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
    padding: 0.5rem 0.75rem;
    background: var(--bg-secondary, #111827);
    border-bottom: 1px solid var(--border-color, #1e293b);
    min-height: 44px;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .department-selector-btn {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.375rem 0.5rem;
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.375rem;
    color: var(--text-primary, #e2e8f0);
    font-size: 0.8125rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
  }

  .department-selector-btn:hover {
    background: var(--bg-hover, #334155);
  }

  .department-selector-btn.placeholder {
    color: var(--text-muted, #64748b);
  }

  .dept-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 4px;
  }

  .dept-name {
    font-weight: 600;
  }

  .chevron {
    transition: transform 0.2s;
  }

  .chevron.rotated {
    transform: rotate(90deg);
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

  /* Department Selector Dropdown */
  .department-selector {
    position: absolute;
    top: 44px;
    left: 0.5rem;
    right: 0.5rem;
    background: var(--bg-secondary, #111827);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.5rem;
    padding: 0.375rem;
    z-index: 100;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
  }

  .dept-option {
    display: flex;
    align-items: center;
    gap: 0.625rem;
    width: 100%;
    padding: 0.5rem 0.625rem;
    background: transparent;
    border: none;
    border-radius: 0.375rem;
    color: var(--text-secondary, #94a3b8);
    font-size: 0.8125rem;
    cursor: pointer;
    transition: all 0.15s;
    text-align: left;
  }

  .dept-option:hover {
    background: var(--bg-tertiary, #1e293b);
    color: var(--text-primary, #e2e8f0);
  }

  .dept-option.active {
    background: var(--bg-tertiary, #1e293b);
    color: var(--text-primary, #e2e8f0);
  }

  .dept-icon-small {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 4px;
    flex-shrink: 0;
  }

  .dept-info {
    display: flex;
    flex-direction: column;
    gap: 0.125rem;
  }

  .dept-label {
    font-weight: 600;
  }

  .dept-desc {
    font-size: 0.6875rem;
    color: var(--text-muted, #64748b);
  }

  /* Messages */
  .messages {
    flex: 1;
    overflow-y: auto;
    padding: 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.625rem;
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex: 1;
    color: var(--text-muted, #64748b);
    text-align: center;
    gap: 0.75rem;
    padding: 2rem;
  }

  .empty-state h4 {
    margin: 0;
    font-size: 1rem;
    color: var(--text-secondary, #94a3b8);
  }

  .empty-state p {
    margin: 0;
    font-size: 0.8125rem;
  }

  .dept-icon-large {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 56px;
    height: 56px;
    border-radius: 12px;
  }

  .message {
    display: flex;
    flex-direction: column;
    max-width: 90%;
  }

  .message.user {
    align-self: flex-end;
  }

  .message.department,
  .message.system {
    align-self: flex-start;
  }

  .message-content {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .message-header {
    display: flex;
    align-items: center;
    gap: 0.375rem;
  }

  .status-indicator {
    display: flex;
    align-items: center;
  }

  .dept-badge,
  .system-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.125rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.6875rem;
    font-weight: 600;
    text-transform: uppercase;
  }

  .system-badge {
    background: rgba(99, 102, 241, 0.2);
    color: #818cf8;
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

  .message.department .message-text {
    background: var(--bg-tertiary, #1e293b);
    border-bottom-left-radius: 0.25rem;
  }

  .message.system .message-text {
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-bottom-left-radius: 0.25rem;
    font-size: 0.75rem;
  }

  .message-time {
    font-size: 0.625rem;
    color: var(--text-muted, #64748b);
    padding-left: 0.5rem;
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

  /* Quick Actions */
  .quick-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.375rem;
    padding: 0.5rem 0.75rem;
    border-top: 1px solid var(--border-color, #1e293b);
    background: var(--bg-secondary, #111827);
  }

  .quick-action-btn {
    padding: 0.25rem 0.5rem;
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.25rem;
    color: var(--text-secondary, #94a3b8);
    font-size: 0.6875rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  .quick-action-btn:hover {
    background: var(--bg-hover, #334155);
    border-color: var(--accent-primary, #3b82f6);
    color: var(--text-primary, #e2e8f0);
  }

  /* Input Area */
  .input-area {
    padding: 0.625rem 0.75rem;
    background: var(--bg-secondary, #111827);
    border-top: 1px solid var(--border-color, #1e293b);
  }

  .input-wrapper {
    position: relative;
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
    cursor: not-allowed;
  }

  .input-wrapper textarea::placeholder {
    color: var(--text-muted, #64748b);
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
</style>

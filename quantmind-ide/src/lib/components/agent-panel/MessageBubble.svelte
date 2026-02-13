<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { fade } from 'svelte/transition';
  import { Copy, Check, RefreshCw, Edit3, MoreHorizontal } from 'lucide-svelte';
  import type { Message } from '../../stores/chatStore';
  
  // Props
  export let message: Message;
  export let agent: { id: string; name: string; icon: string; description: string } | undefined;
  export let AgentIcon: any;
  export let showHeader: boolean = true;
  export let showTimestamp: boolean = true;
  
  const dispatch = createEventDispatcher();
  
  // State
  let isHovered = false;
  let copied = false;
  let showActions = false;
  
  // Format timestamp
  function formatTime(date: Date): string {
    return new Date(date).toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
  }
  
  // Format message content (basic markdown)
  function formatContent(content: string): string {
    // Bold
    let formatted = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Italic
    formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Code blocks
    formatted = formatted.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>');
    // Inline code
    formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Line breaks
    formatted = formatted.replace(/\n/g, '<br>');
    
    return formatted;
  }
  
  // Copy message content
  async function copyContent() {
    try {
      await navigator.clipboard.writeText(message.content);
      copied = true;
      setTimeout(() => copied = false, 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }
  
  // Regenerate response
  function regenerate() {
    dispatch('regenerate', message.id);
  }
  
  // Edit message
  function edit() {
    dispatch('edit', message);
  }
  
  // Get token count display
  function getTokenDisplay(): string | null {
    if (!message.tokenCount) return null;
    if (message.tokenCount > 1000) {
      return `${(message.tokenCount / 1000).toFixed(1)}k tokens`;
    }
    return `${message.tokenCount} tokens`;
  }
  
  // Get latency display
  function getLatencyDisplay(): string | null {
    if (!message.latency) return null;
    if (message.latency > 1000) {
      return `${(message.latency / 1000).toFixed(1)}s`;
    }
    return `${Math.round(message.latency)}ms`;
  }
</script>

<div 
  class="message-bubble {message.role}"
  on:mouseenter={() => isHovered = true}
  on:mouseleave={() => isHovered = false}
  role="article"
  aria-label="{message.role === 'user' ? 'Your message' : 'Assistant response'}"
>
  <!-- Avatar for assistant messages -->
  {#if message.role === 'assistant' && showHeader}
    <div class="avatar">
      <svelte:component this={AgentIcon} size={14} />
    </div>
  {/if}
  
  <!-- Message content -->
  <div class="message-content">
    <!-- Header -->
    {#if showHeader}
      <div class="message-header">
        {#if message.role === 'assistant'}
          <span class="sender-name">{agent?.name || 'Assistant'}</span>
        {:else}
          <span class="sender-name">You</span>
        {/if}
        {#if showTimestamp}
          <span class="timestamp">{formatTime(message.timestamp)}</span>
        {/if}
      </div>
    {/if}
    
    <!-- Body -->
    <div class="message-body">
      {@html formatContent(message.content)}
    </div>
    
    <!-- Footer with metadata -->
    {#if message.role === 'assistant' && (message.tokenCount || message.latency)}
      <div class="message-footer">
        {#if getTokenDisplay()}
          <span class="metadata">{getTokenDisplay()}</span>
        {/if}
        {#if getLatencyDisplay()}
          <span class="metadata">{getLatencyDisplay()}</span>
        {/if}
        {#if message.model}
          <span class="metadata model">{message.model}</span>
        {/if}
      </div>
    {/if}
  </div>
  
  <!-- Hover actions -->
  {#if isHovered}
    <div class="message-actions" transition:fade={{ duration: 100 }}>
      {#if message.role === 'assistant'}
        <button 
          class="action-btn" 
          on:click={copyContent}
          title="Copy"
          aria-label="Copy message"
        >
          {#if copied}
            <Check size={14} class="success" />
          {:else}
            <Copy size={14} />
          {/if}
        </button>
        <button 
          class="action-btn" 
          on:click={regenerate}
          title="Regenerate"
          aria-label="Regenerate response"
        >
          <RefreshCw size={14} />
        </button>
      {:else}
        <button 
          class="action-btn" 
          on:click={edit}
          title="Edit"
          aria-label="Edit message"
        >
          <Edit3 size={14} />
        </button>
        <button 
          class="action-btn" 
          on:click={copyContent}
          title="Copy"
          aria-label="Copy message"
        >
          {#if copied}
            <Check size={14} class="success" />
          {:else}
            <Copy size={14} />
          {/if}
        </button>
      {/if}
    </div>
  {/if}
</div>

<style>
  .message-bubble {
    display: flex;
    gap: 10px;
    padding: 8px 16px;
    position: relative;
    transition: background 0.15s;
  }
  
  .message-bubble:hover {
    background: var(--bg-primary);
  }
  
  .message-bubble.user {
    flex-direction: row-reverse;
  }
  
  /* Avatar */
  .avatar {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: var(--accent-primary);
    border-radius: 6px;
    color: var(--bg-primary);
    flex-shrink: 0;
  }
  
  /* Message Content */
  .message-content {
    flex: 1;
    min-width: 0;
    max-width: 85%;
  }
  
  .message-bubble.user .message-content {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
  }
  
  /* Header */
  .message-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
  }
  
  .sender-name {
    font-size: 11px;
    font-weight: 600;
    color: var(--accent-primary);
  }
  
  .message-bubble.user .sender-name {
    color: var(--text-secondary);
  }
  
  .timestamp {
    font-size: 10px;
    color: var(--text-muted);
  }
  
  /* Body */
  .message-body {
    font-size: 13px;
    line-height: 1.6;
    color: var(--text-primary);
    word-wrap: break-word;
  }
  
  .message-bubble.user .message-body {
    background: var(--accent-primary);
    color: var(--bg-primary);
    padding: 10px 14px;
    border-radius: 12px;
    border-top-right-radius: 4px;
  }
  
  .message-bubble.assistant .message-body {
    background: var(--bg-tertiary);
    padding: 10px 14px;
    border-radius: 12px;
    border-top-left-radius: 4px;
  }
  
  /* Code blocks */
  .message-body :global(pre) {
    background: var(--bg-primary);
    border-radius: 8px;
    padding: 12px;
    overflow-x: auto;
    margin: 8px 0;
  }
  
  .message-body :global(code) {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
  }
  
  .message-body :global(code:not(pre code)) {
    background: var(--bg-primary);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 11px;
  }
  
  /* Footer */
  .message-footer {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 6px;
    padding-left: 4px;
  }
  
  .metadata {
    font-size: 10px;
    color: var(--text-muted);
  }
  
  .metadata.model {
    background: var(--bg-secondary);
    padding: 2px 6px;
    border-radius: 4px;
  }
  
  /* Actions */
  .message-actions {
    display: flex;
    align-items: center;
    gap: 2px;
    position: absolute;
    top: 8px;
    right: 16px;
    background: var(--bg-secondary);
    border-radius: 6px;
    padding: 2px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  }
  
  .message-bubble.user .message-actions {
    right: auto;
    left: 16px;
  }
  
  .action-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: transparent;
    border: none;
    border-radius: 4px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s;
  }
  
  .action-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }
</style>

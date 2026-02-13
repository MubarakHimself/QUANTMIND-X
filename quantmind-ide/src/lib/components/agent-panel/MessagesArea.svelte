<script lang="ts">
  import { onMount, tick, afterUpdate } from 'svelte';
  import { fade, fly } from 'svelte/transition';
  import { Bot, Code, Wand2, Loader } from 'lucide-svelte';
  import MessageBubble from './MessageBubble.svelte';
  import type { Message, AgentType } from '../../stores/chatStore';
  import { agentGreetings } from '../../stores/chatStore';
  
  // Props
  export let messages: Message[];
  export let agent: AgentType;
  export let agents: Array<{ id: AgentType; name: string; icon: string; description: string }>;
  
  // State
  let messagesContainer: HTMLDivElement;
  let isLoading = false;
  let shouldAutoScroll = true;
  
  // Icon mapping
  const iconMap: Record<string, any> = {
    Bot,
    Code,
    Wand2
  };
  
  // Get current agent info
  $: currentAgent = agents.find(a => a.id === agent);
  $: AgentIcon = iconMap[currentAgent?.icon || 'Bot'];
  
  // Auto-scroll to bottom
  async function scrollToBottom(smooth = true) {
    await tick();
    if (messagesContainer) {
      messagesContainer.scrollTo({
        top: messagesContainer.scrollHeight,
        behavior: smooth ? 'smooth' : 'auto'
      });
    }
  }
  
  // Handle scroll to detect if user scrolled up
  function handleScroll() {
    if (!messagesContainer) return;
    const { scrollTop, scrollHeight, clientHeight } = messagesContainer;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;
    shouldAutoScroll = isAtBottom;
  }
  
  // Scroll to bottom on new messages
  afterUpdate(() => {
    if (shouldAutoScroll) {
      scrollToBottom();
    }
  });
  
  // Format timestamp
  function formatTime(date: Date): string {
    return new Date(date).toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
  }
  
  // Group messages by date
  function groupMessagesByDate(messages: Message[]): Map<string, Message[]> {
    const groups = new Map<string, Message[]>();
    
    messages.forEach(msg => {
      const date = new Date(msg.timestamp).toDateString();
      if (!groups.has(date)) {
        groups.set(date, []);
      }
      groups.get(date)!.push(msg);
    });
    
    return groups;
  }
  
  // Format date header
  function formatDateHeader(dateString: string): string {
    const date = new Date(dateString);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    
    if (date.toDateString() === today.toDateString()) {
      return 'Today';
    } else if (date.toDateString() === yesterday.toDateString()) {
      return 'Yesterday';
    } else {
      return date.toLocaleDateString('en-US', {
        weekday: 'long',
        month: 'short',
        day: 'numeric'
      });
    }
  }
  
  // Check if messages are from same sender within 1 minute
  function shouldGroupWithPrevious(current: Message, previous: Message | null): boolean {
    if (!previous) return false;
    if (current.role !== previous.role) return false;
    
    const currentTime = new Date(current.timestamp).getTime();
    const previousTime = new Date(previous.timestamp).getTime();
    const diffMinutes = (currentTime - previousTime) / (1000 * 60);
    
    return diffMinutes < 1;
  }
</script>

<div class="messages-area" role="log" aria-label="Chat messages">
  <div 
    class="messages-container" 
    bind:this={messagesContainer}
    on:scroll={handleScroll}
  >
    {#if messages.length === 0}
      <!-- Empty State / Welcome Message -->
      <div class="welcome-state" transition:fade>
        <div class="welcome-icon">
          <svelte:component this={AgentIcon} size={32} />
        </div>
        <h3 class="welcome-title">{currentAgent?.name || 'Agent'}</h3>
        <p class="welcome-description">{currentAgent?.description || 'AI Assistant'}</p>
        <p class="welcome-greeting">{agentGreetings[agent]}</p>
        
        <div class="quick-actions">
          <button class="quick-action-btn">
            📊 Run a backtest
          </button>
          <button class="quick-action-btn">
            📝 Analyze a strategy
          </button>
          <button class="quick-action-btn">
            🔧 Debug code
          </button>
        </div>
      </div>
    {:else}
      <!-- Messages grouped by date -->
      {#each Array.from(groupMessagesByDate(messages).entries()) as [date, dateMessages], dateIndex}
        <div class="date-group">
          <div class="date-header">
            <span>{formatDateHeader(date)}</span>
          </div>
          
          {#each dateMessages as message, msgIndex}
            {@const previousMessage = msgIndex > 0 ? dateMessages[msgIndex - 1] : null}
            {@const groupWithPrevious = shouldGroupWithPrevious(message, previousMessage)}
            
            <MessageBubble
              {message}
              agent={currentAgent}
              AgentIcon={AgentIcon}
              showHeader={!groupWithPrevious}
              showTimestamp={!groupWithPrevious}
            />
          {/each}
        </div>
      {/each}
      
      <!-- Loading indicator -->
      {#if isLoading}
        <div class="loading-indicator" transition:fly={{ y: 20, duration: 200 }}>
          <div class="loading-avatar">
            <svelte:component this={AgentIcon} size={14} />
          </div>
          <div class="loading-content">
            <div class="loading-dots">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        </div>
      {/if}
    {/if}
  </div>
  
  <!-- Scroll to bottom button -->
  {#if !shouldAutoScroll && messages.length > 0}
    <button 
      class="scroll-to-bottom" 
      on:click={() => scrollToBottom()}
      transition:fade
      aria-label="Scroll to latest messages"
    >
      ↓ New messages
    </button>
  {/if}
</div>

<style>
  .messages-area {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    position: relative;
  }
  
  .messages-container {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  
  /* Welcome State */
  .welcome-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 32px 24px;
    flex: 1;
  }
  
  .welcome-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 64px;
    height: 64px;
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
    border-radius: 16px;
    color: var(--bg-primary);
    margin-bottom: 16px;
  }
  
  .welcome-title {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 4px;
  }
  
  .welcome-description {
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 16px;
  }
  
  .welcome-greeting {
    font-size: 13px;
    color: var(--text-secondary);
    max-width: 280px;
    line-height: 1.6;
    margin-bottom: 24px;
  }
  
  .quick-actions {
    display: flex;
    flex-direction: column;
    gap: 8px;
    width: 100%;
    max-width: 240px;
  }
  
  .quick-action-btn {
    padding: 10px 16px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
    text-align: left;
  }
  
  .quick-action-btn:hover {
    background: var(--bg-secondary);
    border-color: var(--accent-primary);
  }
  
  /* Date Groups */
  .date-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  
  .date-header {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 16px 0 8px;
  }
  
  .date-header span {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    background: var(--bg-secondary);
    padding: 4px 12px;
    border-radius: 12px;
  }
  
  /* Loading Indicator */
  .loading-indicator {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 12px 16px;
    animation: fadeIn 0.3s ease;
  }
  
  .loading-avatar {
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
  
  .loading-content {
    display: flex;
    align-items: center;
    padding: 12px 16px;
    background: var(--bg-tertiary);
    border-radius: 12px;
    border-top-left-radius: 4px;
  }
  
  .loading-dots {
    display: flex;
    gap: 4px;
  }
  
  .loading-dots span {
    width: 6px;
    height: 6px;
    background: var(--text-muted);
    border-radius: 50%;
    animation: bounce 1.4s infinite ease-in-out both;
  }
  
  .loading-dots span:nth-child(1) {
    animation-delay: -0.32s;
  }
  
  .loading-dots span:nth-child(2) {
    animation-delay: -0.16s;
  }
  
  @keyframes bounce {
    0%, 80%, 100% {
      transform: scale(0);
    }
    40% {
      transform: scale(1);
    }
  }
  
  @keyframes fadeIn {
    from {
      opacity: 0;
      transform: translateY(10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
  
  /* Scroll to Bottom Button */
  .scroll-to-bottom {
    position: absolute;
    bottom: 16px;
    left: 50%;
    transform: translateX(-50%);
    padding: 8px 16px;
    background: var(--accent-primary);
    border: none;
    border-radius: 20px;
    color: var(--bg-primary);
    font-size: 11px;
    font-weight: 500;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    transition: all 0.15s;
  }
  
  .scroll-to-bottom:hover {
    background: var(--accent-secondary);
    transform: translateX(-50%) scale(1.05);
  }
</style>

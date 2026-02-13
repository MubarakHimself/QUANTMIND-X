<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { fade } from 'svelte/transition';
  import { Pin, Trash2, MessageSquare } from 'lucide-svelte';
  import type { Chat } from '../../stores/chatStore';
  
  // Props
  export let chat: Chat;
  export let isActive: boolean = false;
  
  const dispatch = createEventDispatcher();
  
  // State
  let isHovered = false;
  
  // Format date for display
  function formatDate(date: Date): string {
    const now = new Date();
    const chatDate = new Date(date);
    const diffDays = Math.floor((now.getTime() - chatDate.getTime()) / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays}d ago`;
    return chatDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  }
  
  // Get message count
  function getMessageCount(): number {
    return chat.messages.length;
  }
  
  // Handle click
  function handleClick() {
    dispatch('select');
  }
  
  // Handle context menu
  function handleContextMenu(e: MouseEvent) {
    dispatch('contextMenu', e);
  }
  
  // Handle pin click
  function handlePinClick(e: MouseEvent) {
    e.stopPropagation();
    dispatch('pin');
  }
  
  // Handle delete click
  function handleDeleteClick(e: MouseEvent) {
    e.stopPropagation();
    dispatch('delete');
  }
</script>

<!-- svelte-ignore a11y-no-static-element-interactions a11y-click-events-have-key-events -->
<div 
  class="chat-item"
  class:active={isActive}
  class:pinned={chat.isPinned}
  on:click={handleClick}
  on:contextmenu={handleContextMenu}
  on:keydown={(e) => e.key === 'Enter' && handleClick()}
  on:mouseenter={() => isHovered = true}
  on:mouseleave={() => isHovered = false}
  role="button"
  tabindex="0"
  aria-pressed={isActive}
  aria-label="Chat: {chat.title}"
>
  <!-- Active indicator -->
  {#if isActive}
    <div class="active-indicator" transition:fade={{ duration: 100 }}></div>
  {/if}
  
  <!-- Chat icon -->
  <div class="chat-icon">
    <MessageSquare size={14} />
  </div>
  
  <!-- Chat info -->
  <div class="chat-info">
    <span class="chat-title" title={chat.title}>{chat.title}</span>
    <div class="chat-meta">
      <span class="chat-date">{formatDate(chat.lastMessageAt)}</span>
      <span class="chat-count">{getMessageCount()} messages</span>
    </div>
  </div>
  
  <!-- Pin indicator -->
  {#if chat.isPinned}
    <div class="pin-indicator">
      <Pin size={12} />
    </div>
  {/if}
  
  <!-- Hover actions -->
  {#if isHovered}
    <div class="hover-actions" transition:fade={{ duration: 100 }}>
      <button 
        class="action-btn pin-btn" 
        on:click={handlePinClick}
        title={chat.isPinned ? 'Unpin' : 'Pin'}
        aria-label={chat.isPinned ? 'Unpin chat' : 'Pin chat'}
      >
        <Pin size={12} />
      </button>
      <button 
        class="action-btn delete-btn" 
        on:click={handleDeleteClick}
        title="Delete"
        aria-label="Delete chat"
      >
        <Trash2 size={12} />
      </button>
    </div>
  {/if}
</div>

<style>
  .chat-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border-radius: 8px;
    cursor: pointer;
    position: relative;
    transition: all 0.15s ease;
    background: transparent;
    border: 1px solid transparent;
  }
  
  .chat-item:hover {
    background: var(--bg-tertiary);
  }
  
  .chat-item.active {
    background: var(--bg-tertiary);
    border-color: var(--border-subtle);
  }
  
  .chat-item.active:hover {
    background: var(--bg-secondary);
  }
  
  .active-indicator {
    position: absolute;
    left: 0;
    top: 4px;
    bottom: 4px;
    width: 3px;
    background: var(--accent-primary);
    border-radius: 0 2px 2px 0;
  }
  
  .chat-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: var(--bg-secondary);
    border-radius: 6px;
    color: var(--text-muted);
    flex-shrink: 0;
  }
  
  .chat-item.active .chat-icon {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }
  
  .chat-info {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  
  .chat-title {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  
  .chat-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 10px;
    color: var(--text-muted);
  }
  
  .chat-date {
    opacity: 0.8;
  }
  
  .chat-count {
    opacity: 0.6;
  }
  
  .pin-indicator {
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--accent-primary);
    opacity: 0.7;
  }
  
  .hover-actions {
    display: flex;
    align-items: center;
    gap: 2px;
    position: absolute;
    right: 8px;
    top: 50%;
    transform: translateY(-50%);
    background: var(--bg-secondary);
    border-radius: 6px;
    padding: 2px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
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
  
  .delete-btn:hover {
    color: var(--accent-danger);
    background: rgba(239, 68, 68, 0.1);
  }
  
  .pin-btn:hover {
    color: var(--accent-primary);
  }
  
  /* Pinned state */
  .chat-item.pinned {
    background: rgba(107, 33, 168, 0.1);
  }
  
  .chat-item.pinned:hover {
    background: rgba(107, 33, 168, 0.15);
  }
</style>

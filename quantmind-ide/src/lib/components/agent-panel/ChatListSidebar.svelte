<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import { fade, slide } from 'svelte/transition';
  import { Search, Plus, Pin, Trash2, MoreVertical } from 'lucide-svelte';
  import ChatItem from './ChatItem.svelte';
  import { chatStore, pinnedChats, unpinnedChats } from '../../stores/chatStore';
  import type { Chat, AgentType } from '../../stores/chatStore';
  
  // State
  let searchQuery = '';
  let filteredChats: Chat[] = [];
  let showContextMenu = false;
  let contextMenuChat: Chat | null = null;
  let contextMenuPosition = { x: 0, y: 0 };
  
  // Reactive filtering
  $: allPinned = $pinnedChats;
  $: allUnpinned = $unpinnedChats;
  $: currentAgent = $chatStore.activeAgent;
  
  $: filteredPinned = searchQuery.trim()
    ? allPinned.filter(chat => 
        chat.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        chat.messages.some(m => m.content.toLowerCase().includes(searchQuery.toLowerCase()))
      )
    : allPinned;
    
  $: filteredUnpinned = searchQuery.trim()
    ? allUnpinned.filter(chat => 
        chat.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        chat.messages.some(m => m.content.toLowerCase().includes(searchQuery.toLowerCase()))
      )
    : allUnpinned;
  
  $: activeChatId = $chatStore.activeChatId;
  
  // Handle chat selection
  function handleSelectChat(chatId: string) {
    chatStore.selectChat(chatId);
  }
  
  // Handle new chat
  function handleNewChat() {
    chatStore.createChat(currentAgent);
  }
  
  // Handle pin toggle
  function handlePinToggle(chatId: string) {
    chatStore.togglePinChat(chatId);
    closeContextMenu();
  }
  
  // Handle delete
  function handleDelete(chatId: string) {
    if (confirm('Are you sure you want to delete this chat?')) {
      chatStore.deleteChat(chatId);
    }
    closeContextMenu();
  }
  
  // Context menu handling
  function openContextMenu(e: MouseEvent, chat: Chat) {
    e.preventDefault();
    contextMenuChat = chat;
    contextMenuPosition = { x: e.clientX, y: e.clientY };
    showContextMenu = true;
  }
  
  function closeContextMenu() {
    showContextMenu = false;
    contextMenuChat = null;
  }
  
  // Click outside to close context menu
  function handleClickOutside() {
    if (showContextMenu) {
      closeContextMenu();
    }
  }

  // Helper to get chat ID safely (avoids non-null assertion in template)
  function getChatId(chat: Chat | null): string {
    return chat?.id || '';
  }
  
  // Format date for display
  function formatDate(date: Date): string {
    const now = new Date();
    const chatDate = new Date(date);
    const diffDays = Math.floor((now.getTime() - chatDate.getTime()) / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    return chatDate.toLocaleDateString();
  }
</script>

<svelte:window on:click={handleClickOutside} />

<div class="chat-list-sidebar" role="navigation" aria-label="Chat history">
  <!-- Search Header -->
  <div class="sidebar-header">
    <div class="search-container">
      <Search size={14} class="search-icon" />
      <input
        type="text"
        placeholder="Search chats..."
        bind:value={searchQuery}
        aria-label="Search chats"
      />
    </div>
    <button 
      class="new-chat-btn" 
      on:click={handleNewChat}
      title="New Chat"
      aria-label="Create new chat"
    >
      <Plus size={16} />
    </button>
  </div>
  
  <!-- Chat List -->
  <div class="chat-list" role="list">
    <!-- Pinned Chats Section -->
    {#if filteredPinned.length > 0}
      <div class="section-divider">
        <span>Pinned</span>
      </div>
      {#each filteredPinned as chat (chat.id)}
        <ChatItem
          {chat}
          isActive={activeChatId === chat.id}
          on:select={() => handleSelectChat(chat.id)}
          on:contextmenu={(e) => openContextMenu(e.detail as MouseEvent, chat)}
        />
      {/each}
    {/if}
    
    <!-- Unpinned Chats Section -->
    {#if filteredUnpinned.length > 0}
      {#if filteredPinned.length > 0}
        <div class="section-divider unpinned">
          <span>Recent</span>
        </div>
      {/if}
      {#each filteredUnpinned as chat (chat.id)}
        <ChatItem
          {chat}
          isActive={activeChatId === chat.id}
          on:select={() => handleSelectChat(chat.id)}
          on:contextmenu={(e) => openContextMenu(e.detail as MouseEvent, chat)}
        />
      {/each}
    {/if}
    
    <!-- Empty State -->
    {#if filteredPinned.length === 0 && filteredUnpinned.length === 0}
      <div class="empty-state" transition:fade>
        <div class="empty-icon">💬</div>
        <p class="empty-title">No chats found</p>
        <p class="empty-description">
          {#if searchQuery}
            Try a different search term
          {:else}
            Start a new conversation
          {/if}
        </p>
        {#if !searchQuery}
          <button class="start-chat-btn" on:click={handleNewChat}>
            <Plus size={14} />
            Start Chat
          </button>
        {/if}
      </div>
    {/if}
  </div>
  
  <!-- Context Menu -->
  {#if showContextMenu && contextMenuChat}
    <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
    <div 
      class="context-menu" 
      style="left: {contextMenuPosition.x}px; top: {contextMenuPosition.y}px;"
      transition:slide={{ duration: 100 }}
      role="menu"
      tabindex="-1"
      on:click|stopPropagation
    >
      <button 
        class="menu-item" 
        on:click={() => handlePinToggle(getChatId(contextMenuChat))}
        role="menuitem"
      >
        <Pin size={14} />
        {contextMenuChat.isPinned ? 'Unpin' : 'Pin'}
      </button>
      <button 
        class="menu-item danger" 
        on:click={() => handleDelete(getChatId(contextMenuChat))}
        role="menuitem"
      >
        <Trash2 size={14} />
        Delete
      </button>
    </div>
  {/if}
</div>

<style>
  .chat-list-sidebar {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary);
  }
  
  .sidebar-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px;
    border-bottom: 1px solid var(--border-subtle);
  }
  
  .search-container {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    border: 1px solid var(--border-subtle);
    transition: border-color 0.15s;
  }
  
  .search-container:focus-within {
    border-color: var(--accent-primary);
  }
  
  .search-container input {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--text-primary);
    font-size: 12px;
    outline: none;
  }
  
  .search-container input::placeholder {
    color: var(--text-muted);
  }
  
  .new-chat-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: var(--accent-primary);
    border: none;
    border-radius: 6px;
    color: var(--bg-primary);
    cursor: pointer;
    transition: all 0.15s;
  }
  
  .new-chat-btn:hover {
    background: var(--accent-secondary);
    transform: scale(1.05);
  }
  
  .chat-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
  }
  
  .section-divider {
    display: flex;
    align-items: center;
    padding: 8px 4px 4px;
    margin-top: 4px;
  }
  
  .section-divider:first-child {
    margin-top: 0;
  }
  
  .section-divider span {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
  }
  
  .section-divider.unpinned {
    margin-top: 12px;
    border-top: 1px solid var(--border-subtle);
    padding-top: 12px;
  }
  
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 32px 16px;
    text-align: center;
  }
  
  .empty-icon {
    font-size: 32px;
    margin-bottom: 12px;
    opacity: 0.5;
  }
  
  .empty-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 4px;
  }
  
  .empty-description {
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 16px;
  }
  
  .start-chat-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    background: var(--accent-primary);
    border: none;
    border-radius: 6px;
    color: var(--bg-primary);
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
  }
  
  .start-chat-btn:hover {
    background: var(--accent-secondary);
  }
  
  .context-menu {
    position: fixed;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 4px;
    min-width: 140px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    z-index: 1000;
  }
  
  .menu-item {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 8px 12px;
    background: transparent;
    border: none;
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 12px;
    cursor: pointer;
    transition: background 0.15s;
  }
  
  .menu-item:hover {
    background: var(--bg-tertiary);
  }
  
  .menu-item.danger {
    color: var(--accent-danger);
  }
  
  .menu-item.danger:hover {
    background: rgba(239, 68, 68, 0.1);
  }
</style>

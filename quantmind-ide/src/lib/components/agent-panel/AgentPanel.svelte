<script lang="ts">
  import { onMount, setContext } from 'svelte';
  import { slide, fade } from 'svelte/transition';
  import { get } from 'svelte/store';
  import { ChevronLeft, ChevronRight } from 'lucide-svelte';
  
  // Import child components
  import AgentHeader from './AgentHeader.svelte';
  import ChatListSidebar from './ChatListSidebar.svelte';
  import MessagesArea from './MessagesArea.svelte';
  import ContextBar from './ContextBar.svelte';
  import InputArea from './InputArea.svelte';
  import SettingsPanel from './settings/SettingsPanel.svelte';
  import ContextPicker from './ContextPicker.svelte';
  
  // Import stores
  import { chatStore, activeChat, activeMessages, activeContext } from '../../stores/chatStore';
  import type { AgentType, Chat, Message, ChatContext, FileReference, StrategyReference, BrokerReference, BacktestReference } from '../../stores/chatStore';
  
  // Import services
  import { contextManager } from '../../services/contextManager';
  import { agentManager } from '../../agents/agentManager';
  
  // Props
  export let isOpen = true;
  
  // State
  let showSettings = false;
  let showChatList = true;
  let chatListWidth = 200;
  let isResizing = false;
  let showContextPicker = false;
  let contextPickerType: 'file' | 'strategy' | 'broker' | 'backtest' = 'file';
  
  // Agent configuration
  const agents: Array<{ id: AgentType; name: string; icon: any; description: string }> = [
    { id: 'copilot', name: 'Copilot', icon: 'Bot', description: 'General assistant' },
    { id: 'quantcode', name: 'QuantCode', icon: 'Code', description: 'Code & debug' },
    { id: 'analyst', name: 'Analyst', icon: 'Wand2', description: 'Strategy analysis' }
  ];
  
  // Set up context for child components
  setContext('agents', agents);
  
  // Initialize store on mount
  onMount(() => {
    chatStore.initialize();
  });
  
  // Handle agent switch
  function handleAgentSwitch(agent: AgentType) {
    chatStore.switchAgent(agent);
  }
  
  // Handle new chat creation
  function handleNewChat() {
    chatStore.createChat($chatStore.activeAgent);
  }
  
  // Handle settings toggle
  function toggleSettings() {
    showSettings = !showSettings;
  }
  
  // Handle panel toggle
  function togglePanel() {
    isOpen = !isOpen;
  }
  
  // Handle chat list toggle
  function toggleChatList() {
    showChatList = !showChatList;
  }
  
  // Handle add context event from ContextBar - opens real selection picker
  function handleAddContext(event: CustomEvent<{ type: string }>) {
    const { type } = event.detail;
    
    // Validate type and open the appropriate picker
    if (['file', 'strategy', 'broker', 'backtest'].includes(type)) {
      contextPickerType = type as 'file' | 'strategy' | 'broker' | 'backtest';
      showContextPicker = true;
    }
  }
  
  // Handle context item selection from picker
  function handleContextSelect(event: CustomEvent<{ type: keyof ChatContext; item: FileReference | StrategyReference | BrokerReference | BacktestReference }>) {
    const { type, item } = event.detail;
    
    // Add the selected item to chat context
    chatStore.addContextItem(type, item);
    
    // Close the picker
    showContextPicker = false;
  }
  
  // Handle context picker error
  function handleContextPickerError(event: CustomEvent<{ message: string }>) {
    console.warn('Context picker error:', event.detail.message);
    // Could show a toast notification here
  }
  
  // Handle remove context event from ContextBar
  function handleRemoveContext(event: CustomEvent<{ type: keyof ChatContext; id: string }>) {
    const { type, id } = event.detail;
    chatStore.removeContextItem(type, id);
  }
  
  // Handle send message event from InputArea - routes to real agent client
  async function handleSendMessage(event: CustomEvent<{
    message: string;
    model: string;
    provider: string;
    context: ChatContext;
  }>) {
    const { message, model, provider, context } = event.detail;
    
    // Ensure a chat exists before sending
    const state = get(chatStore);
    if (!state.activeChatId) {
      chatStore.createChat(state.activeAgent);
    }
    
    try {
      // Invoke the agent through agentManager
      const response = await agentManager.invoke(currentAgent, message, {
        context: contextManager.serializeContext(context),
        model,
        provider
      });
      
      // Handle streaming or complete response
      if (response) {
        // Add assistant message to chat
        chatStore.addMessage({
          role: 'assistant',
          content: response.content || response.response || JSON.stringify(response),
          model,
          metadata: {
            provider,
            latency: response.latency,
            tokenCount: response.tokenCount
          }
        });
      }
    } catch (error) {
      console.error('Agent invocation failed:', error);
      // Add error message to chat
      chatStore.addMessage({
        role: 'assistant',
        content: `Failed to get response from agent: ${error instanceof Error ? error.message : 'Unknown error'}. Please check your connection and try again.`
      });
    }
  }
  
  // Resize handling
  function startResize(e: MouseEvent) {
    isResizing = true;
    document.addEventListener('mousemove', handleResize);
    document.addEventListener('mouseup', stopResize);
  }

  function handleResize(e: MouseEvent) {
    if (!isResizing) return;
    const newWidth = window.innerWidth - e.clientX;
    chatListWidth = Math.max(280, Math.min(600, newWidth));
  }

  // Helper for keyboard event handling
  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter') {
      startResize(e as unknown as MouseEvent);
    }
  }
  
  function stopResize() {
    isResizing = false;
    document.removeEventListener('mousemove', handleResize);
    document.removeEventListener('mouseup', stopResize);
  }
  
  // Reactive subscriptions
  $: currentAgent = $chatStore.activeAgent;
  $: currentChat = $activeChat;
  $: messages = $activeMessages;
  $: context = $activeContext;
</script>

{#if isOpen}
  <aside class="agent-panel" style="width: {chatListWidth}px;">
    <!-- Agent Header with tabs -->
    <AgentHeader 
      {agents}
      activeAgent={currentAgent}
      on:agentSwitch={(e) => handleAgentSwitch(e.detail)}
      on:newChat={handleNewChat}
      on:toggleSettings={toggleSettings}
      on:toggleChatList={toggleChatList}
      showChatList={showChatList}
    />
    
    <!-- Main content area with CSS Grid -->
    <div class="panel-body">
      <!-- Chat List Sidebar (collapsible) -->
      {#if showChatList}
        <div class="chat-list-container" transition:slide={{ axis: 'x' }}>
          <ChatListSidebar />
        </div>
      {/if}
      
      <!-- Main Chat Area -->
      <div class="main-content">
        <!-- Context Bar -->
        <ContextBar 
          context={context} 
          on:addContext={handleAddContext}
          on:removeContext={handleRemoveContext}
        />
        
        <!-- Messages Area -->
        <MessagesArea 
          messages={messages} 
          agent={currentAgent}
          agents={agents}
        />
        
        <!-- Input Area -->
        <InputArea 
          agent={currentAgent} 
          on:send={handleSendMessage}
          on:openSettings={toggleSettings}
        />
      </div>
    </div>
    
    <!-- Resize handle -->
    <!-- svelte-ignore a11y-no-static-element-interactions a11y-no-noninteractive-element-interactions a11y-no-noninteractive-tabindex -->
    <div 
      class="resize-handle" 
      on:mousedown={startResize}
      role="separator"
      aria-label="Resize panel"
      tabindex="0"
      on:keydown={handleKeydown}
    ></div>
    
    <!-- Settings Panel Modal -->
    {#if showSettings}
      <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
      <div class="settings-overlay" on:click={toggleSettings} transition:fade role="button" tabindex="-1" aria-label="Close settings">
        <SettingsPanel on:close={toggleSettings} />
      </div>
    {/if}
    
    <!-- Context Picker Modal -->
    {#if showContextPicker}
      <ContextPicker 
        type={contextPickerType}
        on:select={handleContextSelect}
        on:error={handleContextPickerError}
        on:close={() => showContextPicker = false}
      />
    {/if}
  </aside>
{:else}
  <!-- Toggle button when panel is closed -->
  <button 
    class="toggle-btn" 
    on:click={togglePanel}
    aria-label="Open agent panel"
  >
    <ChevronLeft size={16} />
  </button>
{/if}

<style>
  .agent-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-secondary);
    border-left: 1px solid var(--border-subtle);
    position: relative;
    min-width: 280px;
    max-width: 600px;
  }
  
  .panel-body {
    display: flex;
    flex: 1;
    overflow: hidden;
  }
  
  .chat-list-container {
    width: 200px;
    min-width: 150px;
    max-width: 300px;
    border-right: 1px solid var(--border-subtle);
    background: var(--bg-primary);
    overflow: hidden;
  }
  
  .main-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
    overflow: hidden;
  }
  
  .resize-handle {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    cursor: ew-resize;
    background: transparent;
    transition: background 0.2s;
  }
  
  .resize-handle:hover {
    background: var(--accent-primary);
  }
  
  .settings-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }
  
  .toggle-btn {
    position: fixed;
    right: 0;
    top: 50%;
    transform: translateY(-50%);
    width: 24px;
    height: 48px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-right: none;
    border-radius: 8px 0 0 8px;
    color: var(--text-muted);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
  }
  
  .toggle-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }
  
  /* Responsive adjustments */
  @media (max-width: 768px) {
    .agent-panel {
      width: 100% !important;
      max-width: 100%;
    }
    
    .chat-list-container {
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      z-index: 10;
      box-shadow: 2px 0 8px rgba(0, 0, 0, 0.3);
    }
  }
</style>

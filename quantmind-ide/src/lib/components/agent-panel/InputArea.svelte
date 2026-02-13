<script lang="ts">
  import { createEventDispatcher, onMount, tick } from 'svelte';
  import { fade, slide } from 'svelte/transition';
  import { get } from 'svelte/store';
  import { Send, Paperclip, ChevronDown, X, Loader, CheckCircle, AlertCircle } from 'lucide-svelte';
  import SlashCommandPalette from './SlashCommandPalette.svelte';
  import { chatStore, activeContext } from '../../stores/chatStore';
  import type { AgentType } from '../../stores/chatStore';
  import commandHandler from '../../services/commandHandler';
  
  // Props
  export let agent: AgentType;
  
  const dispatch = createEventDispatcher();
  
  // State
  let message = '';
  let textareaElement: HTMLTextAreaElement;
  let showModelDropdown = false;
  let showSlashCommands = false;
  let slashCommandFilter = '';
  let isLoading = false;
  
  // Command feedback state
  let commandFeedback: { type: 'success' | 'error'; message: string } | null = null;
  let feedbackTimeout: ReturnType<typeof setTimeout> | null = null;
  
  // Model configuration
  const providers = [
    { id: 'google', name: 'Google', models: ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash'] },
    { id: 'anthropic', name: 'Anthropic', models: ['claude-sonnet-4', 'claude-3.5-sonnet'] },
    { id: 'qwen', name: 'Qwen', models: ['qwen-32b-coder'] },
    { id: 'ollama', name: 'Ollama (Local)', models: ['codellama', 'mistral', 'qwen2.5-coder'] }
  ];
  
  let selectedProvider = 'google';
  let selectedModel = 'gemini-2.5-pro';
  
  // Context from store
  $: context = $activeContext;
  $: hasContext = context.files.length > 0 || context.strategies.length > 0 || 
                   context.brokers.length > 0 || context.backtests.length > 0;
  
  // Character counter
  $: charCount = message.length;
  const maxChars = 4000;
  
  // Auto-resize textarea
  function autoResize() {
    if (textareaElement) {
      textareaElement.style.height = 'auto';
      const newHeight = Math.min(Math.max(textareaElement.scrollHeight, 40), 150);
      textareaElement.style.height = newHeight + 'px';
    }
  }
  
  // Handle input changes
  function handleInput() {
    autoResize();
    
    // Check for slash command trigger
    if (message === '/') {
      showSlashCommands = true;
      slashCommandFilter = '';
    } else if (message.startsWith('/')) {
      showSlashCommands = true;
      slashCommandFilter = message.slice(1);
    } else {
      showSlashCommands = false;
    }
  }
  
  // Handle keydown
  function handleKeydown(e: KeyboardEvent) {
    // Send on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
    
    // Close slash commands on Escape
    if (e.key === 'Escape') {
      showSlashCommands = false;
    }
    
    // Navigate slash commands with arrow keys
    if (showSlashCommands && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
      e.preventDefault();
      // Navigation handled in SlashCommandPalette
    }
  }
  
  // Show command feedback with auto-dismiss
  function showCommandFeedback(type: 'success' | 'error', msg: string) {
    // Clear any existing timeout
    if (feedbackTimeout) {
      clearTimeout(feedbackTimeout);
    }
    
    commandFeedback = { type, message: msg };
    
    // Auto-dismiss after 4 seconds
    feedbackTimeout = setTimeout(() => {
      commandFeedback = null;
    }, 4000);
  }
  
  // Execute a slash command
  async function executeCommand(commandInput: string): Promise<boolean> {
    const parsed = commandHandler.parse(commandInput);
    if (!parsed) {
      showCommandFeedback('error', 'Invalid command format');
      return false;
    }
    
    // Get current chat ID from store
    const state = get(chatStore);
    const chatId = state.activeChatId || 'default';
    
    try {
      const result = await commandHandler.execute(commandInput, {
        agent,
        chatId,
        attachedFiles: context.files.map(f => f.path),
        // Provide callback for adding context items
        addContextItem: (type: string, item: any) => {
          chatStore.addContextItem(type as any, item);
        }
      });
      
      if (result.success) {
        showCommandFeedback('success', result.message);
        
        // Handle special command actions - route to concrete behaviors
        const action = result.data?.action;
        
        switch (action) {
          case 'clear-chat':
            // Clear messages for the current chat
            chatStore.clearMessages(chatId);
            break;
            
          case 'open-settings':
            dispatch('openSettings');
            break;
            
          case 'open-skills':
            dispatch('openSkills');
            break;
            
          case 'export':
            // Export chat history
            handleExport(result.data?.format as string || 'json');
            break;
            
          case 'kill-switch':
            // Kill switch already handled by backend, but dispatch event for UI
            dispatch('killSwitch');
            break;
            
          case 'attach':
            // Attach already handled via addContextItem callback
            break;
            
          case 'backtest':
          case 'analyze':
          case 'add-broker':
          case 'terminal':
            // These are handled by backend API calls in commandHandler
            // Dispatch events for UI updates if needed
            dispatch('commandExecuted', { action, data: result.data });
            break;
            
          case 'memory':
            // Memory management - could open memory panel
            dispatch('memoryAction', { subAction: result.data?.subAction });
            break;
        }
        
        return true;
      } else {
        showCommandFeedback('error', result.error || result.message);
        return false;
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Command execution failed';
      showCommandFeedback('error', errorMessage);
      return false;
    }
  }
  
  // Handle chat export
  function handleExport(format: string) {
    const state = get(chatStore);
    const currentChat = state.chats.find(c => c.id === state.activeChatId);
    
    if (!currentChat) {
      showCommandFeedback('error', 'No active chat to export');
      return;
    }
    
    let content: string;
    let mimeType: string;
    let extension: string;
    
    switch (format) {
      case 'json':
        content = JSON.stringify(currentChat, null, 2);
        mimeType = 'application/json';
        extension = 'json';
        break;
      case 'md':
      case 'markdown':
        content = formatChatAsMarkdown(currentChat);
        mimeType = 'text/markdown';
        extension = 'md';
        break;
      case 'txt':
      default:
        content = formatChatAsText(currentChat);
        mimeType = 'text/plain';
        extension = 'txt';
        break;
    }
    
    // Create download
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-${currentChat.title.slice(0, 30).replace(/[^a-z0-9]/gi, '_')}.${extension}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showCommandFeedback('success', `Chat exported as ${extension.toUpperCase()}`);
  }
  
  // Format chat as markdown
  function formatChatAsMarkdown(chat: any): string {
    let md = `# ${chat.title}\n\n`;
    md += `**Agent:** ${chat.agent}\n`;
    md += `**Created:** ${new Date(chat.createdAt).toLocaleString()}\n\n`;
    md += `---\n\n`;
    
    for (const msg of chat.messages) {
      md += `### ${msg.role === 'user' ? '👤 User' : '🤖 Assistant'}\n`;
      md += `*${new Date(msg.timestamp).toLocaleString()}*\n\n`;
      md += `${msg.content}\n\n`;
      md += `---\n\n`;
    }
    
    return md;
  }
  
  // Format chat as plain text
  function formatChatAsText(chat: any): string {
    let txt = `${chat.title}\n`;
    txt += `Agent: ${chat.agent}\n`;
    txt += `Created: ${new Date(chat.createdAt).toLocaleString()}\n`;
    txt += `${'='.repeat(50)}\n\n`;
    
    for (const msg of chat.messages) {
      txt += `[${msg.role.toUpperCase()}] ${new Date(msg.timestamp).toLocaleString()}\n`;
      txt += `${msg.content}\n\n`;
    }
    
    return txt;
  }
  
  // Send message
  async function sendMessage() {
    if (!message.trim() || isLoading) return;
    
    const userMessage = message.trim();
    message = '';
    showSlashCommands = false;
    autoResize();
    
    // Check if the message is a slash command
    if (commandHandler.isCommand(userMessage)) {
      await executeCommand(userMessage);
      return; // Don't add as a regular message
    }
    
    // Ensure a chat exists before adding messages
    const state = get(chatStore);
    if (!state.activeChatId) {
      chatStore.createChat(state.activeAgent);
    }
    
    // Add user message to store
    chatStore.addMessage({
      role: 'user',
      content: userMessage
    });
    
    isLoading = true;
    
    try {
      // Dispatch send event for parent to handle agent API call
      // The parent (AgentPanel) will handle the actual agent invocation
      // and add the assistant response to the chat
      dispatch('send', {
        message: userMessage,
        model: selectedModel,
        provider: selectedProvider,
        context
      });
      
    } catch (error) {
      console.error('Failed to send message:', error);
      chatStore.addMessage({
        role: 'assistant',
        content: 'Failed to get response. Please try again.'
      });
    } finally {
      isLoading = false;
    }
  }
  
  // Handle slash command selection - execute directly
  async function handleCommandSelect(command: string) {
    message = '';
    showSlashCommands = false;
    autoResize();
    textareaElement?.focus();
    
    // Execute the command directly
    await executeCommand(command);
  }
  
  // Handle model selection
  function selectModel(provider: string, model: string) {
    selectedProvider = provider;
    selectedModel = model;
    showModelDropdown = false;
  }
  
  // Attach file
  function attachFile() {
    dispatch('attachFile');
  }
  
  // Remove context item
  function removeContext(type: string, id: string) {
    chatStore.removeContextItem(type as any, id);
  }
  
  // Get short model name
  function getShortModelName(model: string): string {
    const parts = model.split('-');
    if (parts.length > 2) {
      return parts.slice(0, 2).join('-');
    }
    return model;
  }
</script>

<div class="input-area">
  <!-- Context attachments preview -->
  {#if hasContext}
    <div class="context-preview" transition:slide={{ duration: 200 }}>
      {#each context.files as file}
        <div class="context-chip">
          <span>📄 {file.name}</span>
          <button on:click={() => removeContext('files', file.id)} aria-label="Remove file">
            <X size={10} />
          </button>
        </div>
      {/each}
      {#each context.strategies as strategy}
        <div class="context-chip">
          <span>📊 {strategy.name}</span>
          <button on:click={() => removeContext('strategies', strategy.id)} aria-label="Remove strategy">
            <X size={10} />
          </button>
        </div>
      {/each}
      {#each context.brokers as broker}
        <div class="context-chip">
          <span>🔗 {broker.name}</span>
          <button on:click={() => removeContext('brokers', broker.id)} aria-label="Remove broker">
            <X size={10} />
          </button>
        </div>
      {/each}
      {#each context.backtests as backtest}
        <div class="context-chip">
          <span>📈 {backtest.name}</span>
          <button on:click={() => removeContext('backtests', backtest.id)} aria-label="Remove backtest">
            <X size={10} />
          </button>
        </div>
      {/each}
    </div>
  {/if}
  
  <!-- Command feedback -->
  {#if commandFeedback}
    <div class="command-feedback {commandFeedback.type}" transition:fade={{ duration: 200 }}>
      {#if commandFeedback.type === 'success'}
        <CheckCircle size={14} />
      {:else}
        <AlertCircle size={14} />
      {/if}
      <span>{commandFeedback.message}</span>
      <button on:click={() => commandFeedback = null} aria-label="Dismiss">
        <X size={12} />
      </button>
    </div>
  {/if}
  
  <!-- Input container -->
  <div class="input-container">
    <!-- Attach button -->
    <button 
      class="input-btn attach-btn" 
      on:click={attachFile}
      title="Attach file"
      aria-label="Attach file"
    >
      <Paperclip size={16} />
    </button>
    
    <!-- Textarea -->
    <textarea
      bind:value={message}
      bind:this={textareaElement}
      on:input={handleInput}
      on:keydown={handleKeydown}
      placeholder="Message {agent}... (/ for commands)"
      rows="1"
      aria-label="Message input"
      disabled={isLoading}
    ></textarea>
    
    <!-- Slash Command Palette -->
    {#if showSlashCommands}
      <SlashCommandPalette
        filter={slashCommandFilter}
        on:select={(e) => handleCommandSelect(e.detail)}
        on:close={() => showSlashCommands = false}
      />
    {/if}
    
    <!-- Model selector -->
    <div class="model-selector">
      <button 
        class="model-btn" 
        on:click={() => showModelDropdown = !showModelDropdown}
        aria-label="Select model"
        aria-expanded={showModelDropdown}
      >
        <span class="model-name">{getShortModelName(selectedModel)}</span>
        <ChevronDown size={12} />
      </button>
      
      {#if showModelDropdown}
        <div class="model-dropdown" transition:fade={{ duration: 100 }}>
          {#each providers as provider}
            <div class="provider-group">
              <span class="provider-name">{provider.name}</span>
              {#each provider.models as model}
                <button 
                  class="model-option"
                  class:selected={selectedModel === model}
                  on:click={() => selectModel(provider.id, model)}
                >
                  {model}
                </button>
              {/each}
            </div>
          {/each}
        </div>
      {/if}
    </div>
    
    <!-- Send button -->
    <button 
      class="send-btn" 
      on:click={sendMessage}
      disabled={!message.trim() || isLoading}
      aria-label="Send message"
    >
      {#if isLoading}
        <Loader size={16} class="spin" />
      {:else}
        <Send size={16} />
      {/if}
    </button>
  </div>
  
  <!-- Character counter -->
  <div class="char-counter">
    <span class:warning={charCount > maxChars * 0.8} class:error={charCount >= maxChars}>
      {charCount} / {maxChars}
    </span>
  </div>
</div>

<style>
  .input-area {
    padding: 12px;
    border-top: 1px solid var(--border-subtle);
    background: var(--bg-secondary);
  }
  
  .context-preview {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 8px;
  }
  
  .context-chip {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    font-size: 11px;
    color: var(--text-secondary);
  }
  
  .context-chip button {
    display: flex;
    align-items: center;
    justify-content: center;
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    padding: 2px;
    border-radius: 2px;
  }
  
  .context-chip button:hover {
    color: var(--accent-danger);
  }
  
  /* Command feedback */
  .command-feedback {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    margin-bottom: 8px;
    border-radius: 8px;
    font-size: 12px;
    animation: slideIn 0.2s ease-out;
  }
  
  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateY(-4px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
  
  .command-feedback.success {
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.3);
    color: #22c55e;
  }
  
  .command-feedback.error {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: #ef4444;
  }
  
  .command-feedback span {
    flex: 1;
  }
  
  .command-feedback button {
    display: flex;
    align-items: center;
    justify-content: center;
    background: none;
    border: none;
    color: inherit;
    cursor: pointer;
    padding: 2px;
    border-radius: 4px;
    opacity: 0.7;
    transition: opacity 0.15s;
  }
  
  .command-feedback button:hover {
    opacity: 1;
  }
  
  .input-container {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 8px 12px;
    transition: border-color 0.15s;
  }
  
  .input-container:focus-within {
    border-color: var(--accent-primary);
  }
  
  .input-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: transparent;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    border-radius: 6px;
    transition: all 0.15s;
    flex-shrink: 0;
  }
  
  .input-btn:hover {
    background: var(--bg-secondary);
    color: var(--text-primary);
  }
  
  .input-container textarea {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--text-primary);
    font-size: 13px;
    padding: 6px 0;
    resize: none;
    font-family: inherit;
    min-height: 40px;
    max-height: 150px;
    outline: none;
    overflow-y: auto;
  }
  
  .input-container textarea::placeholder {
    color: var(--text-muted);
  }
  
  .input-container textarea:disabled {
    opacity: 0.6;
  }
  
  /* Model Selector */
  .model-selector {
    position: relative;
    flex-shrink: 0;
  }
  
  .model-btn {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s;
  }
  
  .model-btn:hover {
    color: var(--text-primary);
    border-color: var(--accent-primary);
  }
  
  .model-name {
    font-size: 10px;
    max-width: 80px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  
  .model-dropdown {
    position: absolute;
    bottom: 100%;
    right: 0;
    margin-bottom: 8px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    min-width: 200px;
    max-height: 300px;
    overflow-y: auto;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    z-index: 100;
  }
  
  .provider-group {
    padding: 8px 0;
    border-bottom: 1px solid var(--border-subtle);
  }
  
  .provider-group:last-child {
    border-bottom: none;
  }
  
  .provider-name {
    display: block;
    padding: 4px 12px;
    font-size: 10px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  
  .model-option {
    display: block;
    width: 100%;
    padding: 8px 12px;
    background: transparent;
    border: none;
    color: var(--text-secondary);
    font-size: 12px;
    text-align: left;
    cursor: pointer;
    transition: background 0.15s;
  }
  
  .model-option:hover {
    background: var(--bg-tertiary);
  }
  
  .model-option.selected {
    color: var(--accent-primary);
    background: rgba(107, 200, 230, 0.1);
  }
  
  /* Send Button */
  .send-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    background: var(--accent-primary);
    border: none;
    border-radius: 8px;
    color: var(--bg-primary);
    cursor: pointer;
    transition: all 0.15s;
    flex-shrink: 0;
  }
  
  .send-btn:hover:not(:disabled) {
    background: var(--accent-secondary);
    transform: scale(1.05);
  }
  
  .send-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  /* Character Counter */
  .char-counter {
    display: flex;
    justify-content: flex-end;
    padding: 4px 12px 0;
    font-size: 10px;
    color: var(--text-muted);
  }
  
  .char-counter .warning {
    color: var(--accent-warning);
  }
  
  .char-counter .error {
    color: var(--accent-danger);
  }
</style>

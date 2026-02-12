<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import { 
    Save, Undo, Redo, Copy, Download, Search, 
    ChevronDown, ChevronUp, X, FileText, Code,
    Maximize2, Minimize2, Eye, Edit3
  } from 'lucide-svelte';

  export let file: {
    id: string;
    name: string;
    path?: string;
    content: string;
    type?: string;
  } | null = null;

  export let readOnly = false;
  export let showLineNumbers = true;
  export let wordWrap = false;
  export let fontSize = 13;

  const dispatch = createEventDispatcher();

  let editorContent = '';
  let undoStack: string[] = [];
  let redoStack: string[] = [];
  let searchQuery = '';
  let searchResults: number[] = [];
  let currentSearchIndex = -1;
  let showSearch = false;
  let isMaximized = false;
  let copied = false;
  let cursorLine = 1;
  let cursorColumn = 1;

  const API_BASE = 'http://localhost:8000/api';

  // Supported file types for syntax highlighting
  const languageMap: Record<string, string> = {
    'mq5': 'mql5',
    'mqh': 'mql5',
    'py': 'python',
    'ts': 'typescript',
    'js': 'javascript',
    'json': 'json',
    'md': 'markdown',
    'txt': 'text',
    'sql': 'sql',
    'yaml': 'yaml',
    'yml': 'yaml',
    'xml': 'xml',
    'html': 'html',
    'css': 'css'
  };

  $: language = file?.name ? getLanguage(file.name) : 'text';
  $: if (file) {
    editorContent = file.content || '';
    undoStack = [];
    redoStack = [];
  }

  function getLanguage(filename: string): string {
    const ext = filename.split('.').pop()?.toLowerCase() || '';
    return languageMap[ext] || 'text';
  }

  function handleInput(event: Event) {
    const target = event.target as HTMLTextAreaElement;
    const newContent = target.value;
    
    undoStack = [...undoStack, editorContent];
    redoStack = [];
    editorContent = newContent;
    updateCursor(target);
    
    if (searchQuery) highlightSearch();
  }

  function handleKeyDown(event: KeyboardEvent) {
    // Handle Tab key
    if (event.key === 'Tab') {
      event.preventDefault();
      const target = event.target as HTMLTextAreaElement;
      const start = target.selectionStart;
      const end = target.selectionEnd;
      editorContent = editorContent.substring(0, start) + '  ' + editorContent.substring(end);
      
      // Restore cursor position after update
      setTimeout(() => {
        target.selectionStart = target.selectionEnd = start + 2;
        updateCursor(target);
      }, 0);
      
      undoStack = [...undoStack, editorContent.substring(0, start)];
    }
    
    // Undo (Ctrl/Cmd + Z)
    if ((event.ctrlKey || event.metaKey) && event.key === 'z' && !event.shiftKey) {
      event.preventDefault();
      undo();
    }
    
    // Redo (Ctrl/Cmd + Shift + Z or Ctrl/Cmd + Y)
    if ((event.ctrlKey || event.metaKey) && (event.key === 'y' || (event.key === 'z' && event.shiftKey))) {
      event.preventDefault();
      redo();
    }
    
    // Save (Ctrl/Cmd + S)
    if ((event.ctrlKey || event.metaKey) && event.key === 's') {
      event.preventDefault();
      saveFile();
    }
    
    // Copy (Ctrl/Cmd + C)
    if ((event.ctrlKey || event.metaKey) && event.key === 'c') {
      // Let default copy behavior work
    }
  }

  function updateCursor(target: HTMLTextAreaElement) {
    const pos = target.selectionStart;
    const textBeforeCursor = editorContent.substring(0, pos);
    const lines = textBeforeCursor.split('\n');
    cursorLine = lines.length;
    cursorColumn = lines[lines.length - 1].length + 1;
  }

  function handleClick(event: MouseEvent) {
    const target = event.target as HTMLTextAreaElement;
    updateCursor(target);
  }

  function undo() {
    if (undoStack.length > 0) {
      const previous = undoStack[undoStack.length - 1];
      redoStack = [...redoStack, editorContent];
      editorContent = previous;
      undoStack = undoStack.slice(0, -1);
    }
  }

  function redo() {
    if (redoStack.length > 0) {
      const next = redoStack[redoStack.length - 1];
      undoStack = [...undoStack, editorContent];
      editorContent = next;
      redoStack = redoStack.slice(0, -1);
    }
  }

  async function saveFile() {
    if (!file || readOnly) return;
    
    dispatch('save', { content: editorContent });
    
    try {
      const endpoint = `${API_BASE}/files/${file.id}`;
      const response = await fetch(endpoint, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: editorContent })
      });
      
      if (response.ok) {
        dispatch('saved');
      }
    } catch (e) {
      console.error('Failed to save file:', e);
      dispatch('saved'); // Still emit saved event for local state
    }
  }

  function copyContent() {
    navigator.clipboard.writeText(editorContent);
    copied = true;
    setTimeout(() => copied = false, 2000);
  }

  function downloadFile() {
    if (!file) return;
    
    const blob = new Blob([editorContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = file.name;
    a.click();
    URL.revokeObjectURL(url);
  }

  function highlightSearch() {
    if (!searchQuery) {
      searchResults = [];
      return;
    }
    
    const regex = new RegExp(escapeRegExp(searchQuery), 'gi');
    const matches: number[] = [];
    let match;
    while ((match = regex.exec(editorContent)) !== null) {
      matches.push(match.index);
    }
    searchResults = matches;
    currentSearchIndex = matches.length > 0 ? 0 : -1;
  }

  function escapeRegExp(string: string): string {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  function navigateSearch(direction: 'next' | 'prev') {
    if (searchResults.length === 0) return;
    
    if (direction === 'next') {
      currentSearchIndex = (currentSearchIndex + 1) % searchResults.length;
    } else {
      currentSearchIndex = (currentSearchIndex - 1 + searchResults.length) % searchResults.length;
    }
    
    // Scroll to match
    const matchPos = searchResults[currentSearchIndex];
    const textBeforeMatch = editorContent.substring(0, matchPos);
    const lineNumber = textBeforeMatch.split('\n').length;
    const lineElements = document.querySelectorAll('.line-number');
    if (lineElements[lineNumber - 1]) {
      lineElements[lineNumber - 1].scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }

  function getHighlightedContent(): string {
    if (language === 'markdown') {
      return highlightMarkdown(editorContent);
    }
    return escapeHtml(editorContent);
  }

  function highlightMarkdown(content: string): string {
    // Simple markdown highlighting
    let html = escapeHtml(content);
    
    // Code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<code class="code-block">$2</code>');
    
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Headers
    html = html.replace(/^### (.+)$/gm, '<span class="md-header">### $1</span>');
    html = html.replace(/^## (.+)$/gm, '<span class="md-header">## $1</span>');
    html = html.replace(/^# (.+)$/gm, '<span class="md-header"># $1</span>');
    
    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // Italic
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    
    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
    
    // Lists
    html = html.replace(/^- (.+)$/gm, '<span class="md-list">- $1</span>');
    html = html.replace(/^\d+\. (.+)$/gm, '<span class="md-list">$1</span>');
    
    return html;
  }

  function escapeHtml(text: string): string {
    return text
      .replace(/&/g, '&')
      .replace(/</g, '<')
      .replace(/>/g, '>')
      .replace(/"/g, '"')
      .replace(/'/g, '&#039;');
  }

  function getLineCount(): number {
    return editorContent.split('\n').length;
  }

  function toggleMaximize() {
    isMaximized = !isMaximized;
    dispatch('maximize', { maximized: isMaximized });
  }

  $: canUndo = undoStack.length > 0;
  $: canRedo = redoStack.length > 0;
</script>

<div class="file-editor" class:maximized={isMaximized} class:read-only={readOnly}>
  <!-- Editor Header -->
  <div class="editor-header">
    <div class="header-left">
      <FileText size={16} class="file-icon" />
      <span class="file-name">{file?.name || 'Untitled'}</span>
      {#if file?.path}
        <span class="file-path">{file.path}</span>
      {/if}
      <span class="language-badge">{language}</span>
    </div>
    
    <div class="header-actions">
      {#if !readOnly}
        <button 
          class="action-btn" 
          on:click={undo} 
          disabled={!canUndo}
          title="Undo (Ctrl+Z)"
        >
          <Undo size={14} />
        </button>
        <button 
          class="action-btn" 
          on:click={redo} 
          disabled={!canRedo}
          title="Redo (Ctrl+Shift+Z)"
        >
          <Redo size={14} />
        </button>
        <div class="divider"></div>
        <button 
          class="action-btn" 
          on:click={copyContent}
          title="Copy"
        >
          {#if copied}
            <span class="copied-check">✓</span>
          {:else}
            <Copy size={14} />
          {/if}
        </button>
        <button 
          class="action-btn" 
          on:click={downloadFile}
          title="Download"
        >
          <Download size={14} />
        </button>
        {#if showSearch}
          <div class="search-box">
            <input
              type="text"
              placeholder="Search..."
              bind:value={searchQuery}
              on:input={highlightSearch}
            />
            <div class="search-nav">
              <button 
                class="search-btn" 
                on:click={() => navigateSearch('prev')}
                disabled={searchResults.length === 0}
              >
                <ChevronUp size={14} />
              </button>
              <button 
                class="search-btn" 
                on:click={() => navigateSearch('next')}
                disabled={searchResults.length === 0}
              >
                <ChevronDown size={14} />
              </button>
            </div>
            <span class="search-count">
              {#if searchResults.length > 0}
                {currentSearchIndex + 1}/{searchResults.length}
              {:else}
                0/0
              {/if}
            </span>
            <button class="action-btn close-search" on:click={() => { showSearch = false; searchQuery = ''; }}>
              <X size={14} />
            </button>
          </div>
        {:else}
          <button 
            class="action-btn" 
            on:click={() => showSearch = true}
            title="Search (Ctrl+F)"
          >
            <Search size={14} />
          </button>
        {/if}
        <div class="divider"></div>
        <button 
          class="action-btn save-btn" 
          on:click={saveFile}
          title="Save (Ctrl+S)"
        >
          <Save size={14} />
          <span>Save</span>
        </button>
      {/if}
      <button 
        class="action-btn" 
        on:click={toggleMaximize}
        title={isMaximized ? 'Minimize' : 'Maximize'}
      >
        {#if isMaximized}
          <Minimize2 size={14} />
        {:else}
          <Maximize2 size={14} />
        {/if}
      </button>
    </div>
  </div>

  <!-- Editor Toolbar -->
  <div class="editor-toolbar">
    <div class="cursor-position">
      Ln {cursorLine}, Col {cursorColumn}
    </div>
    <div class="file-stats">
      <span>{getLineCount()} lines</span>
      <span>{editorContent.length} characters</span>
    </div>
    <div class="toolbar-spacer"></div>
    <label class="toggle-option">
      <input type="checkbox" bind:checked={showLineNumbers} />
      <span>Line numbers</span>
    </label>
    <label class="toggle-option">
      <input type="checkbox" bind:checked={wordWrap} />
      <span>Word wrap</span>
    </label>
    <div class="font-size-control">
      <span>Size:</span>
      <button on:click={() => fontSize = Math.max(10, fontSize - 1)}>-</button>
      <span>{fontSize}px</span>
      <button on:click={() => fontSize = Math.min(24, fontSize + 1)}>+</button>
    </div>
  </div>

  <!-- Editor Content -->
  <div class="editor-body">
    <div class="line-numbers" class:hidden={!showLineNumbers}>
      {#each Array(getLineCount()) as _, i}
        <div class="line-number" class:current={i + 1 === cursorLine}>{i + 1}</div>
      {/each}
    </div>
    
    <div class="code-area" class:word-wrap={wordWrap} style="font-size: {fontSize}px">
      {#if readOnly}
        <pre class="code-content readonly"><code>{@html getHighlightedContent()}</code></pre>
      {:else}
        <textarea
          class="code-input"
          spellcheck="false"
          on:input={handleInput}
          on:keydown={handleKeyDown}
          on:click={handleClick}
          bind:value={editorContent}
        ></textarea>
        <pre class="code-overlay"><code>{@html getHighlightedContent()}</code></pre>
      {/if}
    </div>
  </div>

  <!-- Status Bar -->
  <div class="editor-statusbar">
    <div class="status-left">
      {#if readOnly}
        <span class="readonly-badge"><Eye size={12} /> Read Only</span>
      {:else}
        <span class="edit-badge"><Edit3 size={12} /> Editing</span>
      {/if}
    </div>
    <div class="status-center">
      {#if searchResults.length > 0}
        <span class="search-info">"{searchQuery}" - {searchResults.length} matches</span>
      {/if}
    </div>
    <div class="status-right">
      <span>UTF-8</span>
      <span class="divider">|</span>
      <span>MQL5</span>
      <span class="divider">|</span>
      <span>LF</span>
    </div>
  </div>
</div>

<style>
  .file-editor {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary, #1a1b26);
    border: 1px solid var(--border-subtle, #2d2d3a);
    border-radius: 8px;
    overflow: hidden;
    transition: all 0.3s ease;
  }

  .file-editor.maximized {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 1000;
    border-radius: 0;
    border: none;
  }

  .file-editor.read-only {
    background: var(--bg-secondary, #16161e);
  }

  /* Header */
  .editor-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    background: var(--bg-tertiary, #1f1f2e);
    border-bottom: 1px solid var(--border-subtle, #2d2d3a);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .file-icon {
    color: var(--text-muted, #787878);
  }

  .file-name {
    font-weight: 600;
    color: var(--text-primary, #c0caf5);
    font-size: 13px;
  }

  .file-path {
    color: var(--text-muted, #565f89);
    font-size: 11px;
  }

  .language-badge {
    padding: 2px 6px;
    background: var(--accent-primary, #7aa2f7);
    color: var(--bg-primary, #1a1b26);
    border-radius: 3px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .action-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    padding: 4px 8px;
    background: transparent;
    border: none;
    color: var(--text-secondary, #a9b1d6);
    cursor: pointer;
    border-radius: 4px;
    transition: all 0.15s ease;
    font-size: 12px;
  }

  .action-btn:hover:not(:disabled) {
    background: var(--bg-secondary, #292a3e);
    color: var(--text-primary, #c0caf5);
  }

  .action-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .action-btn.save-btn {
    background: var(--accent-primary, #7aa2f7);
    color: var(--bg-primary, #1a1b26);
    font-weight: 600;
  }

  .action-btn.save-btn:hover {
    background: var(--accent-secondary, #bb9af7);
  }

  .copied-check {
    color: #9ece6a;
    font-size: 12px;
  }

  .divider {
    width: 1px;
    height: 20px;
    background: var(--border-subtle, #2d2d3a);
    margin: 0 4px;
  }

  .search-box {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    background: var(--bg-secondary, #292a3e);
    border-radius: 4px;
  }

  .search-box input {
    width: 150px;
    padding: 4px 8px;
    background: transparent;
    border: 1px solid var(--border-subtle, #2d2d3a);
    border-radius: 4px;
    color: var(--text-primary, #c0caf5);
    font-size: 12px;
  }

  .search-box input:focus {
    outline: none;
    border-color: var(--accent-primary, #7aa2f7);
  }

  .search-nav {
    display: flex;
    flex-direction: column;
  }

  .search-btn {
    padding: 2px;
    background: none;
    border: none;
    color: var(--text-secondary, #a9b1d6);
    cursor: pointer;
  }

  .search-btn:hover:not(:disabled) {
    color: var(--accent-primary, #7aa2f7);
  }

  .search-btn:disabled {
    opacity: 0.3;
  }

  .search-count {
    font-size: 11px;
    color: var(--text-muted, #565f89);
    min-width: 35px;
    text-align: center;
  }

  .close-search {
    padding: 2px;
  }

  /* Toolbar */
  .editor-toolbar {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 4px 12px;
    background: var(--bg-secondary, #16161e);
    border-bottom: 1px solid var(--border-subtle, #2d2d3a);
    font-size: 11px;
    color: var(--text-muted, #565f89);
  }

  .cursor-position {
    color: var(--text-secondary, #a9b1d6);
    font-family: 'JetBrains Mono', monospace;
  }

  .file-stats {
    display: flex;
    gap: 12px;
  }

  .toolbar-spacer {
    flex: 1;
  }

  .toggle-option {
    display: flex;
    align-items: center;
    gap: 4px;
    cursor: pointer;
    color: var(--text-muted, #565f89);
  }

  .toggle-option input {
    accent-color: var(--accent-primary, #7aa2f7);
  }

  .font-size-control {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .font-size-control button {
    padding: 2px 6px;
    background: var(--bg-tertiary, #1f1f2e);
    border: 1px solid var(--border-subtle, #2d2d3a);
    border-radius: 3px;
    color: var(--text-secondary, #a9b1d6);
    cursor: pointer;
  }

  .font-size-control button:hover {
    background: var(--accent-primary, #7aa2f7);
    color: var(--bg-primary, #1a1b26);
  }

  /* Editor Body */
  .editor-body {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  .line-numbers {
    display: flex;
    flex-direction: column;
    padding: 12px 8px;
    background: var(--bg-secondary, #16161e);
    border-right: 1px solid var(--border-subtle, #2d2d3a);
    overflow-y: auto;
    min-width: 50px;
    text-align: right;
    user-select: none;
  }

  .line-numbers.hidden {
    display: none;
  }

  .line-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    line-height: 1.6;
    color: var(--text-muted, #565f89);
    padding-right: 4px;
  }

  .line-number.current {
    color: var(--accent-primary, #7aa2f7);
    font-weight: 600;
  }

  .code-area {
    flex: 1;
    position: relative;
    overflow: auto;
    padding: 12px;
  }

  .code-area.word-wrap {
    white-space: normal;
  }

  .code-input {
    position: absolute;
    top: 12px;
    left: 12px;
    right: 12px;
    bottom: 12px;
    width: calc(100% - 24px);
    height: calc(100% - 24px);
    background: transparent;
    border: none;
    color: transparent;
    caret-color: var(--accent-primary, #7aa2f7);
    font-family: 'JetBrains Mono', monospace;
    font-size: inherit;
    line-height: 1.6;
    resize: none;
    outline: none;
    white-space: pre;
    overflow: hidden;
  }

  .code-input::selection {
    background: rgba(122, 162, 247, 0.3);
    color: transparent;
  }

  .code-overlay {
    margin: 0;
    padding: 0;
    background: transparent;
    pointer-events: none;
    white-space: pre;
  }

  .code-content {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    line-height: 1.6;
    color: var(--text-primary, #c0caf5);
    margin: 0;
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  .code-content.readonly {
    padding: 0;
  }

  /* Syntax highlighting for overlay */
  :global(.code-overlay code) {
    color: var(--text-primary, #c0caf5);
  }

  :global(.code-overlay .code-block) {
    display: block;
    background: var(--bg-tertiary, #1f1f2e);
    padding: 12px;
    border-radius: 4px;
    margin: 4px 0;
  }

  :global(.code-overlay .md-header) {
    color: var(--accent-secondary, #bb9af7);
    font-weight: 600;
  }

  :global(.code-overlay .md-list) {
    color: var(--accent-primary, #7aa2f7);
  }

  /* Status Bar */
  .editor-statusbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 12px;
    background: var(--bg-tertiary, #1f1f2e);
    border-top: 1px solid var(--border-subtle, #2d2d3a);
    font-size: 11px;
    color: var(--text-muted, #565f89);
  }

  .status-left, .status-right {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .readonly-badge, .edit-badge {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .readonly-badge {
    color: #e0af68;
  }

  .edit-badge {
    color: #9ece6a;
  }

  .status-center {
    flex: 1;
    text-align: center;
  }

  .search-info {
    color: var(--accent-primary, #7aa2f7);
  }

  .status-right .divider {
    color: var(--border-subtle, #2d2d3a);
  }
</style>

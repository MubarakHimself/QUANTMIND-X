<script lang="ts">
  import { X, FileText, Code, BookOpen, Image } from 'lucide-svelte';
  
  // Tabs and content state
  export let openFiles: Array<{id: string, name: string, path?: string, content?: string, type?: string}> = [];
  export let activeTabId: string = '';
  
  const API_BASE = 'http://localhost:8000/api';
  
  // File type icons
  function getFileIcon(name: string) {
    const ext = name.split('.').pop()?.toLowerCase();
    if (ext === 'mq5' || ext === 'mqh' || ext === 'py') return Code;
    if (ext === 'md') return BookOpen;
    if (ext === 'png' || ext === 'jpg') return Image;
    return FileText;
  }
  
  // Close a tab
  function closeTab(e: Event, fileId: string) {
    e.stopPropagation();
    openFiles = openFiles.filter(f => f.id !== fileId);
    if (activeTabId === fileId && openFiles.length > 0) {
      activeTabId = openFiles[0].id;
    }
  }
  
  // Load file content
  export async function openFile(file: {id: string, name: string, path?: string, view?: string}) {
    // Check if already open
    const existing = openFiles.find(f => f.id === file.id);
    if (existing) {
      activeTabId = file.id;
      return;
    }
    
    // Determine endpoint based on view type
    let endpoint = '';
    if (file.view === 'knowledge') {
      endpoint = `${API_BASE}/knowledge/${file.id}/content`;
    } else if (file.view === 'assets') {
      endpoint = `${API_BASE}/assets/${file.id}/content`;
    }
    
    let content = '';
    if (endpoint) {
      try {
        const response = await fetch(endpoint);
        if (response.ok) {
          const data = await response.json();
          content = data.content || '';
        }
      } catch (e) {
        content = `// Failed to load file: ${file.name}\n// This is demo content`;
      }
    } else {
      content = getDemoContent(file.name);
    }
    
    openFiles = [...openFiles, {
      id: file.id,
      name: file.name,
      path: file.path,
      content: content,
      type: getFileType(file.name)
    }];
    activeTabId = file.id;
  }
  
  function getFileType(name: string): string {
    const ext = name.split('.').pop()?.toLowerCase();
    if (ext === 'mq5' || ext === 'mqh') return 'mql5';
    if (ext === 'py') return 'python';
    if (ext === 'md') return 'markdown';
    if (ext === 'json') return 'json';
    return 'text';
  }
  
  function getDemoContent(name: string): string {
    if (name.includes('.mqh') || name.includes('.mq5')) {
      return `//+------------------------------------------------------------------+
//| ${name}
//| QuantMindX Shared Asset
//+------------------------------------------------------------------+

#property copyright "QuantMindX"
#property version   "1.00"

// This is a demo file. Connect the backend API for real content.

input double RiskPercent = 2.0;
input int    MaxTrades   = 3;

bool Init() {
    Print("${name} initialized");
    return true;
}
`;
    }
    if (name.includes('.md')) {
      return `# ${name.replace('.md', '')}

This is a demo markdown file.

## Contents
- Connect the backend API for real content
- This file would contain strategy notes or documentation

## Example
\`\`\`mql5
// Sample code reference
input double Lots = 0.01;
\`\`\`
`;
    }
    return `// ${name}\n// Demo content - connect backend for real data`;
  }
  
  $: activeFile = openFiles.find(f => f.id === activeTabId);
</script>

<main class="main-editor">
  <!-- Tab Bar -->
  <div class="tab-bar">
    {#if openFiles.length === 0}
      <div class="tab active">
        <span class="tab-name">Welcome</span>
      </div>
    {:else}
      {#each openFiles as file}
        <div 
          class="tab" 
          class:active={activeTabId === file.id}
          on:click={() => activeTabId = file.id}
          on:keypress={(e) => e.key === 'Enter' && (activeTabId = file.id)}
          role="tab"
          tabindex="0"
        >
          <svelte:component this={getFileIcon(file.name)} size={14} />
          <span class="tab-name">{file.name}</span>
          <button class="tab-close" on:click={(e) => closeTab(e, file.id)}>
            <X size={14} />
          </button>
        </div>
      {/each}
    {/if}
  </div>
  
  <!-- Editor Content -->
  <div class="editor-content">
    {#if openFiles.length === 0}
      <!-- Welcome Screen -->
      <div class="welcome-screen">
        <h1>QuantMind IDE</h1>
        <p class="subtitle">Algorithmic Trading Development Environment</p>
        
        <div class="quick-actions">
          <button class="action-btn">
            <span class="icon">📹</span>
            <div class="action-text">
              <span class="action-title">Process New NPRD</span>
              <span class="action-desc">Upload YouTube video</span>
            </div>
          </button>
          <button class="action-btn">
            <span class="icon">🤖</span>
            <div class="action-text">
              <span class="action-title">Create New EA</span>
              <span class="action-desc">From template</span>
            </div>
          </button>
          <button class="action-btn">
            <span class="icon">📊</span>
            <div class="action-text">
              <span class="action-title">Run Backtest</span>
              <span class="action-desc">Mode A/B/C</span>
            </div>
          </button>
          <button class="action-btn">
            <span class="icon">📚</span>
            <div class="action-text">
              <span class="action-title">Browse Knowledge</span>
              <span class="action-desc">PageIndex search</span>
            </div>
          </button>
        </div>
        
        <p class="hint">Select a file from the sidebar to begin editing</p>
      </div>
    {:else if activeFile}
      <!-- File Content -->
      <div class="file-viewer">
        {#if activeFile.type === 'markdown'}
          <div class="markdown-preview">
            <pre>{activeFile.content}</pre>
          </div>
        {:else}
          <div class="code-viewer">
            <pre class="code-content"><code>{activeFile.content}</code></pre>
          </div>
        {/if}
      </div>
    {/if}
  </div>
</main>

<style>
  .main-editor {
    grid-column: 3;
    grid-row: 1;
    display: flex;
    flex-direction: column;
    background: var(--bg-primary);
    overflow: hidden;
  }
  
  .tab-bar {
    display: flex;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
    height: 36px;
    overflow-x: auto;
  }
  
  .tab {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0 12px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-subtle);
    color: var(--text-secondary);
    cursor: pointer;
    min-width: fit-content;
    max-width: 180px;
  }
  
  .tab.active {
    background: var(--bg-primary);
    color: var(--text-primary);
    border-bottom: 1px solid var(--bg-primary);
    margin-bottom: -1px;
  }
  
  .tab-name {
    font-size: 12px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  
  .tab-close {
    display: flex;
    align-items: center;
    justify-content: center;
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    padding: 2px;
    border-radius: 4px;
    opacity: 0;
    transition: opacity 0.1s;
  }
  
  .tab:hover .tab-close {
    opacity: 1;
  }
  
  .tab-close:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }
  
  .editor-content {
    flex: 1;
    overflow-y: auto;
    display: flex;
  }
  
  .welcome-screen {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
    text-align: center;
  }
  
  h1 {
    font-size: 32px;
    font-weight: 600;
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
  }
  
  .subtitle {
    color: var(--text-secondary);
    font-size: 14px;
    margin-bottom: 32px;
  }
  
  .quick-actions {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    max-width: 500px;
    margin-bottom: 32px;
  }
  
  .action-btn {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    color: var(--text-primary);
    cursor: pointer;
    transition: all 0.15s ease;
    text-align: left;
  }
  
  .action-btn:hover {
    background: var(--glass-bg);
    border-color: var(--accent-primary);
    transform: translateY(-2px);
  }
  
  .action-btn .icon {
    font-size: 24px;
  }
  
  .action-text {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  
  .action-title {
    font-size: 13px;
    font-weight: 500;
  }
  
  .action-desc {
    font-size: 11px;
    color: var(--text-muted);
  }
  
  .hint {
    color: var(--text-muted);
    font-size: 12px;
  }
  
  .file-viewer {
    flex: 1;
    overflow: auto;
    padding: 16px;
  }
  
  .code-viewer, .markdown-preview {
    height: 100%;
  }
  
  .code-content {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    line-height: 1.6;
    color: var(--text-primary);
    background: transparent;
    margin: 0;
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  
  .markdown-preview pre {
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    line-height: 1.7;
    white-space: pre-wrap;
    color: var(--text-primary);
  }
</style>

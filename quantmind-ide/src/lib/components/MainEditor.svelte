<script lang="ts">
  import { X, FileText, Code, BookOpen, Image } from 'lucide-svelte';
  import MonacoEditor from './MonacoEditor.svelte';
  import CodeEditor from './CodeEditor.svelte';
  import { buildApiUrl } from '$lib/api';

  
  interface Props {
    // Tabs and content state
    openFiles?: Array<{id: string, name: string, path?: string, content?: string, type?: string}>;
    activeTabId?: string;
    useMonaco?: boolean; // Toggle between Monaco and legacy editor
  }

  let { openFiles = $bindable([]), activeTabId = $bindable(''), useMonaco = $bindable(true) }: Props = $props();

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
      endpoint = buildApiUrl(`/api/knowledge/${file.id}/content`);
    } else if (file.view === 'assets') {
      endpoint = buildApiUrl(`/api/assets/${file.id}/content`);
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
        content = `// Failed to load file: ${file.name}\n// Please check your connection`;
      }
    } else {
      content = '';
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

  function handleEditorChange(event: CustomEvent) {
    const { content, filename } = event.detail;
    // Update the file content
    openFiles = openFiles.map(f =>
      f.id === activeTabId ? { ...f, content } : f
    );
  }

  function handleEditorSave(event: CustomEvent) {
    const { content, filename } = event.detail;
    // TODO: Save to backend
    console.log('Saving file:', filename);
  }

  let activeFile = $derived(openFiles.find(f => f.id === activeTabId));
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
        {@const SvelteComponent = getFileIcon(file.name)}
        <div
          class="tab"
          class:active={activeTabId === file.id}
          onclick={() => activeTabId = file.id}
          onkeypress={(e) => e.key === 'Enter' && (activeTabId = file.id)}
          role="tab"
          tabindex="0"
        >
          <SvelteComponent size={14} />
          <span class="tab-name">{file.name}</span>
          <button class="tab-close" onclick={(e) => closeTab(e, file.id)}>
            <X size={14} />
          </button>
        </div>
      {/each}
    {/if}

    <!-- Editor Toggle -->
    <div class="editor-toggle">
      <button
        class:active={useMonaco}
        onclick={() => useMonaco = true}
        title="Use Monaco Editor"
      >
        <Code size={12} />
      </button>
      <button
        class:active={!useMonaco}
        onclick={() => useMonaco = false}
        title="Use Legacy Editor"
      >
        <FileText size={12} />
      </button>
    </div>
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
              <span class="action-title">Video Ingest</span>
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

        <div class="editor-info">
          <p>Using: {useMonaco ? 'Monaco Editor' : 'Legacy Editor'}</p>
          <p class="features">
            {#if useMonaco}
              MQL5 IntelliSense • Syntax Highlighting • Debug Support • Git Integration
            {:else}
              Basic syntax highlighting • Fast loading
            {/if}
          </p>
        </div>
      </div>
    {:else if activeFile}
      <!-- File Content -->
      <div class="file-viewer">
        {#if activeFile.type === 'markdown'}
          <div class="markdown-preview">
            <pre>{activeFile.content}</pre>
          </div>
        {:else if useMonaco}
          <MonacoEditor
            content={activeFile.content || ''}
            filename={activeFile.name}
            language={activeFile.type || 'text'}
            fileId={activeFile.id}
            filePath={activeFile.path || ''}
            on:change={handleEditorChange}
            on:save={handleEditorSave}
          />
        {:else}
          <CodeEditor
            content={activeFile.content || ''}
            filename={activeFile.name}
            language={activeFile.type || 'text'}
            fileId={activeFile.id}
            filePath={activeFile.path || ''}
            on:change={handleEditorChange}
            on:save={handleEditorSave}
          />
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
    background: var(--color-bg-base);
    overflow: hidden;
  }

  .tab-bar {
    display: flex;
    background: var(--color-bg-surface);
    border-bottom: 1px solid var(--color-border-subtle);
    height: 36px;
    overflow-x: auto;
  }

  .tab {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0 12px;
    background: var(--color-bg-surface);
    border-right: 1px solid var(--color-border-subtle);
    color: var(--color-text-secondary);
    cursor: pointer;
    min-width: fit-content;
    max-width: 180px;
  }

  .tab.active {
    background: var(--color-bg-base);
    color: var(--color-text-primary);
    border-bottom: 1px solid var(--color-bg-base);
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
    color: var(--color-text-muted);
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
    background: var(--color-bg-elevated);
    color: var(--color-text-primary);
  }

  .editor-toggle {
    display: flex;
    margin-left: auto;
    padding: 4px;
    gap: 2px;
    background: var(--color-bg-elevated);
  }

  .editor-toggle button {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 4px 8px;
    background: transparent;
    border: none;
    border-radius: 4px;
    color: var(--color-text-muted);
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .editor-toggle button:hover {
    color: var(--color-text-primary);
  }

  .editor-toggle button.active {
    background: var(--color-accent-cyan);
    color: #000;
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
    background: linear-gradient(135deg, var(--color-accent-cyan), var(--color-accent-amber));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
  }

  .subtitle {
    color: var(--color-text-secondary);
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
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-radius: 8px;
    color: var(--color-text-primary);
    cursor: pointer;
    transition: all 0.15s ease;
    text-align: left;
  }

  .action-btn:hover {
    background: var(--glass-bg);
    border-color: var(--color-accent-cyan);
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
    color: var(--color-text-muted);
  }

  .hint {
    color: var(--color-text-muted);
    font-size: 12px;
    margin-bottom: 24px;
  }

  .editor-info {
    padding: 16px 24px;
    background: var(--color-bg-surface);
    border-radius: 8px;
    border: 1px solid var(--color-border-subtle);
  }

  .editor-info p {
    margin: 4px 0;
    font-size: 12px;
    color: var(--color-text-secondary);
  }

  .editor-info .features {
    color: var(--color-text-muted);
    font-size: 11px;
  }

  .file-viewer {
    flex: 1;
    overflow: auto;
    display: flex;
    flex-direction: column;
  }

  .markdown-preview pre {
    padding: 16px;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    line-height: 1.7;
    white-space: pre-wrap;
    color: var(--color-text-primary);
  }
</style>

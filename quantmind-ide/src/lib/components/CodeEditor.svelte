<script lang="ts">
  import { onMount, createEventDispatcher } from 'svelte';
  import hljs from 'highlight.js';
  import { theme, currentTheme } from '../stores/themeStore';
  import { Save, FileText, Copy, Download, Upload, X, Edit3, Eye, History } from 'lucide-svelte';
  import { fileHistoryManager } from '../services/fileHistoryManager';
  import type { AgentType } from '../stores/chatStore';

  export let content = '';
  export let language = 'plaintext';
  export let filename = 'untitled';
  export let readOnly = false;
  export let showLineNumbers = true;
  export let filePath = '';
  export let fileId = '';
  export let agent: AgentType = 'copilot';

  const dispatch = createEventDispatcher();
  
  // Track previous content for diff computation
  let previousContent = '';
  let showHistoryPanel = false;

  let editorElement: HTMLTextAreaElement;
  let highlightedContent = '';
  let cursorPosition = 0;
  let isFullscreen = false;
  let currentThemeColors = $theme.colors;
  let currentThemeEffects = $theme.effects;
  
  // Subscribe to theme changes
  $: if ($theme) {
    currentThemeColors = $theme.colors;
    currentThemeEffects = $theme.effects;
    updateHighlighting();
    updateThemeStyles();
  }

  // Language detection and highlighting
  function detectLanguage(filename: string): string {
    const ext = filename.split('.').pop()?.toLowerCase();
    const langMap: Record<string, string> = {
      'js': 'javascript',
      'ts': 'typescript',
      'py': 'python',
      'mql5': 'cpp',
      'mq5': 'cpp',
      'mqh': 'cpp',
      'cpp': 'cpp',
      'c': 'cpp',
      'h': 'cpp',
      'java': 'java',
      'cs': 'csharp',
      'php': 'php',
      'rb': 'ruby',
      'go': 'go',
      'rs': 'rust',
      'sql': 'sql',
      'json': 'json',
      'xml': 'xml',
      'html': 'html',
      'css': 'css',
      'scss': 'scss',
      'less': 'less',
      'md': 'markdown',
      'yaml': 'yaml',
      'yml': 'yaml',
      'sh': 'bash',
      'bash': 'bash',
      'zsh': 'bash',
      'dockerfile': 'dockerfile'
    };
    return langMap[ext || ''] || language;
  }

  function updateHighlighting() {
    const detectedLang = detectLanguage(filename);
    try {
      const result = hljs.highlight(content, { language: detectedLang });
      highlightedContent = applyThemeSyntax(result.value);
    } catch {
      highlightedContent = applyThemeSyntax(content);
    }
  }
  
  function applyThemeSyntax(code: string): string {
    if (!currentThemeColors.syntax) return code;
    
    // Apply theme-specific syntax highlighting
    let themedCode = code;
    
    // Replace hljs classes with theme colors
    themedCode = themedCode.replace(/class="hljs-keyword"/g, `style="color: ${currentThemeColors.syntax.keyword}"`);
    themedCode = themedCode.replace(/class="hljs-string"/g, `style="color: ${currentThemeColors.syntax.string}"`);
    themedCode = themedCode.replace(/class="hljs-number"/g, `style="color: ${currentThemeColors.syntax.number}"`);
    themedCode = themedCode.replace(/class="hljs-comment"/g, `style="color: ${currentThemeColors.syntax.comment}"`);
    themedCode = themedCode.replace(/class="hljs-function"/g, `style="color: ${currentThemeColors.syntax.function}"`);
    themedCode = themedCode.replace(/class="hljs-variable"/g, `style="color: ${currentThemeColors.syntax.variable}"`);
    themedCode = themedCode.replace(/class="hljs-operator"/g, `style="color: ${currentThemeColors.syntax.operator}"`);
    
    return themedCode;
  }
  
  function updateThemeStyles() {
    if (!editorElement) return;
    
    // Update CSS variables for theme
    const root = document.documentElement;
    root.style.setProperty('--editor-bg', currentThemeColors.syntax.background);
    root.style.setProperty('--editor-text', currentThemeColors.text.primary);
    root.style.setProperty('--editor-accent', currentThemeColors.accent.primary);
  }

  function handleInput() {
    content = editorElement.value;
    cursorPosition = editorElement.selectionStart;
    updateHighlighting();
    dispatch('change', { content, filename });
  }

  function handleScroll() {
    const lineNumbers = document.querySelector('.line-numbers');
    const editor = document.querySelector('.editor-textarea');
    const highlight = document.querySelector('.highlighted-code');
    
    if (lineNumbers && editor && highlight) {
      lineNumbers.scrollTop = editor.scrollTop;
      highlight.scrollTop = editor.scrollTop;
      highlight.scrollLeft = editor.scrollLeft;
    }
  }

  function saveFile() {
    // Record the operation in file history before saving
    if (fileId && content !== previousContent) {
      const action = previousContent === '' ? 'created' : 'modified';
      fileHistoryManager.recordOperation(
        fileId || `file_${filename}_${Date.now()}`,
        filename,
        filePath || filename,
        agent,
        action,
        content,
        previousContent || undefined
      );
      previousContent = content;
    }
    dispatch('save', { content, filename });
  }
  
  function toggleHistoryPanel() {
    showHistoryPanel = !showHistoryPanel;
    dispatch('toggle-history', { showHistoryPanel });
  }
  
  function revertToVersion(versionContent: string) {
    const oldContent = content;
    content = versionContent;
    previousContent = versionContent;
    updateHighlighting();
    dispatch('revert', { content: versionContent, previousContent: oldContent, filename });
  }
  
  function handleRevertFromHistory(event: CustomEvent) {
    const { versionContent } = event.detail;
    revertToVersion(versionContent);
    showHistoryPanel = false;
  }

  function copyContent() {
    navigator.clipboard.writeText(content);
  }

  function downloadFile() {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  function toggleFullscreen() {
    isFullscreen = !isFullscreen;
  }

  function getLineNumbers(): string {
    const lines = content.split('\n');
    return lines.map((_, i) => i + 1).join('\n');
  }

  $: if (content) updateHighlighting();
  $: if (filename) updateHighlighting();

  onMount(() => {
    updateHighlighting();
    updateThemeStyles();
    // Initialize previous content for tracking changes
    previousContent = content;
  });
</script>

<div class="code-editor" 
     class:fullscreen={isFullscreen}
     class:scanlines={currentThemeEffects.scanlines}
     class:glow={currentThemeEffects.glow}
     class:glass={currentThemeEffects.glass}
     data-theme={$theme.name}>
  <!-- Editor Header -->
  <div class="editor-header">
    <div class="file-info">
      <FileText size={16} />
      <span class="filename">{filename}</span>
      <span class="language-badge">{detectLanguage(filename)}</span>
    </div>
    
    <div class="editor-actions">
      {#if !readOnly}
        <button class="action-btn" on:click={saveFile} title="Save (Ctrl+S)">
          <Save size={14} />
        </button>
      {/if}
      <button class="action-btn" on:click={copyContent} title="Copy">
        <Copy size={14} />
      </button>
      <button class="action-btn" on:click={downloadFile} title="Download">
        <Download size={14} />
      </button>
      {#if fileId}
        <button class="action-btn" on:click={toggleHistoryPanel} title="File History" class:active={showHistoryPanel}>
          <History size={14} />
        </button>
      {/if}
      <button class="action-btn" on:click={toggleFullscreen} title="Toggle Fullscreen">
        {#if isFullscreen}
          <X size={14} />
        {:else}
          <Eye size={14} />
        {/if}
      </button>
    </div>
  </div>

  <!-- Editor Content -->
  <div class="editor-content">
    {#if showLineNumbers}
      <div class="line-numbers">
        <pre>{getLineNumbers()}</pre>
      </div>
    {/if}
    
    <div class="code-container">
      <div class="highlighted-code">
        <pre>{@html highlightedContent}</pre>
      </div>
      <textarea
        bind:this={editorElement}
        class="editor-textarea"
        class:readonly={readOnly}
        bind:value={content}
        on:input={handleInput}
        on:scroll={handleScroll}
        on:keydown={(e) => {
          if (e.key === 's' && e.ctrlKey && !readOnly) {
            e.preventDefault();
            saveFile();
          }
        }}
        spellcheck="false"
        placeholder="Start typing..."
      ></textarea>
    </div>
  </div>

  <!-- Status Bar -->
  <div class="status-bar">
    <div class="status-left">
      <span class="line-info">Line {content.substring(0, cursorPosition).split('\n').length}, Column {cursorPosition - content.lastIndexOf('\n', cursorPosition - 1)}</span>
      <span class="char-count">{content.length} characters</span>
    </div>
    <div class="status-right">
      <span class="encoding">UTF-8</span>
      <span class="eol">LF</span>
    </div>
  </div>
</div>

<style>
  .code-editor {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--editor-bg, var(--bg-primary));
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    overflow: hidden;
    position: relative;
  }

  .code-editor.fullscreen {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 9999;
    border-radius: 0;
  }

  /* Theme effects */
  .code-editor.scanlines::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(
      transparent 50%,
      rgba(0, 255, 0, 0.03) 50%
    );
    background-size: 100% 4px;
    pointer-events: none;
    z-index: 1;
  }

  .code-editor.glow {
    box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
  }

  .code-editor.glass {
    backdrop-filter: blur(10px);
    background: rgba(10, 10, 10, 0.8);
  }

  .editor-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 16px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
    position: relative;
    z-index: 2;
  }

  .file-info {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: var(--text-secondary);
  }

  .filename {
    font-weight: 600;
    color: var(--text-primary);
  }

  .language-badge {
    padding: 2px 8px;
    background: var(--accent-primary);
    color: white;
    border-radius: 4px;
    font-size: 11px;
    text-transform: uppercase;
    animation: pulse 2s infinite;
  }

  .editor-actions {
    display: flex;
    gap: 4px;
  }

  .action-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: transparent;
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  }

  .action-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 255, 0, 0.2);
  }

  .action-btn.active {
    background: var(--accent-primary);
    color: #000;
    box-shadow: 0 0 8px rgba(0, 255, 0, 0.4);
  }

  .editor-content {
    display: flex;
    flex: 1;
    overflow: hidden;
    position: relative;
  }

  .line-numbers {
    width: 50px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-subtle);
    overflow: hidden;
    position: relative;
    z-index: 2;
  }

  .line-numbers pre {
    margin: 0;
    padding: 16px 8px;
    font-family: 'Fira Code', 'Monaco', 'Menlo', 'Courier New', monospace;
    font-size: 14px;
    line-height: 1.6;
    color: var(--text-muted);
    text-align: right;
    user-select: none;
  }

  .code-container {
    flex: 1;
    position: relative;
    overflow: auto;
  }

  .highlighted-code {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    pointer-events: none;
    overflow: auto;
    z-index: 1;
  }

  .highlighted-code pre {
    margin: 0;
    padding: 16px;
    font-family: 'Fira Code', 'Monaco', 'Menlo', 'Courier New', monospace;
    font-size: 14px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-wrap: break-word;
    background: transparent;
    color: var(--editor-text);
  }

  /* Enhanced block highlighting */
  .highlighted-code code {
    display: block;
    position: relative;
  }

  .highlighted-code code::before {
    content: '';
    position: absolute;
    top: 0;
    left: -1000px;
    right: -1000px;
    bottom: 0;
    background: linear-gradient(
      90deg,
      transparent 0%,
      rgba(0, 255, 0, 0.02) 10%,
      rgba(0, 255, 0, 0.02) 90%,
      transparent 100%
    );
    pointer-events: none;
    z-index: -1;
  }

  .editor-textarea {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    padding: 16px;
    font-family: 'Fira Code', 'Monaco', 'Menlo', 'Courier New', monospace;
    font-size: 14px;
    line-height: 1.6;
    background: transparent;
    color: transparent;
    border: none;
    outline: none;
    resize: none;
    white-space: pre-wrap;
    word-wrap: break-word;
    caret-color: var(--editor-accent);
    z-index: 3;
  }

  .editor-textarea.readonly {
    caret-color: transparent;
  }

  .status-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 16px;
    background: var(--bg-secondary);
    border-top: 1px solid var(--border-subtle);
    font-size: 11px;
    color: var(--text-muted);
    position: relative;
    z-index: 2;
  }

  .status-left, .status-right {
    display: flex;
    gap: 12px;
  }

  /* Animations */
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }

  @keyframes glow {
    0%, 100% { box-shadow: 0 0 5px var(--editor-accent); }
    50% { box-shadow: 0 0 20px var(--editor-accent); }
  }

  /* Theme-specific overrides */
  :global(.hljs) {
    background: transparent !important;
  }

  /* Trading Terminal theme specific */
  .code-editor[data-theme="trading-terminal"] .language-badge {
    background: #00ff00;
    color: #000000;
    font-weight: bold;
    text-shadow: 0 0 5px #00ff00;
  }

  .code-editor[data-theme="trading-terminal"] .action-btn:hover {
    box-shadow: 0 4px 12px rgba(0, 255, 0, 0.4);
  }

  /* Matrix theme specific */
  .code-editor[data-theme="matrix"] .language-badge {
    background: #00ff00;
    color: #000000;
    font-family: 'Courier New', monospace;
  }

  /* Cyberpunk theme specific */
  .code-editor[data-theme="cyberpunk"] .language-badge {
    background: linear-gradient(45deg, #00ffff, #ff00ff);
    background-clip: text;
    -webkit-background-clip: text;
    color: transparent;
    border: 1px solid #ff00ff;
  }

  /* Ambient theme specific */
  .code-editor[data-theme="ambient"] .action-btn:hover {
    box-shadow: 0 4px 12px rgba(255, 147, 41, 0.3);
  }
</style>

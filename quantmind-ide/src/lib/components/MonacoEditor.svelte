<script lang="ts">
  import { run } from 'svelte/legacy';

  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import { theme, currentTheme } from '../stores/themeStore';
  import { Save, FileText, Copy, Download, Upload, X, Edit3, Eye, History, GitBranch, Play, Bug } from 'lucide-svelte';
  import { fileHistoryManager } from '../services/fileHistoryManager';
  import type { AgentType } from '../stores/chatStore';
  import { registerMQL5Language, getMQL5ThemeColors } from '../monaco/mql5-language';
  import { registerIntelliSense } from '../monaco/intellisense-provider';

  interface Props {
    content?: string;
    language?: string;
    filename?: string;
    readOnly?: boolean;
    showLineNumbers?: boolean;
    filePath?: string;
    fileId?: string;
    agent?: AgentType;
    breakpoints?: number[];
  }

  let {
    content = $bindable(''),
    language = 'plaintext',
    filename = 'untitled',
    readOnly = false,
    showLineNumbers = true,
    filePath = '',
    fileId = '',
    agent = 'copilot',
    breakpoints = []
  }: Props = $props();

  const dispatch = createEventDispatcher();

  let editorContainer: HTMLDivElement = $state();
  let editor: any = $state(null); // Monaco editor instance
  let monaco: any = $state(null); // Monaco module

  let previousContent = '';
  let isFullscreen = $state(false);
  let currentBreakpoints = new Set<number>();

  // Cursor position tracking
  let cursorLine = $state(1);
  let cursorColumn = $state(1);

  // Git status decorations
  let gitDecorations: string[] = [];
  let diffEditor: any = null;
  let showDiff = false;

  // Debug state
  let isDebugging = false;
  let debugLine: number | null = null;

  async function initMonaco() {
    try {
      // Dynamically import Monaco
      const monacoModule = await import('monaco-editor');
      monaco = monacoModule;

      // Register MQL5 language
      registerMQL5Language(monaco);

      // Register IntelliSense providers
      registerIntelliSense(monaco);

      // Define custom theme based on app theme
      const themeType = $theme.name.includes('dark') || $theme.name.includes('matrix') ||
        $theme.name.includes('trading') ? 'dark' : 'light';

      const customTheme = getMQL5ThemeColors(themeType);
      monaco.editor.defineTheme('quantmindx-theme', customTheme);

      // Detect language from filename
      const detectedLang = detectLanguage(filename);

      // Create editor
      editor = monaco.editor.create(editorContainer, {
        value: content,
        language: detectedLang,
        theme: 'quantmindx-theme',
        readOnly: readOnly,
        lineNumbers: showLineNumbers ? 'on' : 'off',
        minimap: { enabled: true },
        fontSize: 14,
        fontFamily: "'JetBrains Mono', 'Fira Code', 'Monaco', 'Menlo', monospace",
        fontLigatures: true,
        scrollBeyondLastLine: false,
        automaticLayout: true,
        tabSize: 4,
        insertSpaces: true,
        wordWrap: 'on',
        renderWhitespace: 'selection',
        bracketPairColorization: { enabled: true },
        guides: {
          bracketPairs: true,
          indentation: true
        },
        cursorBlinking: 'smooth',
        cursorSmoothCaretAnimation: 'on',
        smoothScrolling: true,
        padding: { top: 16 },
        suggest: {
          showKeywords: true,
          showSnippets: true,
          showFunctions: true,
          showVariables: true,
          showConstants: true
        },
        quickSuggestions: {
          other: true,
          comments: false,
          strings: false
        }
      });

      // Set up event listeners
      editor.onDidChangeModelContent(() => {
        content = editor.getValue();
        dispatch('change', { content, filename });
      });

      editor.onDidChangeCursorPosition((e: any) => {
        cursorLine = e.position.lineNumber;
        cursorColumn = e.position.column;
      });

      // Register keyboard shortcuts
      editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
        if (!readOnly) saveFile();
      });

      editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyF, () => {
        editor.trigger('keyboard', 'actions.find', null);
      });

      editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyH, () => {
        editor.trigger('keyboard', 'editor.action.startFindReplaceAction', null);
      });

      editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyG, () => {
        editor.trigger('keyboard', 'editor.action.gotoLine', null);
      });

      // Apply existing breakpoints
      breakpoints.forEach(bp => addBreakpoint(bp));

    } catch (error) {
      console.error('Failed to initialize Monaco Editor:', error);
      // Fallback to textarea if Monaco fails
    }
  }

  function detectLanguage(filename: string): string {
    const ext = filename.split('.').pop()?.toLowerCase();
    const langMap: Record<string, string> = {
      'js': 'javascript',
      'ts': 'typescript',
      'jsx': 'javascript',
      'tsx': 'typescript',
      'py': 'python',
      'mql5': 'mql5',
      'mq5': 'mql5',
      'mqh': 'mql5',
      'cpp': 'cpp',
      'c': 'c',
      'h': 'cpp',
      'hpp': 'cpp',
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
      'htm': 'html',
      'css': 'css',
      'scss': 'scss',
      'less': 'less',
      'md': 'markdown',
      'yaml': 'yaml',
      'yml': 'yaml',
      'sh': 'shell',
      'bash': 'shell',
      'dockerfile': 'dockerfile'
    };
    return langMap[ext || ''] || language;
  }

  function saveFile() {
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
    setTimeout(() => {
      if (editor) editor.layout();
    }, 100);
  }

  function formatDocument() {
    if (editor) {
      editor.trigger('keyboard', 'editor.action.formatDocument', null);
    }
  }

  function addBreakpoint(lineNumber: number) {
    if (!editor) return;

    currentBreakpoints.add(lineNumber);

    const decorations = editor.deltaDecorations([], [
      {
        range: new monaco.Range(lineNumber, 1, lineNumber, 1),
        options: {
          isWholeLine: true,
          glyphMarginClassName: 'breakpoint-glyph',
          glyphMarginHoverMessage: { value: `Breakpoint at line ${lineNumber}` }
        }
      }
    ]);

    gitDecorations.push(...decorations);
    dispatch('breakpoint', { lineNumber, action: 'add' });
  }

  function removeBreakpoint(lineNumber: number) {
    if (!editor) return;
    currentBreakpoints.delete(lineNumber);
    dispatch('breakpoint', { lineNumber, action: 'remove' });
  }

  function toggleBreakpoint(lineNumber: number) {
    if (currentBreakpoints.has(lineNumber)) {
      removeBreakpoint(lineNumber);
    } else {
      addBreakpoint(lineNumber);
    }
  }

  function setDebugLine(lineNumber: number | null) {
    if (!editor) return;

    // Clear previous debug line
    if (debugLine !== null) {
      gitDecorations = editor.deltaDecorations(gitDecorations, []);
    }

    debugLine = lineNumber;

    if (lineNumber !== null) {
      const decorations = editor.deltaDecorations([], [
        {
          range: new monaco.Range(lineNumber, 1, lineNumber, 1),
          options: {
            isWholeLine: true,
            className: 'debug-line-highlight',
            glyphMarginClassName: 'debug-glyph'
          }
        }
      ]);
      gitDecorations.push(...decorations);
    }
  }

  function applyGitDecorations(changes: Array<{line: number, type: 'added' | 'modified' | 'deleted'}>) {
    if (!editor || !monaco) return;

    const decorations = changes.map(change => ({
      range: new monaco.Range(change.line, 1, change.line, 1),
      options: {
        isWholeLine: true,
        className: change.type === 'added' ? 'git-added-line' :
          change.type === 'deleted' ? 'git-deleted-line' : 'git-modified-line',
        glyphMarginClassName: change.type === 'added' ? 'git-added-glyph' :
          change.type === 'deleted' ? 'git-deleted-glyph' : 'git-modified-glyph'
      }
    }));

    gitDecorations = editor.deltaDecorations(gitDecorations, decorations);
  }

  function goToLine(lineNumber: number) {
    if (editor) {
      editor.revealLineInCenter(lineNumber);
      editor.setPosition({ lineNumber, column: 1 });
      editor.focus();
    }
  }

  function setSelection(startLine: number, startCol: number, endLine: number, endCol: number) {
    if (editor) {
      editor.setSelection({
        startLineNumber: startLine,
        startColumn: startCol,
        endLineNumber: endLine,
        endColumn: endCol
      });
    }
  }

  // Watch for content changes from outside
  run(() => {
    if (editor && content !== editor.getValue()) {
      const position = editor.getPosition();
      editor.setValue(content);
      if (position) editor.setPosition(position);
    }
  });

  // Watch for theme changes
  run(() => {
    if (editor && monaco && $theme) {
      const themeType = $theme.name.includes('dark') || $theme.name.includes('matrix') ||
        $theme.name.includes('trading') ? 'dark' : 'light';
      const customTheme = getMQL5ThemeColors(themeType);
      monaco.editor.defineTheme('quantmindx-theme', customTheme);
      monaco.editor.setTheme('quantmindx-theme');
    }
  });

  onMount(() => {
    initMonaco();
    previousContent = content;
  });

  onDestroy(() => {
    if (editor) {
      editor.dispose();
    }
    if (diffEditor) {
      diffEditor.dispose();
    }
  });

  // Expose methods
  export function getEditor() { return editor; }
  export function getMonaco() { return monaco; }
</script>

<div class="monaco-editor-wrapper"
     class:fullscreen={isFullscreen}
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
        <button class="action-btn" onclick={saveFile} title="Save (Ctrl+S)">
          <Save size={14} />
        </button>
        <button class="action-btn" onclick={formatDocument} title="Format Document">
          <Edit3 size={14} />
        </button>
      {/if}
      <button class="action-btn" onclick={copyContent} title="Copy">
        <Copy size={14} />
      </button>
      <button class="action-btn" onclick={downloadFile} title="Download">
        <Download size={14} />
      </button>
      <button class="action-btn" onclick={() => dispatch('toggle-git')} title="Git Status">
        <GitBranch size={14} />
      </button>
      <button class="action-btn" onclick={() => dispatch('toggle-debug')} title="Debug">
        <Bug size={14} />
      </button>
      <button class="action-btn" onclick={toggleFullscreen} title="Toggle Fullscreen">
        {#if isFullscreen}
          <X size={14} />
        {:else}
          <Eye size={14} />
        {/if}
      </button>
    </div>
  </div>

  <!-- Monaco Editor Container -->
  <div class="editor-container" bind:this={editorContainer}></div>

  <!-- Status Bar -->
  <div class="status-bar">
    <div class="status-left">
      <span class="line-info">Ln {cursorLine}, Col {cursorColumn}</span>
      <span class="char-count">{content.length} chars</span>
      {#if currentBreakpoints.size > 0}
        <span class="breakpoint-count">{currentBreakpoints.size} breakpoints</span>
      {/if}
    </div>
    <div class="status-right">
      <span class="encoding">UTF-8</span>
      <span class="eol">LF</span>
      <span class="language">{detectLanguage(filename)}</span>
    </div>
  </div>
</div>

<style>
  .monaco-editor-wrapper {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    overflow: hidden;
    position: relative;
  }

  .monaco-editor-wrapper.fullscreen {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 9999;
    border-radius: 0;
  }

  .editor-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 16px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
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
    transition: all 0.2s ease;
  }

  .action-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
    transform: translateY(-1px);
  }

  .editor-container {
    flex: 1;
    min-height: 200px;
    overflow: hidden;
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
    z-index: 2;
  }

  .status-left, .status-right {
    display: flex;
    gap: 12px;
  }

  .breakpoint-count {
    color: var(--accent-warning);
  }

  /* Git decoration styles */
  :global(.git-added-line) {
    background: rgba(0, 255, 0, 0.15) !important;
  }

  :global(.git-modified-line) {
    background: rgba(255, 165, 0, 0.15) !important;
  }

  :global(.git-deleted-line) {
    background: rgba(255, 0, 0, 0.15) !important;
  }

  :global(.git-added-glyph) {
    background: #00ff00;
    width: 4px !important;
    margin-left: 3px;
  }

  :global(.git-modified-glyph) {
    background: #ffa500;
    width: 4px !important;
    margin-left: 3px;
  }

  :global(.git-deleted-glyph) {
    background: #ff0000;
    width: 4px !important;
    margin-left: 3px;
  }

  /* Debug styles */
  :global(.debug-line-highlight) {
    background: rgba(255, 255, 0, 0.2) !important;
    border-left: 3px solid #ffff00;
  }

  :global(.debug-glyph) {
    background: #ffff00;
    width: 10px !important;
    height: 10px !important;
    border-radius: 50%;
    margin-left: 2px;
  }

  /* Breakpoint styles */
  :global(.breakpoint-glyph) {
    background: #ff4444;
    width: 10px !important;
    height: 10px !important;
    border-radius: 50%;
    margin-left: 2px;
    cursor: pointer;
  }
</style>

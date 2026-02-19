<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import { X, GitBranch, ChevronDown, ChevronUp, RefreshCw, Check, XCircle } from 'lucide-svelte';
  import { theme } from '../stores/themeStore';
  import { getMQL5ThemeColors } from '../monaco/mql5-language';

  export let originalContent = '';
  export let modifiedContent = '';
  export let filename = 'diff';
  export let language = 'mql5';

  const dispatch = createEventDispatcher();

  let diffContainer: HTMLDivElement;
  let diffEditor: any = null;
  let monaco: any = null;
  let diffSummary = { additions: 0, deletions: 0, modifications: 0 };
  let showUnified = false;

  async function initDiffEditor() {
    try {
      const monacoModule = await import('monaco-editor');
      monaco = monacoModule;

      const themeType = $theme.name.includes('dark') ? 'dark' : 'light';
      const customTheme = getMQL5ThemeColors(themeType);
      monaco.editor.defineTheme('quantmindx-diff-theme', customTheme);

      diffEditor = monaco.editor.createDiffEditor(diffContainer, {
        theme: 'quantmindx-diff-theme',
        readOnly: true,
        renderSideBySide: !showUnified,
        originalEditable: false,
        enableSplitViewResizing: true,
        ignoreTrimWhitespace: false,
        renderIndicators: true,
        minimap: { enabled: false },
        scrollBeyondLastLine: false,
        fontSize: 13,
        fontFamily: "'JetBrains Mono', monospace",
        diffAlgorithm: 'advanced',
        renderOverviewRuler: true,
        overviewRulerBorder: false,
        hideUnchangedRegions: {
          enabled: true,
          revealByClick: true
        }
      });

      // Set the diff model
      const originalModel = monaco.editor.createModel(originalContent, language);
      const modifiedModel = monaco.editor.createModel(modifiedContent, language);

      diffEditor.setModel({
        original: originalModel,
        modified: modifiedModel
      });

      // Calculate diff summary
      calculateDiffSummary();

    } catch (error) {
      console.error('Failed to initialize diff editor:', error);
    }
  }

  function calculateDiffSummary() {
    if (!diffEditor) return;

    const diff = diffEditor.getDiffLineInformationForModified(1);
    const originalLines = originalContent.split('\n');
    const modifiedLines = modifiedContent.split('\n');

    let additions = 0;
    let deletions = 0;
    let modifications = 0;

    // Simple line-based diff calculation
    const maxLines = Math.max(originalLines.length, modifiedLines.length);

    for (let i = 0; i < maxLines; i++) {
      const origLine = originalLines[i];
      const modLine = modifiedLines[i];

      if (origLine === undefined) {
        additions++;
      } else if (modLine === undefined) {
        deletions++;
      } else if (origLine !== modLine) {
        modifications++;
      }
    }

    diffSummary = { additions, deletions, modifications };
  }

  function toggleView() {
    showUnified = !showUnified;
    if (diffEditor) {
      diffEditor.dispose();
      initDiffEditor();
    }
  }

  function refresh() {
    if (diffEditor) {
      const originalModel = monaco.editor.createModel(originalContent, language);
      const modifiedModel = monaco.editor.createModel(modifiedContent, language);
      diffEditor.setModel({
        original: originalModel,
        modified: modifiedModel
      });
      calculateDiffSummary();
    }
    dispatch('refresh');
  }

  function acceptChanges() {
    dispatch('accept', { content: modifiedContent });
  }

  function rejectChanges() {
    dispatch('reject', { content: originalContent });
  }

  function close() {
    dispatch('close');
  }

  // Watch for content changes
  $: if (diffEditor && (originalContent || modifiedContent)) {
    const originalModel = monaco?.editor.createModel(originalContent, language);
    const modifiedModel = monaco?.editor.createModel(modifiedContent, language);
    if (originalModel && modifiedModel) {
      diffEditor.setModel({
        original: originalModel,
        modified: modifiedModel
      });
      calculateDiffSummary();
    }
  }

  // Watch for theme changes
  $: if (diffEditor && monaco && $theme) {
    const themeType = $theme.name.includes('dark') ? 'dark' : 'light';
    const customTheme = getMQL5ThemeColors(themeType);
    monaco.editor.defineTheme('quantmindx-diff-theme', customTheme);
    monaco.editor.setTheme('quantmindx-diff-theme');
  }

  onMount(() => {
    initDiffEditor();
  });

  onDestroy(() => {
    if (diffEditor) {
      diffEditor.dispose();
    }
  });
</script>

<div class="git-diff-view">
  <!-- Header -->
  <div class="diff-header">
    <div class="diff-title">
      <GitBranch size={16} />
      <span>Diff: {filename}</span>
    </div>

    <div class="diff-stats">
      <span class="stat additions">
        <ChevronUp size={14} />
        {diffSummary.additions}
      </span>
      <span class="stat deletions">
        <ChevronDown size={14} />
        {diffSummary.deletions}
      </span>
      <span class="stat modifications">
        • {diffSummary.modifications} changed
      </span>
    </div>

    <div class="diff-actions">
      <button class="action-btn" on:click={toggleView} title="Toggle View">
        {showUnified ? 'Side by Side' : 'Unified'}
      </button>
      <button class="action-btn" on:click={refresh} title="Refresh">
        <RefreshCw size={14} />
      </button>
      <button class="action-btn accept" on:click={acceptChanges} title="Accept Changes">
        <Check size={14} />
      </button>
      <button class="action-btn reject" on:click={rejectChanges} title="Reject Changes">
        <XCircle size={14} />
      </button>
      <button class="action-btn" on:click={close} title="Close">
        <X size={14} />
      </button>
    </div>
  </div>

  <!-- Diff Editor Container -->
  <div class="diff-container" bind:this={diffContainer}></div>

  <!-- Legend -->
  <div class="diff-legend">
    <div class="legend-item">
      <span class="legend-color added"></span>
      <span>Added</span>
    </div>
    <div class="legend-item">
      <span class="legend-color removed"></span>
      <span>Removed</span>
    </div>
    <div class="legend-item">
      <span class="legend-color modified"></span>
      <span>Modified</span>
    </div>
  </div>
</div>

<style>
  .git-diff-view {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    overflow: hidden;
  }

  .diff-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 16px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .diff-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .diff-stats {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 12px;
  }

  .stat {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .stat.additions {
    color: #4caf50;
  }

  .stat.deletions {
    color: #f44336;
  }

  .stat.modifications {
    color: var(--text-muted);
  }

  .diff-actions {
    display: flex;
    gap: 4px;
  }

  .action-btn {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    background: transparent;
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .action-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .action-btn.accept:hover {
    background: rgba(76, 175, 80, 0.2);
    border-color: #4caf50;
    color: #4caf50;
  }

  .action-btn.reject:hover {
    background: rgba(244, 67, 54, 0.2);
    border-color: #f44336;
    color: #f44336;
  }

  .diff-container {
    flex: 1;
    min-height: 300px;
    overflow: hidden;
  }

  .diff-legend {
    display: flex;
    justify-content: center;
    gap: 24px;
    padding: 8px;
    background: var(--bg-secondary);
    border-top: 1px solid var(--border-subtle);
    font-size: 11px;
    color: var(--text-muted);
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .legend-color {
    width: 12px;
    height: 12px;
    border-radius: 2px;
  }

  .legend-color.added {
    background: rgba(76, 175, 80, 0.4);
    border: 1px solid #4caf50;
  }

  .legend-color.removed {
    background: rgba(244, 67, 54, 0.4);
    border: 1px solid #f44336;
  }

  .legend-color.modified {
    background: rgba(255, 152, 0, 0.4);
    border: 1px solid #ff9800;
  }
</style>

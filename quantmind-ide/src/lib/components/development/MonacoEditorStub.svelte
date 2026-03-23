<script lang="ts">
  /**
   * Monaco Editor Stub
   *
   * Variant-specific code viewer with edit mode and action bar.
   * Integrates with existing MonacoEditor for MQL5 syntax highlighting.
   */
  import { onMount, createEventDispatcher } from 'svelte';
  import MonacoEditor from '../MonacoEditor.svelte';
  import {
    Save,
    Play,
    GitBranch,
    Eye,
    Edit3,
    ChevronRight,
    Clock,
    TrendingUp,
    FileCode
  } from 'lucide-svelte';

  interface VersionTimelineItem {
    version_tag: string;
    created_at: string;
    author: string;
    improvement_cycle: number;
    is_active: boolean;
  }

  interface Props {
    code?: string;
    readOnly?: boolean;
    language?: string;
    strategyName?: string;
    variantType?: string;
    version?: string;
    versionTimeline?: VersionTimelineItem[];
    promotionStatus?: string;
    onSave?: () => void;
    onRun?: () => void;
    onDiff?: () => void;
  }

  let {
    code = '',
    readOnly = true,
    language = 'mql5',
    strategyName = '',
    variantType = '',
    version = '1.0.0',
    versionTimeline = [],
    promotionStatus = 'development',
    onSave,
    onRun,
    onDiff
  }: Props = $props();

  let isEditMode = $state(false);
  let showDiff = $state(false);
  let editedCode = $state(code);

  function toggleEditMode() {
    isEditMode = !isEditMode;
    if (isEditMode) {
      editedCode = code;
    }
  }

  function handleSave() {
    code = editedCode;
    isEditMode = false;
    if (onSave) onSave();
  }

  function handleRun() {
    if (onRun) onRun();
  }

  function handleDiff() {
    showDiff = !showDiff;
    if (onDiff) onDiff();
  }

  function getPromotionColor(status: string): string {
    switch (status) {
      case 'live': return '#22c55e';
      case 'sit': return '#f59e0b';
      case 'paper_trading': return '#3b82f6';
      default: return '#6b7280';
    }
  }

  function getPromotionLabel(status: string): string {
    switch (status) {
      case 'live': return 'Live Trading';
      case 'sit': return 'SIT Validation';
      case 'paper_trading': return 'Paper Trading';
      default: return 'Development';
    }
  }

  function formatDate(dateStr: string): string {
    if (!dateStr) return '';
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    } catch {
      return dateStr;
    }
  }

  $effect(() => {
    if (code) {
      editedCode = code;
    }
  });
</script>

<div class="monaco-editor-stub">
  <!-- Header -->
  <div class="editor-header">
    <div class="breadcrumb">
      <FileCode size={14} />
      <span class="strategy-name">{strategyName}</span>
      <ChevronRight size={12} />
      <span class="variant-type">{variantType}</span>
      <span class="version">v{version}</span>
    </div>

    <div class="promotion-tracker" style="--status-color: {getPromotionColor(promotionStatus)}">
      <TrendingUp size={12} />
      <span>{getPromotionLabel(promotionStatus)}</span>
    </div>
  </div>

  <!-- Action Bar -->
  <div class="action-bar">
    <div class="mode-toggle">
      {#if readOnly || isEditMode}
        <button
          class="mode-btn"
          class:active={!isEditMode}
          onclick={() => isEditMode = false}
          disabled={readOnly}
        >
          <Eye size={14} />
          <span>Read</span>
        </button>
        <button
          class="mode-btn"
          class:active={isEditMode}
          onclick={toggleEditMode}
          disabled={readOnly}
        >
          <Edit3 size={14} />
          <span>Edit</span>
        </button>
      {/if}
    </div>

    {#if isEditMode}
      <div class="edit-actions">
        <button class="action-btn save" onclick={handleSave}>
          <Save size={14} />
          <span>Save</span>
        </button>
        <button class="action-btn run" onclick={handleRun}>
          <Play size={14} />
          <span>Run</span>
        </button>
        <button class="action-btn diff" onclick={handleDiff}>
          <GitBranch size={14} />
          <span>Diff</span>
        </button>
      </div>
    {/if}

    <div class="language-selector">
      <span>MQL5</span>
    </div>
  </div>

  <!-- Version Timeline -->
  {#if versionTimeline.length > 0}
    <div class="version-timeline">
      <div class="timeline-header">
        <Clock size={12} />
        <span>Improvement Cycle History</span>
      </div>
      <div class="timeline-items">
        {#each versionTimeline as item, i}
          <button
            class="timeline-item"
            class:active={item.is_active}
            onclick={() => version = item.version_tag}
          >
            <span class="version-tag">{item.version_tag}</span>
            <span class="cycle-label">v{item.improvement_cycle + 1}</span>
          </button>
          {#if i < versionTimeline.length - 1}
            <span class="timeline-arrow">→</span>
          {/if}
        {/each}
      </div>
    </div>
  {/if}

  <!-- Editor -->
  <div class="editor-content">
    {#if showDiff && isEditMode}
      <!-- Diff view would go here - simplified for now -->
      <div class="diff-placeholder">
        <GitBranch size={24} />
        <span>Diff view vs previous version</span>
      </div>
    {:else}
      <MonacoEditor
        content={isEditMode ? editedCode : code}
        language={language}
        readOnly={!isEditMode}
        filename="{strategyName}_{variantType}.mq5"
      />
    {/if}
  </div>
</div>

<style>
  .monaco-editor-stub {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: rgba(10, 15, 26, 0.95);
    border-radius: 8px;
    border: 1px solid rgba(0, 212, 255, 0.1);
    overflow: hidden;
  }

  .editor-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: rgba(8, 13, 20, 0.6);
    border-bottom: 1px solid rgba(0, 212, 255, 0.1);
  }

  .breadcrumb {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: rgba(255, 255, 255, 0.6);
  }

  .breadcrumb :global(svg) {
    color: #a855f7;
  }

  .strategy-name {
    color: #a855f7;
  }

  .variant-type {
    color: #00d4ff;
    text-transform: capitalize;
  }

  .version {
    color: rgba(255, 255, 255, 0.4);
  }

  .promotion-tracker {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: rgba(var(--status-color), 0.1);
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--status-color);
    text-transform: uppercase;
  }

  .promotion-tracker :global(svg) {
    color: var(--status-color);
  }

  .action-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 16px;
    background: rgba(8, 13, 20, 0.4);
    border-bottom: 1px solid rgba(0, 212, 255, 0.05);
  }

  .mode-toggle {
    display: flex;
    gap: 4px;
  }

  .mode-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: transparent;
    border: 1px solid rgba(0, 212, 255, 0.1);
    border-radius: 6px;
    color: rgba(255, 255, 255, 0.5);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .mode-btn:hover:not(:disabled) {
    background: rgba(0, 212, 255, 0.05);
    border-color: rgba(0, 212, 255, 0.3);
  }

  .mode-btn.active {
    background: rgba(0, 212, 255, 0.1);
    border-color: #00d4ff;
    color: #00d4ff;
  }

  .mode-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .edit-actions {
    display: flex;
    gap: 8px;
  }

  .action-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: rgba(0, 212, 255, 0.08);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 6px;
    color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .action-btn:hover {
    background: rgba(0, 212, 255, 0.15);
    border-color: rgba(0, 212, 255, 0.4);
  }

  .action-btn.save:hover {
    background: rgba(34, 197, 94, 0.15);
    border-color: rgba(34, 197, 94, 0.4);
    color: #22c55e;
  }

  .action-btn.run:hover {
    background: rgba(168, 85, 247, 0.15);
    border-color: rgba(168, 85, 247, 0.4);
    color: #a855f7;
  }

  .language-selector {
    padding: 4px 8px;
    background: rgba(168, 85, 247, 0.1);
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #a855f7;
    text-transform: uppercase;
  }

  .version-timeline {
    padding: 12px 16px;
    background: rgba(8, 13, 20, 0.3);
    border-bottom: 1px solid rgba(0, 212, 255, 0.05);
  }

  .timeline-header {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: rgba(255, 255, 255, 0.4);
    text-transform: uppercase;
  }

  .timeline-items {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .timeline-item {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid transparent;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: rgba(255, 255, 255, 0.5);
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .timeline-item:hover {
    background: rgba(0, 212, 255, 0.1);
    border-color: rgba(0, 212, 255, 0.3);
  }

  .timeline-item.active {
    background: rgba(0, 212, 255, 0.15);
    border-color: #00d4ff;
    color: #00d4ff;
  }

  .cycle-label {
    font-size: 9px;
    color: rgba(255, 255, 255, 0.3);
  }

  .timeline-arrow {
    color: rgba(255, 255, 255, 0.2);
    font-size: 10px;
  }

  .editor-content {
    flex: 1;
    min-height: 0;
  }

  .diff-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 12px;
    color: rgba(255, 255, 255, 0.3);
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }
</style>
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { X, ArrowLeft, ArrowRight, Copy, Check } from 'lucide-svelte';
  import type { FileVersion } from '../services/fileHistoryManager';

  export let oldContent: string;
  export let newContent: string;
  export let oldLabel: string = 'Previous';
  export let newLabel: string = 'Current';
  export let oldVersion: FileVersion | null = null;
  export let newVersion: FileVersion | null = null;

  const dispatch = createEventDispatcher();

  let viewMode: 'split' | 'unified' = 'split';
  let copied = false;

  interface DiffLine {
    type: 'added' | 'removed' | 'unchanged';
    oldLineNumber?: number;
    newLineNumber?: number;
    content: string;
  }

  function computeDiff(oldLines: string[], newLines: string[]): DiffLine[] {
    const diff: DiffLine[] = [];
    const maxLen = Math.max(oldLines.length, newLines.length);
    
    // Simple line-by-line diff
    // For a more sophisticated diff, consider using a library like 'diff'
    let oldIdx = 0;
    let newIdx = 0;
    
    while (oldIdx < oldLines.length || newIdx < newLines.length) {
      const oldLine = oldLines[oldIdx];
      const newLine = newLines[newIdx];
      
      if (oldIdx >= oldLines.length) {
        // Remaining new lines are additions
        diff.push({
          type: 'added',
          newLineNumber: newIdx + 1,
          content: newLine
        });
        newIdx++;
      } else if (newIdx >= newLines.length) {
        // Remaining old lines are removals
        diff.push({
          type: 'removed',
          oldLineNumber: oldIdx + 1,
          content: oldLine
        });
        oldIdx++;
      } else if (oldLine === newLine) {
        // Lines match
        diff.push({
          type: 'unchanged',
          oldLineNumber: oldIdx + 1,
          newLineNumber: newIdx + 1,
          content: oldLine
        });
        oldIdx++;
        newIdx++;
      } else {
        // Check if the new line appears later in old content
        const oldLineInNew = newLines.slice(newIdx).indexOf(oldLine);
        const newLineInOld = oldLines.slice(oldIdx).indexOf(newLine);
        
        if (oldLineInNew === -1 && newLineInOld === -1) {
          // Both lines are unique, show as change
          diff.push({
            type: 'removed',
            oldLineNumber: oldIdx + 1,
            content: oldLine
          });
          diff.push({
            type: 'added',
            newLineNumber: newIdx + 1,
            content: newLine
          });
          oldIdx++;
          newIdx++;
        } else if (oldLineInNew !== -1 && (newLineInOld === -1 || oldLineInNew <= newLineInOld)) {
          // Old line appears later in new, so additions happened
          for (let i = 0; i < oldLineInNew; i++) {
            diff.push({
              type: 'added',
              newLineNumber: newIdx + 1,
              content: newLines[newIdx]
            });
            newIdx++;
          }
        } else if (newLineInOld !== -1) {
          // New line appears later in old, so removals happened
          for (let i = 0; i < newLineInOld; i++) {
            diff.push({
              type: 'removed',
              oldLineNumber: oldIdx + 1,
              content: oldLines[oldIdx]
            });
            oldIdx++;
          }
        } else {
          diff.push({
            type: 'removed',
            oldLineNumber: oldIdx + 1,
            content: oldLine
          });
          diff.push({
            type: 'added',
            newLineNumber: newIdx + 1,
            content: newLine
          });
          oldIdx++;
          newIdx++;
        }
      }
    }
    
    return diff;
  }

  $: oldLines = oldContent.split('\n');
  $: newLines = newContent.split('\n');
  $: diffLines = computeDiff(oldLines, newLines);
  $: addedCount = diffLines.filter(l => l.type === 'added').length;
  $: removedCount = diffLines.filter(l => l.type === 'removed').length;

  function copyDiff() {
    const diffText = diffLines.map(line => {
      const prefix = line.type === 'added' ? '+' : line.type === 'removed' ? '-' : ' ';
      return `${prefix} ${line.content}`;
    }).join('\n');
    navigator.clipboard.writeText(diffText);
    copied = true;
    setTimeout(() => copied = false, 2000);
  }

  function formatTimestamp(date: Date | undefined): string {
    if (!date) return '';
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(new Date(date));
  }
</script>

<div class="diff-viewer">
  <div class="diff-header">
    <div class="header-left">
      <h3>Diff View</h3>
      <div class="diff-stats">
        <span class="added">+{addedCount}</span>
        <span class="removed">-{removedCount}</span>
      </div>
    </div>
    <div class="header-actions">
      <div class="view-toggle">
        <button 
          class:active={viewMode === 'split'} 
          on:click={() => viewMode = 'split'}
        >
          Split
        </button>
        <button 
          class:active={viewMode === 'unified'} 
          on:click={() => viewMode = 'unified'}
        >
          Unified
        </button>
      </div>
      <button class="action-btn" on:click={copyDiff} title="Copy diff">
        {#if copied}
          <Check size={14} />
        {:else}
          <Copy size={14} />
        {/if}
      </button>
      <button class="close-btn" on:click={() => dispatch('close')}>
        <X size={16} />
      </button>
    </div>
  </div>

  {#if viewMode === 'split'}
    <div class="diff-content split">
      <div class="diff-side old">
        <div class="side-header">
          <span class="label">{oldLabel}</span>
          {#if oldVersion}
            <span class="timestamp">{formatTimestamp(oldVersion.timestamp)}</span>
          {/if}
        </div>
        <div class="side-content">
          {#each diffLines as line}
            <div class="line {line.type}" class:removed={line.type === 'removed'} class:unchanged={line.type === 'unchanged'}>
              {#if line.type !== 'added'}
                <span class="line-number">{line.oldLineNumber || ''}</span>
                <span class="line-content">{line.content}</span>
              {:else}
                <span class="line-number"></span>
                <span class="line-content empty"></span>
              {/if}
            </div>
          {/each}
        </div>
      </div>
      <div class="diff-side new">
        <div class="side-header">
          <span class="label">{newLabel}</span>
          {#if newVersion}
            <span class="timestamp">{formatTimestamp(newVersion.timestamp)}</span>
          {/if}
        </div>
        <div class="side-content">
          {#each diffLines as line}
            <div class="line {line.type}" class:added={line.type === 'added'} class:unchanged={line.type === 'unchanged'}>
              {#if line.type !== 'removed'}
                <span class="line-number">{line.newLineNumber || ''}</span>
                <span class="line-content">{line.content}</span>
              {:else}
                <span class="line-number"></span>
                <span class="line-content empty"></span>
              {/if}
            </div>
          {/each}
        </div>
      </div>
    </div>
  {:else}
    <div class="diff-content unified">
      <div class="unified-header">
        <span>{oldLabel} → {newLabel}</span>
      </div>
      <div class="unified-content">
        {#each diffLines as line}
          <div class="line {line.type}">
            <span class="line-numbers">
              {#if line.oldLineNumber}
                <span class="old-num">{line.oldLineNumber}</span>
              {:else}
                <span class="old-num"></span>
              {/if}
              {#if line.newLineNumber}
                <span class="new-num">{line.newLineNumber}</span>
              {:else}
                <span class="new-num"></span>
              {/if}
            </span>
            <span class="prefix">
              {line.type === 'added' ? '+' : line.type === 'removed' ? '-' : ' '}
            </span>
            <span class="line-content">{line.content}</span>
          </div>
        {/each}
      </div>
    </div>
  {/if}
</div>

<style>
  .diff-viewer {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary, #1a1b26);
    border: 1px solid var(--border-subtle, #2d2d3a);
    border-radius: 8px;
    overflow: hidden;
  }

  .diff-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    background: var(--bg-secondary, #16161e);
    border-bottom: 1px solid var(--border-subtle, #2d2d3a);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary, #c0caf5);
  }

  .diff-stats {
    display: flex;
    gap: 12px;
    font-size: 12px;
    font-family: 'Fira Code', monospace;
  }

  .added {
    color: #4ade80;
  }

  .removed {
    color: #f87171;
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .view-toggle {
    display: flex;
    background: var(--bg-tertiary, #1f1f2e);
    border-radius: 4px;
    overflow: hidden;
  }

  .view-toggle button {
    padding: 6px 12px;
    background: transparent;
    border: none;
    font-size: 12px;
    color: var(--text-secondary, #a9b1d6);
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .view-toggle button.active {
    background: var(--accent-primary, #00ff00);
    color: #000;
  }

  .view-toggle button:hover:not(.active) {
    background: var(--bg-secondary, #292a3e);
  }

  .action-btn, .close-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: transparent;
    border: none;
    color: var(--text-secondary, #a9b1d6);
    cursor: pointer;
    border-radius: 4px;
    transition: all 0.15s ease;
  }

  .action-btn:hover, .close-btn:hover {
    background: var(--bg-tertiary, #292a3e);
    color: var(--text-primary, #c0caf5);
  }

  .diff-content {
    flex: 1;
    overflow: auto;
  }

  .diff-content.split {
    display: flex;
  }

  .diff-side {
    flex: 1;
    display: flex;
    flex-direction: column;
    border-right: 1px solid var(--border-subtle, #2d2d3a);
  }

  .diff-side:last-child {
    border-right: none;
  }

  .side-header, .unified-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    background: var(--bg-tertiary, #1f1f2e);
    border-bottom: 1px solid var(--border-subtle, #2d2d3a);
    font-size: 12px;
    color: var(--text-secondary, #a9b1d6);
  }

  .side-header .label {
    font-weight: 500;
  }

  .timestamp {
    font-size: 11px;
    color: var(--text-muted, #565f89);
  }

  .side-content, .unified-content {
    flex: 1;
    overflow: auto;
    font-family: 'Fira Code', 'Monaco', monospace;
    font-size: 12px;
    line-height: 1.5;
  }

  .line {
    display: flex;
    min-height: 18px;
  }

  .line.added {
    background: rgba(74, 222, 128, 0.15);
  }

  .line.removed {
    background: rgba(248, 113, 113, 0.15);
  }

  .line-number {
    min-width: 40px;
    padding: 0 8px;
    text-align: right;
    color: var(--text-muted, #565f89);
    background: var(--bg-secondary, #16161e);
    user-select: none;
    border-right: 1px solid var(--border-subtle, #2d2d3a);
  }

  .line-content {
    flex: 1;
    padding: 0 8px;
    white-space: pre;
    color: var(--text-primary, #c0caf5);
  }

  .line-content.empty {
    background: var(--bg-tertiary, #1f1f2e);
  }

  .unified-content .line-numbers {
    display: flex;
    min-width: 80px;
    background: var(--bg-secondary, #16161e);
    border-right: 1px solid var(--border-subtle, #2d2d3a);
  }

  .unified-content .old-num, .unified-content .new-num {
    min-width: 40px;
    padding: 0 8px;
    text-align: right;
    color: var(--text-muted, #565f89);
    user-select: none;
  }

  .unified-content .prefix {
    min-width: 20px;
    text-align: center;
    font-weight: bold;
  }

  .unified-content .line.added .prefix {
    color: #4ade80;
  }

  .unified-content .line.removed .prefix {
    color: #f87171;
  }

  .unified-content .line.unchanged .prefix {
    color: var(--text-muted, #565f89);
  }
</style>

<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import { 
    History, Clock, User, FileText, ArrowLeft, ArrowRight,
    RotateCcw, Eye, Trash2, ChevronDown, ChevronUp, X
  } from 'lucide-svelte';
  import { fileHistoryManager, type FileHistory, type FileVersion } from '../services/fileHistoryManager';
  import type { AgentType } from '../stores/chatStore';

  export let fileId: string;
  export let fileName: string;
  export let currentContent: string = '';

  const dispatch = createEventDispatcher();

  let history: FileHistory | null = null;
  let selectedVersion: FileVersion | null = null;
  let compareVersion: FileVersion | null = null;
  let showDiff = false;
  let expandedVersions: Set<string> = new Set();

  onMount(() => {
    loadHistory();
  });

  function loadHistory() {
    history = fileHistoryManager.getFileHistory(fileId);
  }

  function formatTimestamp(date: Date): string {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    }).format(new Date(date));
  }

  function getAgentLabel(agent: AgentType): string {
    const labels: Record<AgentType, string> = {
      copilot: 'Copilot',
      quantcode: 'QuantCode',
      analyst: 'Analyst'
    };
    return labels[agent] || agent;
  }

  function getAgentColor(agent: AgentType): string {
    const colors: Record<AgentType, string> = {
      copilot: '#00ff00',
      quantcode: '#00ffff',
      analyst: '#ff9900'
    };
    return colors[agent] || '#888';
  }

  function getActionIcon(action: string): string {
    const icons: Record<string, string> = {
      created: '✨',
      modified: '✏️',
      deleted: '🗑️'
    };
    return icons[action] || '📄';
  }

  function selectVersion(version: FileVersion) {
    selectedVersion = version;
    dispatch('select-version', { version });
  }

  function viewDiff(version: FileVersion) {
    compareVersion = version;
    showDiff = true;
  }

  function revertToVersion(version: FileVersion) {
    dispatch('revert', { versionContent: version.content, version });
  }

  function toggleVersionExpand(versionId: string) {
    if (expandedVersions.has(versionId)) {
      expandedVersions.delete(versionId);
    } else {
      expandedVersions.add(versionId);
    }
    expandedVersions = expandedVersions;
  }

  function getLineCount(content: string): number {
    return content.split('\n').length;
  }

  function truncateContent(content: string, maxLength: number = 200): string {
    if (content.length <= maxLength) return content;
    return content.substring(0, maxLength) + '...';
  }

  function clearHistory() {
    if (confirm('Are you sure you want to clear all history for this file?')) {
      fileHistoryManager.deleteFileHistory(fileId);
      history = null;
      dispatch('history-cleared');
    }
  }
</script>

<div class="history-panel">
  <div class="panel-header">
    <div class="header-title">
      <History size={18} />
      <span>File History</span>
    </div>
    <button class="close-btn" on:click={() => dispatch('close')}>
      <X size={16} />
    </button>
  </div>

  <div class="panel-info">
    <FileText size={14} />
    <span class="file-name">{fileName}</span>
  </div>

  {#if history && history.versions.length > 0}
    <div class="history-stats">
      <span class="stat">{history.versions.length} versions</span>
      <span class="stat-divider">•</span>
      <span class="stat">Since {formatTimestamp(history.createdAt)}</span>
    </div>

    <div class="versions-list">
      {#each history.versions as version, i}
        <div 
          class="version-item" 
          class:selected={selectedVersion?.id === version.id}
          class:latest={i === 0}
        >
          <div class="version-header" on:click={() => toggleVersionExpand(version.id)}>
            <div class="version-meta">
              <span class="action-icon">{getActionIcon(version.action)}</span>
              <span class="action-label">{version.action}</span>
              {#if i === 0}
                <span class="latest-badge">Latest</span>
              {/if}
            </div>
            <div class="version-info">
              <span class="timestamp">
                <Clock size={12} />
                {formatTimestamp(version.timestamp)}
              </span>
              <span class="agent" style="color: {getAgentColor(version.agent)}">
                <User size={12} />
                {getAgentLabel(version.agent)}
              </span>
            </div>
            <button class="expand-btn">
              {#if expandedVersions.has(version.id)}
                <ChevronUp size={14} />
              {:else}
                <ChevronDown size={14} />
              {/if}
            </button>
          </div>

          {#if expandedVersions.has(version.id)}
            <div class="version-details">
              <div class="summary">{version.summary}</div>
              
              {#if version.diff}
                <div class="diff-preview">
                  <div class="diff-header">Changes:</div>
                  <pre class="diff-content">{version.diff}</pre>
                </div>
              {/if}

              <div class="content-preview">
                <div class="preview-header">
                  <span>Content ({getLineCount(version.content)} lines)</span>
                </div>
                <pre class="preview-content">{truncateContent(version.content)}</pre>
              </div>

              <div class="version-actions">
                <button 
                  class="action-btn primary" 
                  on:click={() => revertToVersion(version)}
                  title="Revert to this version"
                >
                  <RotateCcw size={14} />
                  Revert
                </button>
                <button 
                  class="action-btn" 
                  on:click={() => viewDiff(version)}
                  title="View diff with current"
                >
                  <Eye size={14} />
                  Compare
                </button>
              </div>
            </div>
          {/if}
        </div>
      {/each}
    </div>

    <div class="panel-footer">
      <button class="clear-btn" on:click={clearHistory}>
        <Trash2 size={14} />
        Clear History
      </button>
    </div>
  {:else}
    <div class="no-history">
      <History size={32} />
      <p>No history available</p>
      <span>File versions will appear here as changes are made</span>
    </div>
  {/if}
</div>

<style>
  .history-panel {
    display: flex;
    flex-direction: column;
    width: 350px;
    max-width: 100%;
    height: 100%;
    background: var(--bg-primary, #1a1b26);
    border-left: 1px solid var(--border-subtle, #2d2d3a);
    overflow: hidden;
  }

  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    background: var(--bg-secondary, #16161e);
    border-bottom: 1px solid var(--border-subtle, #2d2d3a);
  }

  .header-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
    color: var(--text-primary, #c0caf5);
  }

  .close-btn {
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

  .close-btn:hover {
    background: var(--bg-tertiary, #292a3e);
    color: var(--text-primary, #c0caf5);
  }

  .panel-info {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    background: var(--bg-tertiary, #1f1f2e);
    border-bottom: 1px solid var(--border-subtle, #2d2d3a);
    color: var(--text-secondary, #a9b1d6);
  }

  .file-name {
    font-weight: 500;
    color: var(--text-primary, #c0caf5);
  }

  .history-stats {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    font-size: 12px;
    color: var(--text-muted, #565f89);
    border-bottom: 1px solid var(--border-subtle, #2d2d3a);
  }

  .stat-divider {
    opacity: 0.5;
  }

  .versions-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
  }

  .version-item {
    background: var(--bg-secondary, #16161e);
    border: 1px solid var(--border-subtle, #2d2d3a);
    border-radius: 6px;
    margin-bottom: 8px;
    overflow: hidden;
    transition: all 0.15s ease;
  }

  .version-item:hover {
    border-color: var(--accent-primary, #00ff00);
  }

  .version-item.selected {
    border-color: var(--accent-primary, #00ff00);
    box-shadow: 0 0 8px rgba(0, 255, 0, 0.2);
  }

  .version-item.latest {
    border-color: var(--accent-primary, #00ff00);
  }

  .version-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 12px;
    cursor: pointer;
    transition: background 0.15s ease;
  }

  .version-header:hover {
    background: var(--bg-tertiary, #1f1f2e);
  }

  .version-meta {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .action-icon {
    font-size: 14px;
  }

  .action-label {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary, #c0caf5);
    text-transform: capitalize;
  }

  .latest-badge {
    padding: 2px 6px;
    background: var(--accent-primary, #00ff00);
    color: #000;
    font-size: 10px;
    font-weight: 600;
    border-radius: 3px;
    text-transform: uppercase;
  }

  .version-info {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 11px;
    color: var(--text-muted, #565f89);
  }

  .timestamp, .agent {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .expand-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: transparent;
    border: none;
    color: var(--text-secondary, #a9b1d6);
    cursor: pointer;
    border-radius: 4px;
    transition: all 0.15s ease;
  }

  .expand-btn:hover {
    background: var(--bg-tertiary, #292a3e);
  }

  .version-details {
    padding: 12px;
    border-top: 1px solid var(--border-subtle, #2d2d3a);
    background: var(--bg-tertiary, #1f1f2e);
  }

  .summary {
    font-size: 12px;
    color: var(--text-secondary, #a9b1d6);
    margin-bottom: 10px;
  }

  .diff-preview, .content-preview {
    margin-bottom: 10px;
  }

  .diff-header, .preview-header {
    font-size: 11px;
    font-weight: 500;
    color: var(--text-muted, #565f89);
    margin-bottom: 6px;
    text-transform: uppercase;
  }

  .diff-content, .preview-content {
    padding: 8px;
    background: var(--bg-primary, #1a1b26);
    border: 1px solid var(--border-subtle, #2d2d3a);
    border-radius: 4px;
    font-family: 'Fira Code', 'Monaco', monospace;
    font-size: 11px;
    color: var(--text-secondary, #a9b1d6);
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-all;
    max-height: 100px;
    margin: 0;
  }

  .version-actions {
    display: flex;
    gap: 8px;
    margin-top: 12px;
  }

  .action-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: var(--bg-secondary, #16161e);
    border: 1px solid var(--border-subtle, #2d2d3a);
    border-radius: 4px;
    font-size: 12px;
    color: var(--text-secondary, #a9b1d6);
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .action-btn:hover {
    background: var(--bg-tertiary, #292a3e);
    color: var(--text-primary, #c0caf5);
  }

  .action-btn.primary {
    background: var(--accent-primary, #00ff00);
    color: #000;
    border-color: var(--accent-primary, #00ff00);
  }

  .action-btn.primary:hover {
    background: #00cc00;
  }

  .panel-footer {
    padding: 12px 16px;
    border-top: 1px solid var(--border-subtle, #2d2d3a);
    background: var(--bg-secondary, #16161e);
  }

  .clear-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    background: transparent;
    border: 1px solid var(--border-subtle, #2d2d3a);
    border-radius: 4px;
    font-size: 12px;
    color: var(--text-muted, #565f89);
    cursor: pointer;
    transition: all 0.15s ease;
    width: 100%;
    justify-content: center;
  }

  .clear-btn:hover {
    background: rgba(255, 0, 0, 0.1);
    border-color: #ff4444;
    color: #ff4444;
  }

  .no-history {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px 20px;
    color: var(--text-muted, #565f89);
    text-align: center;
  }

  .no-history p {
    margin: 12px 0 4px;
    font-size: 14px;
    color: var(--text-secondary, #a9b1d6);
  }

  .no-history span {
    font-size: 12px;
  }
</style>

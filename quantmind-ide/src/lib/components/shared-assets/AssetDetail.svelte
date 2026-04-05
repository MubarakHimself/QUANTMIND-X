<script lang="ts">
  /**
   * Asset Detail Component
   *
   * Displays full asset metadata and code content with Monaco editor
   */
  import { FileText, Layout, Code, Sparkles, Workflow, Settings, FolderTree, Edit3, Save, ArrowLeft } from 'lucide-svelte';
  import GlassTile from '$lib/components/live-trading/GlassTile.svelte';
  import MonacoEditor from '$lib/components/MonacoEditor.svelte';
  import { sharedAssetsStore, selectedAsset } from '$lib/stores/sharedAssets';
  import { API_CONFIG } from '$lib/config/api';
  import type { SharedAsset, AssetType } from '$lib/api/sharedAssetsApi';

  interface Props {
    onBack?: () => void;
  }

  let { onBack }: Props = $props();

  // Edit mode state
  let isEditMode = $state(false);
  let editedContent = $state('');
  let hasChanges = $state(false);

  // Get selected asset from store
  let asset = $derived($selectedAsset);

  // Icon mapping
  const iconMap: Record<AssetType, any> = {
    'docs': FileText,
    'strategy-templates': Layout,
    'indicators': Code,
    'skills': Sparkles,
    'flow-components': Workflow,
    'mcp-configs': Settings,
    'strategies': FolderTree
  };

  // Type label mapping
  const typeLabels: Record<AssetType, string> = {
    'docs': 'Document',
    'strategy-templates': 'Strategy Template',
    'indicators': 'Indicator',
    'skills': 'Skill',
    'flow-components': 'Flow Component',
    'mcp-configs': 'MCP Config',
    'strategies': 'Strategy'
  };

  // Only a subset of asset types should be editable as code/text.
  let isEditableContent = $derived(
    asset?.type === 'indicators' ||
    asset?.type === 'flow-components' ||
    asset?.type === 'mcp-configs'
  );

  let hasRenderableContent = $derived(Boolean(asset?.content));
  let strategyPayload = $derived.by(() => {
    if (asset?.type !== 'strategies' || !asset.content) return null;
    try {
      const parsed = JSON.parse(asset.content);
      return parsed?.type === 'strategy_tree' ? parsed : null;
    } catch {
      return null;
    }
  });
  let strategyDetail = $derived((strategyPayload?.detail ?? asset?.details) as Record<string, any> | null);
  const stageSummaries = $derived.by(() => {
    if (!strategyDetail) return [];
    return [
      {
        label: 'Research',
        count: strategyDetail.research_files?.length || 0,
        detail: strategyDetail.has_trd ? 'TRD/SDD artifacts ready' : 'Pending handoff'
      },
      {
        label: 'Development',
        count: strategyDetail.development_files?.length || 0,
        detail: strategyDetail.has_ea ? 'EA source present' : 'Pending code generation'
      },
      {
        label: 'Variants',
        count: strategyDetail.variant_files?.length || 0,
        detail: strategyDetail.has_variants ? 'Variant roots available' : 'No variants yet'
      },
      {
        label: 'Compilation',
        count: strategyDetail.compilation_files?.length || 0,
        detail: strategyDetail.has_compilation ? 'Build artifacts present' : 'No compiled builds yet'
      },
      {
        label: 'Reports',
        count: (strategyDetail.report_files?.length || 0) + (strategyDetail.backtest_files?.length || 0),
        detail: strategyDetail.has_reports || strategyDetail.has_backtest ? 'Reports/backtests recorded' : 'No reports yet'
      }
    ];
  });

  // Format date
  function formatDate(dateStr: string): string {
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return dateStr;
    }
  }

  // Handle back click
  function handleBack() {
    isEditMode = false;
    hasChanges = false;
    sharedAssetsStore.clearSelection();
    if (onBack) {
      onBack();
    }
  }

  // Toggle edit mode
  function toggleEditMode() {
    if (!isEditMode && asset) {
      editedContent = asset.content || '';
    }
    isEditMode = !isEditMode;
  }

  // Handle content change in editor
  function handleContentChange(event: CustomEvent<string>) {
    editedContent = event.detail;
    hasChanges = asset?.content !== editedContent;
  }

  // Handle save
  async function handleSave() {
    if (!asset || !hasChanges) return;

    try {
      // Try to save to backend first
      const response = await fetch(`${API_CONFIG?.API_BASE || ''}/api/assets/${asset.id}/content`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: editedContent })
      });

      if (!response.ok) {
        throw new Error('Failed to save to backend');
      }

      // Update local store with new content
      sharedAssetsStore.updateAssetContent(asset.id, editedContent);
      hasChanges = false;
      isEditMode = false;
    } catch (e) {
      console.error('Failed to save asset content:', e);
      hasChanges = false;
      isEditMode = false;
    }
  }

  // Handle cancel
  function handleCancel() {
    editedContent = asset?.content || '';
    hasChanges = false;
    isEditMode = false;
  }
</script>

{#if asset}
  <div class="asset-detail">
    <!-- Header with back button -->
    <div class="detail-header">
      <button class="back-button" onclick={handleBack}>
        <ArrowLeft size={16} />
        Back to List
      </button>
    </div>

    <!-- Breadcrumb navigation -->
    <nav class="breadcrumb-nav" aria-label="Breadcrumb navigation">
      <span class="breadcrumb-item">Shared Assets</span>
      <span class="separator">/</span>
      <span class="breadcrumb-item">{typeLabels[asset.type]}</span>
      <span class="separator">/</span>
      <span class="breadcrumb-item current">{asset.name}</span>
    </nav>

    <!-- Asset metadata -->
    <GlassTile>
      <div class="asset-metadata">
        <div class="asset-icon-large">
          {#if iconMap[asset.type]}
            <svelte:component this={iconMap[asset.type]} size={48} />
          {:else}
            <FileText size={48} />
          {/if}
        </div>

        <div class="metadata-content">
          <h2 class="asset-title">{asset.name}</h2>

          <div class="metadata-grid">
            <div class="metadata-item">
              <span class="metadata-label">Type</span>
              <span class="metadata-value">{typeLabels[asset.type]}</span>
            </div>
            <div class="metadata-item">
              <span class="metadata-label">Version</span>
              <span class="metadata-value">v{asset.metadata.version}</span>
            </div>
            <div class="metadata-item">
              <span class="metadata-label">Usage Count</span>
              <span class="metadata-value">{asset.metadata.usage_count} workflows</span>
            </div>
            <div class="metadata-item">
              <span class="metadata-label">Last Updated</span>
              <span class="metadata-value">{formatDate(asset.metadata.last_updated)}</span>
            </div>
            {#if strategyDetail}
              <div class="metadata-item">
                <span class="metadata-label">Status</span>
                <span class="metadata-value">{strategyDetail.status || 'pending'}</span>
              </div>
              <div class="metadata-item">
                <span class="metadata-label">Video Ingest</span>
                <span class="metadata-value">{strategyDetail.has_video_ingest ? 'Present' : 'Pending'}</span>
              </div>
              <div class="metadata-item">
                <span class="metadata-label">Source Captions</span>
                <span class="metadata-value">{strategyDetail.has_source_captions ? 'Available' : 'None'}</span>
              </div>
              <div class="metadata-item">
                <span class="metadata-label">Source Audio</span>
                <span class="metadata-value">{strategyDetail.has_source_audio ? 'Available' : 'None'}</span>
              </div>
              <div class="metadata-item">
                <span class="metadata-label">Chunk Manifest</span>
                <span class="metadata-value">{strategyDetail.has_chunk_manifest ? 'Available' : 'None'}</span>
              </div>
            {/if}
          </div>

          {#if asset.metadata.description}
            <p class="asset-description">{asset.metadata.description}</p>
          {/if}

          {#if asset.metadata.author}
            <p class="asset-author">Author: {asset.metadata.author}</p>
          {/if}

          {#if strategyDetail?.blocking_error}
            <div class="strategy-error-banner" title={strategyDetail.blocking_error_detail || strategyDetail.blocking_error}>
              {strategyDetail.blocking_error}
            </div>
          {/if}
        </div>
      </div>
    </GlassTile>

    {#if strategyDetail}
      <GlassTile>
        <div class="strategy-artifact-grid">
          <div class="artifact-group">
            <h3>Source</h3>
            <p>{strategyPayload?.root || asset.name}</p>
          </div>
          <div class="artifact-group">
            <h3>Timelines</h3>
            <p>{strategyDetail.source_artifacts?.timeline_files?.length || 0} root files</p>
            <p>{strategyDetail.source_artifacts?.chunk_timeline_files?.length || 0} chunk files</p>
          </div>
          <div class="artifact-group">
            <h3>Captions</h3>
            <p>{strategyDetail.source_artifacts?.caption_files?.length || 0} files</p>
          </div>
          <div class="artifact-group">
            <h3>Audio</h3>
            <p>{strategyDetail.source_artifacts?.audio_files?.length || 0} files</p>
          </div>
          <div class="artifact-group">
            <h3>Chunk Plans</h3>
            <p>{strategyDetail.source_artifacts?.chunk_manifest_files?.length || 0} manifests</p>
          </div>
        </div>
      </GlassTile>

      <GlassTile>
        <div class="strategy-artifact-grid strategy-stage-grid">
          {#each stageSummaries as stage}
            <div class="artifact-group">
              <h3>{stage.label}</h3>
              <p>{stage.count} files</p>
              <p>{stage.detail}</p>
            </div>
          {/each}
        </div>
      </GlassTile>
    {/if}

    <!-- Content section -->
    {#if hasRenderableContent}
      <div class="code-section">
        <div class="code-header">
          <h3 class="code-title">{isEditableContent ? 'Code' : 'Content'}</h3>

          {#if isEditableContent && !isEditMode}
            <button class="action-button" onclick={toggleEditMode}>
              <Edit3 size={14} />
              Edit
            </button>
          {:else if isEditableContent}
            <div class="edit-actions">
              <button
                class="action-button cancel"
                onclick={handleCancel}
                disabled={!hasChanges}
              >
                Cancel
              </button>
              <button
                class="action-button save"
                onclick={handleSave}
                disabled={!hasChanges}
              >
                <Save size={14} />
                Save
              </button>
            </div>
          {/if}
        </div>

        <div class="code-editor-container">
          <MonacoEditor
            content={isEditMode ? editedContent : asset.content}
            language={asset.language || (asset.type === 'strategies' ? 'json' : 'plaintext')}
            filename={asset.name}
            readOnly={!isEditMode || !isEditableContent}
            showLineNumbers={true}
          />
        </div>
      </div>
    {/if}

    <!-- Diff view (when in edit mode with changes) -->
    {#if isEditMode && hasChanges}
      <div class="diff-info">
        <p>You have unsaved changes. Click Save to persist or Cancel to discard.</p>
      </div>
    {/if}
  </div>
{:else}
  <div class="no-asset">
    <p>No asset selected</p>
    <button class="back-button" onclick={handleBack}>Go Back</button>
  </div>
{/if}

<style>
  .asset-detail {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 16px;
    height: 100%;
    overflow-y: auto;
  }

  .detail-header {
    display: flex;
    align-items: center;
  }

  .back-button {
    display: flex;
    align-items: center;
    gap: 8px;
    background: transparent;
    border: none;
    color: rgba(0, 212, 255, 0.8);
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    cursor: pointer;
    padding: 8px 12px;
    border-radius: 4px;
    transition: all 0.15s ease;
  }

  .back-button:hover {
    background: rgba(0, 212, 255, 0.1);
    color: rgba(0, 212, 255, 1);
  }

  /* Breadcrumb navigation */
  .breadcrumb-nav {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    font-size: 13px;
  }

  .breadcrumb-item {
    color: rgba(255, 255, 255, 0.5);
  }

  .breadcrumb-item.current {
    color: #e0e0e0;
    font-weight: 500;
  }

  .separator {
    color: rgba(255, 255, 255, 0.3);
  }

  /* Asset metadata */
  .asset-metadata {
    display: flex;
    gap: 24px;
  }

  .asset-icon-large {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 80px;
    height: 80px;
    background: rgba(0, 212, 255, 0.1);
    border-radius: 12px;
    color: rgba(0, 212, 255, 0.8);
    flex-shrink: 0;
  }

  .metadata-content {
    flex: 1;
    min-width: 0;
  }

  .asset-title {
    font-size: 20px;
    font-weight: 500;
    color: #e0e0e0;
    margin: 0 0 16px 0;
  }

  .metadata-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 16px;
    margin-bottom: 16px;
  }

  .metadata-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .metadata-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: rgba(255, 255, 255, 0.4);
  }

  .metadata-value {
    font-size: 14px;
    color: #e0e0e0;
  }

  .asset-description {
    font-size: 13px;
    color: rgba(255, 255, 255, 0.7);
    margin: 0 0 8px 0;
    line-height: 1.5;
  }

  .asset-author {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.5);
    margin: 0;
  }

  .strategy-error-banner {
    margin-top: 12px;
    padding: 12px 14px;
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.28);
    border-radius: 8px;
    font-size: 12px;
    line-height: 1.45;
    color: #fca5a5;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 180px;
    overflow: auto;
  }

  .strategy-artifact-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 16px;
  }

  .artifact-group h3 {
    margin: 0 0 6px 0;
    font-size: 12px;
    color: rgba(0, 212, 255, 0.9);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .artifact-group p {
    margin: 0;
    font-size: 12px;
    color: rgba(255, 255, 255, 0.72);
    line-height: 1.5;
  }

  /* Code section */
  .code-section {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .code-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .code-title {
    font-size: 14px;
    font-weight: 500;
    color: #e0e0e0;
    margin: 0;
  }

  .action-button {
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.2);
    color: rgba(0, 212, 255, 0.9);
    padding: 6px 12px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .action-button:hover:not(:disabled) {
    background: rgba(0, 212, 255, 0.2);
  }

  .action-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .action-button.save {
    background: rgba(0, 212, 255, 0.2);
    border-color: rgba(0, 212, 255, 0.4);
  }

  .action-button.cancel {
    background: transparent;
    border-color: rgba(255, 255, 255, 0.2);
    color: rgba(255, 255, 255, 0.7);
  }

  .edit-actions {
    display: flex;
    gap: 8px;
  }

  .code-editor-container {
    height: 400px;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid rgba(0, 212, 255, 0.1);
  }

  .diff-info {
    padding: 12px 16px;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 4px;
    font-size: 13px;
    color: rgba(0, 212, 255, 0.9);
  }

  /* No asset state */
  .no-asset {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 48px;
    color: rgba(255, 255, 255, 0.5);
    gap: 16px;
  }

  /* Frosted Terminal glass styling */
  :global(.asset-detail .glass-tile) {
    background: rgba(8, 13, 20, 0.35);
    backdrop-filter: blur(16px) saturate(120%);
    -webkit-backdrop-filter: blur(16px) saturate(120%);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 8px;
    padding: 20px;
  }
</style>

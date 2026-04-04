<script lang="ts">
  /**
   * Research Detail Page Component
   * Displays full document details with provenance and send to copilot action
   */
  import GlassTile from '$lib/components/live-trading/GlassTile.svelte';
  import { SOURCE_BADGE_COLORS, type KnowledgeSearchResult } from '$lib/api/knowledgeApi';
  import { API_CONFIG } from '$lib/config/api';
  import { ChevronRight, Home, ExternalLink, Database, Send } from 'lucide-svelte';

  export let result: KnowledgeSearchResult;
  export let onGoBack: () => void = () => {};
  export let onShowToast: () => void = () => {};

  function getSourceBadgeColor(sourceType: string): string {
    return SOURCE_BADGE_COLORS[sourceType] || '#888888';
  }

  async function sendToCopilot() {
    const contextMessage = `[Knowledge Context]
Title: ${result.title}
Source: ${result.source_type}

Excerpt:
${result.excerpt}`;

    try {
      const response = await fetch(`${API_CONFIG.API_BASE}/copilot/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: contextMessage,
          canvas_context: 'research',
          session_id: null
        })
      });

      if (response.ok) {
        onShowToast();
      }
    } catch (e) {
      console.error('Failed to send to Copilot:', e);
    }
  }
</script>

<div class="detail-view">
  <!-- Breadcrumb Navigation (inline) -->
  <nav class="breadcrumb">
    <button class="breadcrumb-item" on:click={onGoBack}>
      <Home size={14} />
      <span>Research</span>
    </button>
    <ChevronRight size={14} />
    <span class="breadcrumb-current">{result.title}</span>
  </nav>

  <!-- Document Detail -->
  <GlassTile clickable={false}>
    <div class="detail-content">
      <!-- Source Badge -->
      <div class="detail-badges">
        <span
          class="source-badge"
          style="background-color: {getSourceBadgeColor(result.source_type)}"
        >
          {result.source_type}
        </span>
        <span class="relevance-badge">
          Relevance: {result.relevance_score.toFixed(2)}
        </span>
      </div>

      <!-- Title -->
      <h2 class="detail-title">{result.title}</h2>

      <!-- Full Excerpt -->
      <div class="detail-excerpt">
        <p>{result.excerpt}</p>
      </div>

      <!-- Provenance -->
      <div class="provenance">
        {#if result.provenance.source_url}
          <div class="provenance-item">
            <ExternalLink size={12} />
            <span>Source: {result.provenance.source_url}</span>
          </div>
        {/if}
        {#if result.provenance.indexed_at_utc}
          <div class="provenance-item">
            <Database size={12} />
            <span>Indexed: {result.provenance.indexed_at_utc}</span>
          </div>
        {/if}
      </div>

      <!-- Actions -->
      <div class="detail-actions">
        <button class="send-to-copilot-btn" on:click={sendToCopilot}>
          <Send size={16} />
          <span>Send to Copilot</span>
        </button>
      </div>
    </div>
  </GlassTile>
</div>

<style>
  .detail-view {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  .breadcrumb {
    display: flex;
    align-items: center;
    gap: 8px;
    color: rgba(224, 224, 224, 0.6);
    font-size: 13px;
  }

  .breadcrumb-item {
    display: flex;
    align-items: center;
    gap: 4px;
    background: transparent;
    border: none;
    color: #00d4ff;
    cursor: pointer;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    padding: 4px 8px;
    border-radius: 4px;
    transition: background 0.2s;
  }

  .breadcrumb-item:hover {
    background: rgba(0, 212, 255, 0.1);
  }

  .breadcrumb-current {
    color: #e0e0e0;
    font-weight: 500;
  }

  .detail-content {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .detail-badges {
    display: flex;
    gap: 8px;
  }

  .source-badge {
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 10px;
    text-transform: uppercase;
    color: #0a0f1a;
    font-weight: 600;
  }

  .relevance-badge {
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    background: rgba(0, 200, 100, 0.15);
    border: 1px solid rgba(0, 200, 100, 0.3);
    color: #00c864;
  }

  .detail-title {
    font-size: 20px;
    font-weight: 600;
    margin: 0;
    color: #e0e0e0;
  }

  .detail-excerpt {
    font-size: 14px;
    line-height: 1.6;
    color: rgba(224, 224, 224, 0.85);
  }

  .provenance {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding-top: 12px;
    border-top: 1px solid rgba(0, 212, 255, 0.1);
  }

  .provenance-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: rgba(224, 224, 224, 0.5);
  }

  .detail-actions {
    padding-top: 12px;
  }

  .send-to-copilot-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(0, 212, 255, 0.15);
    border: 1px solid rgba(0, 212, 255, 0.3);
    color: #00d4ff;
    padding: 10px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    transition: all 0.2s;
  }

  .send-to-copilot-btn:hover {
    background: rgba(0, 212, 255, 0.25);
  }
</style>

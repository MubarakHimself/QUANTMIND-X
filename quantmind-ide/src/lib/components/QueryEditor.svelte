<script lang="ts">
  import {
    Terminal,
    Play,
    RefreshCw,
    Code,
    FileText,
    AlertCircle,
  } from "lucide-svelte";


  interface Props {
    queryInput?: string;
    queryHistory?: string[];
    queryResults?: any;
    isQueryRunning?: boolean;
    executeQuery: () => Promise<void>;
    navigateHistory: (direction: "up" | "down") => void;
    isJsonColumn: (value: any) => boolean;
    previewJson: (data: any) => void;
  }

  let {
    queryInput = $bindable(""),
    queryHistory = [],
    queryResults = null,
    isQueryRunning = false,
    executeQuery,
    navigateHistory,
    isJsonColumn,
    previewJson
  }: Props = $props();

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      executeQuery();
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      navigateHistory("up");
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      navigateHistory("down");
    }
  }
</script>

<div class="query-panel">
  <div class="query-header">
    <Terminal size={14} />
    <h3>SQL Query Editor</h3>
  </div>

  <div class="query-editor">
    <textarea
      placeholder="Enter SQL query...
Examples:
  SELECT * FROM prop_firm_accounts LIMIT 10
  SELECT COUNT(*) FROM trade_proposals WHERE status = 'active'
  SELECT strategy_name, AVG(profit) FROM crypto_trades GROUP BY strategy_name"
      bind:value={queryInput}
      onkeydown={handleKeyDown}
    ></textarea>
  </div>

  <div class="query-actions">
    <div class="query-hint">
      <span>Ctrl+Enter to run</span>
      <span>Use Up/Down for history</span>
    </div>
    <button
      class="btn primary"
      onclick={executeQuery}
      disabled={isQueryRunning || !queryInput.trim()}
    >
      {#if isQueryRunning}
        <RefreshCw size={14} class="spin" />
        <span>Running...</span>
      {:else}
        <Play size={14} />
        <span>Run Query</span>
      {/if}
    </button>
  </div>

  <!-- Query History -->
  {#if queryHistory.length > 0}
    <div class="query-history">
      <h4>Recent Queries</h4>
      <div class="history-list">
        {#each queryHistory.slice(0, 5) as query}
          <div
            class="history-item"
            onclick={() => (queryInput = query)}
            onkeydown={(e) => e.key === "Enter" && (queryInput = query)}
            role="button"
            tabindex="0"
            aria-label="Use query: {query.slice(0, 30)}..."
          >
            <Code size={10} />
            <span class="query-text">{query.slice(0, 60)}...</span>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Query Results -->
  {#if queryResults}
    <div class="query-results">
      <div class="results-header">
        <h4>Results</h4>
        {#if queryResults.error}
          <div class="error-message">
            <AlertCircle size={12} />
            <span>{queryResults.error}</span>
          </div>
        {:else}
          <span class="results-stats">
            {queryResults.row_count} rows in {queryResults.execution_time_ms.toFixed(
              2,
            )}ms
          </span>
        {/if}
      </div>

      {#if queryResults.rows.length > 0}
        <div class="results-grid">
          <div class="results-header-row">
            {#each queryResults.columns as column}
              <div class="results-cell header">{column}</div>
            {/each}
          </div>
          {#each queryResults.rows.slice(0, 50) as row}
            <div class="results-row">
              {#each queryResults.columns as column}
                <div class="results-cell">
                  {#if isJsonColumn(row[column])}
                    <span
                      class="json-link"
                      onclick={() => previewJson(row[column])}
                      onkeydown={(e) =>
                        e.key === "Enter" && previewJson(row[column])}
                      role="button"
                      tabindex="0"
                      aria-label="View JSON data"
                    >
                      <Code size={8} />
                      <span>JSON</span>
                    </span>
                  {:else}
                    <span
                      >{row[column] !== null
                        ? String(row[column])
                        : "<NULL>"}</span
                    >
                  {/if}
                </div>
              {/each}
            </div>
          {/each}
        </div>
      {:else if !queryResults.error}
        <div class="empty-results">
          <FileText size={24} />
          <p>Query executed successfully but returned no rows</p>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .query-panel {
    width: 400px;
    min-width: 300px;
    display: flex;
    flex-direction: column;
    border-left: 1px solid var(--border-color);
    background: var(--surface-2);
  }

  .query-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border-color);
  }

  .query-header h3 {
    margin: 0;
    font-size: 0.9rem;
    font-weight: 600;
  }

  .query-editor {
    padding: 0.75rem;
  }

  .query-editor textarea {
    width: 100%;
    min-height: 150px;
    padding: 0.75rem;
    background: var(--surface-1);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    font-family: monospace;
    font-size: 0.8rem;
    resize: vertical;
  }

  .query-editor textarea:focus {
    outline: none;
    border-color: var(--primary-color);
  }

  .query-actions {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--border-color);
  }

  .query-hint {
    display: flex;
    gap: 1rem;
    font-size: 0.7rem;
    color: var(--color-text-muted);
  }

  .btn {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.4rem 0.75rem;
    font-size: 0.75rem;
    background: var(--surface-3);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .btn.primary {
    background: var(--primary-color);
    color: white;
    border-color: var(--primary-color);
  }

  .btn:hover:not(:disabled) {
    background: var(--surface-4);
  }

  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .query-history {
    padding: 0.75rem;
    border-bottom: 1px solid var(--border-color);
  }

  .query-history h4 {
    margin: 0 0 0.5rem 0;
    font-size: 0.8rem;
    color: var(--color-text-muted);
  }

  .history-list {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .history-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.5rem;
    background: var(--surface-3);
    border-radius: 4px;
    font-size: 0.75rem;
    cursor: pointer;
    overflow: hidden;
  }

  .history-item:hover {
    background: var(--surface-4);
  }

  .query-text {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .query-results {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .results-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem;
    border-bottom: 1px solid var(--border-color);
  }

  .results-header h4 {
    margin: 0;
    font-size: 0.85rem;
  }

  .results-stats {
    font-size: 0.75rem;
    color: var(--color-text-muted);
  }

  .error-message {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    color: var(--error-color);
    font-size: 0.75rem;
  }

  .results-grid {
    flex: 1;
    overflow: auto;
  }

  .results-header-row {
    display: flex;
    background: var(--surface-3);
    position: sticky;
    top: 0;
  }

  .results-row {
    display: flex;
    border-bottom: 1px solid var(--border-color);
  }

  .results-cell {
    flex: 1;
    min-width: 100px;
    max-width: 150px;
    padding: 0.4rem 0.5rem;
    font-size: 0.75rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    border-right: 1px solid var(--border-color);
  }

  .results-cell.header {
    font-weight: 600;
    background: var(--surface-3);
  }

  .json-link {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    color: var(--primary-color);
    cursor: pointer;
    font-size: 0.7rem;
  }

  .json-link:hover {
    text-decoration: underline;
  }

  .empty-results {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2rem;
    gap: 0.5rem;
    color: var(--color-text-muted);
    font-size: 0.85rem;
  }

  :global(.spin) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }
</style>

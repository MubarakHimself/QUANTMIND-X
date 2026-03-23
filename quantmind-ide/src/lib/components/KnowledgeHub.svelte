<script lang="ts">
  import { run } from 'svelte/legacy';

  import { onMount } from "svelte";
  import { fade, slide, fly } from "svelte/transition";
  import { navigationStore } from "../stores/navigationStore";

  // Types
  interface PDFDocument {
    id: string;
    filename: string;
    namespace: string;
    size_bytes: number;
    pages: number;
    indexed_at: string;
    status: string;
  }

  interface KnowledgeArticle {
    id: string;
    name: string;
    category: string;
    size_bytes: number;
    indexed: boolean;
  }

  interface IndexingStatus {
    job_id: string;
    status: "pending" | "processing" | "completed" | "failed";
    progress: number;
    pages_processed: number;
    pages_total: number;
    started_at: string | null;
    completed_at: string | null;
    error: string | null;
  }

  interface Namespace {
    name: string;
    document_count: number;
  }

  // State
  let documents: PDFDocument[] = $state([]);
  let articles: KnowledgeArticle[] = $state([]);
  let namespaces: Namespace[] = $state([]);
  let loading = $state(true);
  let uploading = $state(false);
  let error: string | null = $state(null);
  let articlesLoading = $state(false);

  // Upload state
  let dragOver = $state(false);
  let uploadProgress = $state(0);
  let currentUpload: string | null = $state(null);
  let indexingStatus: IndexingStatus | null = $state(null);

  // Selected namespace
  let selectedNamespace = $state("knowledge");

  // Search state
  let searchQuery = $state("");
  let searchResults: any[] = $state([]);

  // Sync state
  let syncStatus: {
    status: string;
    last_sync: string | null;
    articles_synced: number;
    errors: string[];
    scraper_available: boolean;
    source_available: boolean;
    existing_articles: number;
  } | null = $state(null);
  let syncLoading = $state(false);
  let syncing = $state(false);

  // Firecrawl settings state
  let firecrawlSettings: {
    api_key_set: boolean;
    scraper_type: string;
    scraper_available: boolean;
    firecrawl_available: boolean;
  } | null = null;
  let showApiKeyModal = false;
  let apiKeyInput = "";
  let selectedScraperType = "simple";
  let savingSettings = false;

  // Articles sorting and filtering
  let sortBy = "name";
  let sortOrder = "asc";
  let selectedCategory = "";
  let availableCategories: string[] = [];
  let progress = 0;

  // Update breadcrumbs when namespace changes
  run(() => {
    if (selectedNamespace) {
      navigationStore.navigateToFolder(
        selectedNamespace,
        selectedNamespace.charAt(0).toUpperCase() + selectedNamespace.slice(1),
      );
    }
  });

  // Fetch documents
  async function fetchDocuments() {
    try {
      const response = await fetch("/api/pdf/documents");
      if (!response.ok) throw new Error("Failed to fetch documents");
      const data = await response.json();
      documents = data.documents;
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to fetch documents";
    }
  }

  // Fetch scraped articles from knowledge base
  async function fetchArticles() {
    articlesLoading = true;
    try {
      let url = `/api/knowledge/articles?sort_by=${sortBy}&order=${sortOrder}&limit=100`;
      if (selectedCategory) {
        url += `&category=${selectedCategory}`;
      }
      const response = await fetch(url);
      if (response.ok) {
        const data = await response.json();
        articles = data.articles;
        availableCategories = data.categories || [];
      } else {
        error = "Failed to fetch articles";
      }
    } catch (e: any) {
      error = e.message || "Failed to fetch articles";
      console.error("Failed to fetch articles:", e);
    } finally {
      articlesLoading = false;
    }
  }

  // Fetch articles when sort/filter changes
  $: if (sortBy || sortOrder || selectedCategory) {
    fetchArticles();
  }

  // Fetch namespaces
  async function fetchNamespaces() {
    try {
      const response = await fetch("/api/pdf/namespaces");
      if (!response.ok) throw new Error("Failed to fetch namespaces");
      const data = await response.json();
      namespaces = data.namespaces;
    } catch (e) {
      console.error("Failed to fetch namespaces:", e);
    }
  }

  // Fetch sync status
  async function fetchSyncStatus() {
    syncLoading = true;
    try {
      const response = await fetch("/api/knowledge/sync/status");
      if (response.ok) {
        const data = await response.json();
        syncStatus = data;
        progress = data.progress || 0;
        if (data.categories) {
          availableCategories = Object.keys(data.categories);
        }
      }
    } catch (e) {
      console.error("Failed to fetch sync status:", e);
    } finally {
      syncLoading = false;
    }
  }

  // Fetch Firecrawl settings
  async function fetchFirecrawlSettings() {
    try {
      const response = await fetch("/api/knowledge/firecrawl/settings");
      if (response.ok) {
        firecrawlSettings = await response.json();
        selectedScraperType = firecrawlSettings?.scraper_type || "simple";
      }
    } catch (e) {
      console.error("Failed to fetch Firecrawl settings:", e);
    }
  }

  // Save Firecrawl settings
  async function saveFirecrawlSettings() {
    savingSettings = true;
    try {
      const response = await fetch("/api/knowledge/firecrawl/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          api_key: apiKeyInput,
          scraper_type: selectedScraperType
        }),
      });
      if (response.ok) {
        firecrawlSettings = await response.json();
        showApiKeyModal = false;
        apiKeyInput = "";
      }
    } catch (e) {
      console.error("Failed to save Firecrawl settings:", e);
    } finally {
      savingSettings = false;
    }
  }

  // Trigger manual sync
  async function triggerSync() {
    syncing = true;
    try {
      const response = await fetch("/api/knowledge/sync", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ batch_size: 10, start_index: 0, scraper_type: selectedScraperType }),
      });
      if (response.ok) {
        const data = await response.json();
        // Poll for sync completion
        await pollSyncStatus();
      }
    } catch (e) {
      console.error("Failed to trigger sync:", e);
    } finally {
      syncing = false;
    }
  }

  // Poll sync status until complete
  async function pollSyncStatus() {
    const maxAttempts = 30;
    let attempts = 0;
    while (attempts < maxAttempts) {
      await fetchSyncStatus();
      if (syncStatus?.status === "completed" || syncStatus?.status === "failed" || syncStatus?.status === "idle") {
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, 2000));
      attempts++;
    }
    // Refresh articles after sync
    await fetchArticles();
  }

  // Handle file upload
  async function handleFileUpload(files: FileList) {
    if (files.length === 0) return;

    const file = files[0];
    if (!file.name.endsWith(".pdf")) {
      error = "Only PDF files are allowed";
      return;
    }

    uploading = true;
    uploadProgress = 0;
    currentUpload = file.name;
    error = null;

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(
        `/api/pdf/upload?namespace=${selectedNamespace}&auto_index=true`,
        {
          method: "POST",
          body: formData,
        },
      );

      if (!response.ok) throw new Error("Upload failed");

      const data = await response.json();

      // Start polling for indexing status
      if (data.indexing_job_id) {
        await pollIndexingStatus(data.indexing_job_id);
      }

      // Refresh documents
      await fetchDocuments();
    } catch (e) {
      error = e instanceof Error ? e.message : "Upload failed";
    } finally {
      uploading = false;
      currentUpload = null;
      indexingStatus = null;
    }
  }

  // Poll indexing status
  async function pollIndexingStatus(jobId: string) {
    while (true) {
      try {
        const response = await fetch(`/api/pdf/status/${jobId}`);
        if (!response.ok) break;

        indexingStatus = await response.json();
        uploadProgress = indexingStatus.progress;

        if (
          indexingStatus.status === "completed" ||
          indexingStatus.status === "failed"
        ) {
          break;
        }

        await new Promise((resolve) => setTimeout(resolve, 1000));
      } catch (e) {
        break;
      }
    }
  }

  // Handle drag and drop
  function handleDragOver(e: DragEvent) {
    e.preventDefault();
    dragOver = true;
  }

  function handleDragLeave(e: DragEvent) {
    e.preventDefault();
    dragOver = false;
  }

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    dragOver = false;

    if (e.dataTransfer?.files) {
      handleFileUpload(e.dataTransfer.files);
    }
  }

  // Handle file input change
  function handleFileInput(e: Event) {
    const target = e.target as HTMLInputElement;
    if (target.files) {
      handleFileUpload(target.files);
    }
  }

  // Delete document
  async function deleteDocument(docId: string) {
    if (!confirm("Are you sure you want to delete this document?")) return;

    try {
      const response = await fetch(`/api/pdf/documents/${docId}`, {
        method: "DELETE",
      });

      if (!response.ok) throw new Error("Delete failed");

      await fetchDocuments();
    } catch (e) {
      error = e instanceof Error ? e.message : "Delete failed";
    }
  }

  // Search documents
  async function searchDocuments() {
    if (!searchQuery.trim()) return;

    try {
      const response = await fetch("/api/pdf/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: searchQuery,
          namespaces: [selectedNamespace],
          max_results: 10,
        }),
      });

      if (!response.ok) throw new Error("Search failed");

      const data = await response.json();
      searchResults = data.results;
    } catch (e) {
      error = e instanceof Error ? e.message : "Search failed";
    }
  }

  // Format file size
  function formatSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  // Format date
  function formatDate(isoString: string): string {
    return new Date(isoString).toLocaleDateString();
  }

  // Lifecycle
  onMount(async () => {
    await Promise.all([fetchDocuments(), fetchNamespaces(), fetchArticles(), fetchSyncStatus(), fetchFirecrawlSettings()]);
    loading = false;
  });
</script>

<div class="knowledge-hub">
  <!-- Header -->
  <div class="hub-header">
    <h2>Knowledge Hub</h2>
    <p class="subtitle">Upload and index PDF documents for AI-powered search</p>
  </div>

  <!-- Sync Status Panel -->
  {#if syncStatus}
    <div class="sync-status-panel" in:slide>
      <div class="sync-header">
        <div class="sync-info">
          <span class="sync-title">Knowledge Sync</span>
          <span class="sync-status-badge" class:syncing={syncStatus.status === 'syncing'} class:error={syncStatus.status === 'failed'}>
            {#if syncStatus.status === 'syncing'}
              <span class="spinner-small"></span> Syncing...
            {:else if syncStatus.status === 'failed'}
              Failed
            {:else if syncStatus.status === 'completed'}
              Completed
            {:else}
              Idle
            {/if}
          </span>
        </div>
        <!-- Scraper Type Selector -->
        <div class="scraper-selector">
          <select
            bind:value={selectedScraperType}
            class="scraper-select"
            disabled={syncing}
          >
            <option value="simple">Simple Scraper</option>
            <option value="firecrawl" disabled={!firecrawlSettings?.firecrawl_available}>
              Firecrawl {firecrawlSettings?.firecrawl_available ? '' : '(unavailable)'}
            </option>
          </select>
          {#if selectedScraperType === 'firecrawl'}
            <button
              class="api-key-btn"
              on:click={() => showApiKeyModal = true}
              title="Configure API Key"
            >
              {firecrawlSettings?.api_key_set ? 'API Key Set' : 'Set API Key'}
            </button>
          {/if}
        </div>
        <button
          class="sync-btn"
          onclick={triggerSync}
          disabled={syncing || syncLoading}
          title="Sync knowledge articles"
        >
          {#if syncing}
            <span class="spinner-small"></span> Syncing...
          {:else}
            Sync Now
          {/if}
        </button>
      </div>
      <div class="sync-details">
        <!-- Progress Bar -->
        {#if syncStatus.status === 'running' || syncStatus.status === 'syncing'}
          <div class="sync-progress">
            <div class="progress-bar">
              <div class="progress-fill" style="width: {progress}%"></div>
            </div>
            <span class="progress-text">{progress}% Complete</span>
          </div>
        {/if}

        <div class="sync-stats-row">
          <div class="sync-stat">
            <span class="stat-label">Articles:</span>
            <span class="stat-value">{syncStatus.existing_articles || 0}</span>
          </div>
          <div class="sync-stat">
            <span class="stat-label">Last Sync:</span>
            <span class="stat-value">
              {#if syncStatus.sync_state?.last_sync}
                {new Date(syncStatus.sync_state.last_sync).toLocaleString()}
              {:else}
                Never
              {/if}
            </span>
          </div>
          <div class="sync-stat">
            <span class="stat-label">Last Synced:</span>
            <span class="stat-value">{syncStatus.sync_state?.articles_synced || 0} articles</span>
          </div>
        </div>

        <!-- Categories breakdown -->
        {#if syncStatus.categories && Object.keys(syncStatus.categories).length > 0}
          <div class="categories-breakdown">
            <span class="stat-label">By Category:</span>
            <div class="category-tags">
              {#each Object.entries(syncStatus.categories) as [cat, data]}
                <span class="category-count">{cat}: {data.count}</span>
              {/each}
            </div>
          </div>
        {/if}

        {#if syncStatus.errors && syncStatus.errors.length > 0}
          <div class="sync-errors">
            <span class="error-label">Errors:</span>
            <span class="error-value">{syncStatus.errors.join(', ')}</span>
          </div>
        {/if}
      </div>
    </div>
  {/if}

  <!-- API Key Modal -->
  {#if showApiKeyModal}
    <div class="modal-backdrop" on:click={() => showApiKeyModal = false}>
      <div class="modal-content" on:click|stopPropagation>
        <div class="modal-header">
          <h3>Configure Firecrawl API Key</h3>
          <button class="close-btn" on:click={() => showApiKeyModal = false}>×</button>
        </div>
        <div class="modal-body">
          <p class="modal-description">
            Enter your Firecrawl API key to enable the Firecrawl scraper.
            You can get an API key from <a href="https://firecrawl.dev" target="_blank" rel="noopener">firecrawl.dev</a>
          </p>
          <div class="form-group">
            <label for="apiKey">API Key</label>
            <input
              id="apiKey"
              type="password"
              bind:value={apiKeyInput}
              placeholder="Enter your Firecrawl API key"
              class="form-input"
            />
          </div>
          <div class="form-group">
            <label for="scraperType">Scraper Type</label>
            <select id="scraperType" bind:value={selectedScraperType} class="form-select">
              <option value="simple">Simple Scraper</option>
              <option value="firecrawl">Firecrawl</option>
            </select>
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn-secondary" on:click={() => showApiKeyModal = false}>Cancel</button>
          <button
            class="btn-primary"
            on:click={saveFirecrawlSettings}
            disabled={savingSettings}
          >
            {savingSettings ? 'Saving...' : 'Save Settings'}
          </button>
        </div>
      </div>
    </div>
  {/if}

  <!-- Error Display -->
  {#if error}
    <div class="error-banner" in:fly={{ y: -20 }}>
      <span class="error-icon">⚠️</span>
      <span>{error}</span>
      <button class="dismiss-btn" onclick={() => (error = null)}>×</button>
    </div>
  {/if}

  <!-- Namespace Selector -->
  <div class="namespace-selector">
    <label for="namespace">Namespace:</label>
    <select id="namespace" bind:value={selectedNamespace}>
      {#each namespaces as ns}
        <option value={ns.name}>{ns.name} ({ns.document_count} docs)</option>
      {/each}
      <option value="knowledge">knowledge (default)</option>
      <option value="strategies">strategies</option>
    </select>
  </div>

  <!-- Upload Area -->
  <div
    class="upload-area"
    class:drag-over={dragOver}
    ondragover={handleDragOver}
    ondragleave={handleDragLeave}
    ondrop={handleDrop}
    role="region"
    aria-label="File upload area"
  >
    {#if uploading}
      <div class="upload-progress" in:fade>
        <div class="progress-info">
          <span class="filename">{currentUpload}</span>
          <span class="progress-text">{uploadProgress}%</span>
        </div>
        <div class="progress-bar">
          <div class="progress-fill" style="width: {uploadProgress}%"></div>
        </div>
        {#if indexingStatus}
          <p class="status-text">
            {#if indexingStatus.status === "processing"}
              Indexing... {indexingStatus.pages_processed}/{indexingStatus.pages_total}
              pages
            {:else if indexingStatus.status === "completed"}
              ✓ Indexing complete
            {:else if indexingStatus.status === "failed"}
              ✗ Indexing failed: {indexingStatus.error}
            {:else}
              Queued for indexing...
            {/if}
          </p>
        {/if}
      </div>
    {:else}
      <div class="upload-prompt">
        <span class="upload-icon">📄</span>
        <p>Drag and drop a PDF here, or</p>
        <label class="upload-btn">
          Browse Files
          <input
            type="file"
            accept=".pdf"
            onchange={handleFileInput}
            style="display: none"
          />
        </label>
      </div>
    {/if}
  </div>

  <!-- Search Section -->
  <div class="search-section">
    <div class="search-input-wrapper">
      <input
        type="text"
        placeholder="Search indexed documents..."
        bind:value={searchQuery}
        onkeydown={(e) => e.key === "Enter" && searchDocuments()}
      />
      <button class="search-btn" onclick={searchDocuments}> 🔍 </button>
    </div>

    {#if searchResults.length > 0}
      <div class="search-results" in:slide>
        <h4>Search Results</h4>
        <ul>
          {#each searchResults as result}
            <li>
              <span class="result-filename">{result.filename}</span>
              <span class="result-page">Page {result.page}</span>
              <span class="result-relevance"
                >{(result.relevance * 100).toFixed(0)}% match</span
              >
            </li>
          {/each}
        </ul>
      </div>
    {/if}
  </div>

  <!-- Documents List -->
  <div class="documents-section">
    <h3>Indexed Documents ({documents.length})</h3>

    {#if loading}
      <div class="loading-state">
        <div class="spinner"></div>
        <p>Loading documents...</p>
      </div>
    {:else if documents.length === 0}
      <div class="empty-state">
        <span class="empty-icon">📚</span>
        <p>No documents indexed yet</p>
        <p class="hint">Upload a PDF to get started</p>
      </div>
    {:else}
      <div class="documents-list">
        {#each documents as doc (doc.id)}
          <div class="document-card" in:fly={{ y: 20 }}>
            <div class="doc-icon">📄</div>
            <div class="doc-info">
              <h4 class="doc-name">{doc.filename}</h4>
              <div class="doc-meta">
                <span>{formatSize(doc.size_bytes)}</span>
                <span>•</span>
                <span>{doc.pages} pages</span>
                <span>•</span>
                <span>{doc.namespace}</span>
                <span>•</span>
                <span>Indexed {formatDate(doc.indexed_at)}</span>
              </div>
            </div>
            <div class="doc-actions">
              <button
                class="delete-btn"
                onclick={() => deleteDocument(doc.id)}
                title="Delete document"
              >
                🗑️
              </button>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>

  <!-- Scraped Articles Section -->
  <div class="articles-section">
    <div class="articles-header">
      <h3>Scraped Articles ({articles.length})</h3>
      <div class="articles-controls">
        <!-- Category Filter -->
        <select
          bind:value={selectedCategory}
          class="filter-select"
        >
          <option value="">All Categories</option>
          {#each availableCategories as cat}
            <option value={cat}>{cat}</option>
          {/each}
        </select>

        <!-- Sort By -->
        <select
          bind:value={sortBy}
          class="filter-select"
        >
          <option value="name">Name</option>
          <option value="size">Size</option>
          <option value="modified">Date</option>
          <option value="category">Category</option>
        </select>

        <!-- Sort Order -->
        <button
          class="sort-order-btn"
          on:click={() => sortOrder = sortOrder === 'asc' ? 'desc' : 'asc'}
          title={sortOrder === 'asc' ? 'Ascending' : 'Descending'}
        >
          {sortOrder === 'asc' ? '↑' : '↓'}
        </button>
      </div>
    </div>

    {#if articlesLoading}
      <div class="loading-state">
        <p>Loading articles...</p>
      </div>
    {:else if articles.length === 0}
      <div class="empty-state">
        <p>No scraped articles found.</p>
      </div>
    {:else}
      <div class="articles-list">
        {#each articles as article}
          <div class="article-item">
            <div class="article-info">
              <div class="article-name">{article.name}</div>
              <div class="article-meta">
                <span class="category-tag">{article.category}</span>
                <span>•</span>
                <span>{formatSize(article.size_bytes)}</span>
                <span>•</span>
                <span class="date">{article.modified ? new Date(article.modified).toLocaleDateString() : ''}</span>
              </div>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  .knowledge-hub {
    padding: 24px;
    max-width: 900px;
    margin: 0 auto;
  }

  .hub-header {
    margin-bottom: 24px;
  }

  .hub-header h2 {
    margin: 0;
    font-size: 1.5rem;
    color: var(--text-primary, #cdd6f4);
  }

  .subtitle {
    margin: 8px 0 0;
    color: var(--text-secondary, #a6adc8);
    font-size: 0.875rem;
  }

  /* Sync Status Panel */
  .sync-status-panel {
    background: var(--bg-secondary, #1e1e2e);
    border: 1px solid var(--border-color, #313244);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
  }

  .sync-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
  }

  .sync-info {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .sync-title {
    font-weight: 600;
    color: var(--text-primary, #cdd6f4);
    font-size: 0.875rem;
  }

  .sync-status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 500;
    background: rgba(137, 180, 250, 0.15);
    color: var(--accent, #89b4fa);
  }

  .sync-status-badge.syncing {
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
  }

  .sync-status-badge.error {
    background: rgba(243, 139, 168, 0.15);
    color: #f38ba8;
  }

  .sync-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--accent, #89b4fa);
    color: var(--bg-primary, #1e1e2e);
    border: none;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 0.8rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .sync-btn:hover:not(:disabled) {
    opacity: 0.9;
    transform: translateY(-1px);
  }

  .sync-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .scraper-selector {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0 12px;
  }

  .scraper-select {
    background: var(--bg-secondary, #313244);
    color: var(--text-primary, #cdd6f4);
    border: 1px solid var(--border, #45475a);
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 0.75rem;
    cursor: pointer;
  }

  .scraper-select:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .api-key-btn {
    background: transparent;
    border: 1px solid var(--accent, #89b4fa);
    color: var(--accent, #89b4fa);
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 0.7rem;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .api-key-btn:hover {
    background: var(--accent, #89b4fa);
    color: var(--bg-primary, #1e1e2e);
  }

  .sync-details {
    display: flex;
    flex-direction: column;
    gap: 12px;
    font-size: 0.8rem;
  }

  .sync-stats-row {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
  }

  .sync-progress {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .progress-bar {
    flex: 1;
    height: 6px;
    background: var(--bg-tertiary, #45475a);
    border-radius: 3px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: var(--accent, #89b4fa);
    transition: width 0.3s ease;
  }

  .progress-text {
    font-size: 0.75rem;
    color: var(--accent, #89b4fa);
    min-width: 80px;
  }

  .categories-breakdown {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }

  .category-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .category-count {
    font-size: 0.7rem;
    padding: 2px 6px;
    background: var(--bg-tertiary, #45475a);
    border-radius: 4px;
    color: var(--text-secondary, #a6adc8);
  }

  .sync-stat {
    display: flex;
    gap: 6px;
  }

  .stat-label {
    color: var(--text-secondary, #a6adc8);
  }

  .stat-value {
    color: var(--text-primary, #cdd6f4);
    font-weight: 500;
  }

  .sync-errors {
    display: flex;
    gap: 6px;
    width: 100%;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--border-color, #313244);
  }

  .error-label {
    color: #f38ba8;
  }

  .error-value {
    color: #f38ba8;
    font-size: 0.75rem;
  }

  .spinner-small {
    width: 12px;
    height: 12px;
    border: 2px solid transparent;
    border-top-color: currentColor;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: rgba(243, 139, 168, 0.1);
    border: 1px solid #f38ba8;
    border-radius: 8px;
    margin-bottom: 16px;
    color: #f38ba8;
  }

  .dismiss-btn {
    margin-left: auto;
    background: transparent;
    border: none;
    color: inherit;
    font-size: 1.25rem;
    cursor: pointer;
  }

  .namespace-selector {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
  }

  .namespace-selector label {
    color: var(--text-secondary, #a6adc8);
    font-size: 0.875rem;
  }

  .namespace-selector select {
    background: var(--bg-tertiary, #313244);
    border: 1px solid var(--border-color, #45475a);
    border-radius: 6px;
    padding: 8px 12px;
    color: var(--text-primary, #cdd6f4);
    font-size: 0.875rem;
  }

  .upload-area {
    border: 2px dashed var(--border-color, #45475a);
    border-radius: 12px;
    padding: 48px;
    text-align: center;
    transition: all 0.2s ease;
    margin-bottom: 24px;
    background: var(--bg-secondary, #1e1e2e);
  }

  .upload-area.drag-over {
    border-color: var(--accent, #89b4fa);
    background: rgba(137, 180, 250, 0.05);
  }

  .upload-prompt {
    color: var(--text-secondary, #a6adc8);
  }

  .upload-icon {
    font-size: 3rem;
    display: block;
    margin-bottom: 12px;
  }

  .upload-btn {
    display: inline-block;
    background: var(--accent, #89b4fa);
    color: var(--bg-primary, #1e1e2e);
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 500;
    margin-top: 12px;
  }

  .upload-btn:hover {
    opacity: 0.9;
  }

  .upload-progress {
    text-align: center;
  }

  .progress-info {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
  }

  .filename {
    color: var(--text-primary, #cdd6f4);
    font-weight: 500;
  }

  .progress-text {
    color: var(--text-secondary, #a6adc8);
  }

  .progress-bar {
    height: 8px;
    background: var(--bg-tertiary, #313244);
    border-radius: 4px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: var(--accent, #89b4fa);
    transition: width 0.3s ease;
  }

  .status-text {
    margin-top: 8px;
    font-size: 0.875rem;
    color: var(--text-secondary, #a6adc8);
  }

  .search-section {
    margin-bottom: 24px;
  }

  .search-input-wrapper {
    display: flex;
    gap: 8px;
  }

  .search-input-wrapper input {
    flex: 1;
    background: var(--bg-tertiary, #313244);
    border: 1px solid var(--border-color, #45475a);
    border-radius: 8px;
    padding: 12px 16px;
    color: var(--text-primary, #cdd6f4);
    font-size: 0.875rem;
  }

  .search-input-wrapper input:focus {
    outline: none;
    border-color: var(--accent, #89b4fa);
  }

  .search-btn {
    background: var(--accent, #89b4fa);
    border: none;
    border-radius: 8px;
    padding: 0 16px;
    cursor: pointer;
    font-size: 1.125rem;
  }

  .search-results {
    margin-top: 16px;
    background: var(--bg-secondary, #1e1e2e);
    border-radius: 8px;
    padding: 16px;
  }

  .search-results h4 {
    margin: 0 0 12px;
    font-size: 0.875rem;
    color: var(--text-secondary, #a6adc8);
  }

  .search-results ul {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  .search-results li {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid var(--border-color, #313244);
    font-size: 0.875rem;
  }

  .result-filename {
    color: var(--text-primary, #cdd6f4);
  }

  .result-page,
  .result-relevance {
    color: var(--text-secondary, #a6adc8);
  }

  .documents-section h3 {
    margin: 0 0 16px;
    font-size: 1rem;
    color: var(--text-primary, #cdd6f4);
  }

  .loading-state,
  .empty-state {
    text-align: center;
    padding: 48px;
    color: var(--text-secondary, #a6adc8);
  }

  .spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--border-color, #45475a);
    border-top-color: var(--accent, #89b4fa);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto 12px;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .empty-icon {
    font-size: 3rem;
    display: block;
    margin-bottom: 12px;
  }

  .hint {
    font-size: 0.75rem;
    margin-top: 8px;
  }

  .documents-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .document-card {
    display: flex;
    align-items: center;
    gap: 16px;
    background: var(--bg-secondary, #1e1e2e);
    border: 1px solid var(--border-color, #313244);
    border-radius: 8px;
    padding: 16px;
  }

  .doc-icon {
    font-size: 1.5rem;
  }

  .doc-info {
    flex: 1;
  }

  .doc-name {
    margin: 0;
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--text-primary, #cdd6f4);
  }

  .doc-meta {
    margin-top: 4px;
    font-size: 0.75rem;
    color: var(--text-secondary, #a6adc8);
    display: flex;
    gap: 8px;
  }

  .delete-btn {
    background: transparent;
    border: none;
    cursor: pointer;
    padding: 8px;
    border-radius: 4px;
    opacity: 0.7;
  }

  .delete-btn:hover {
    opacity: 1;
    background: rgba(243, 139, 168, 0.1);
  }

  /* Articles Section */
  .articles-section {
    margin-top: 24px;
    padding-top: 24px;
    border-top: 1px solid var(--border-color, #313244);
  }

  .articles-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    flex-wrap: wrap;
    gap: 12px;
  }

  .articles-section h3 {
    margin: 0;
    font-size: 1rem;
    color: var(--text-primary, #cdd6f4);
  }

  .articles-controls {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .filter-select {
    background: var(--bg-secondary, #313244);
    color: var(--text-primary, #cdd6f4);
    border: 1px solid var(--border-color, #45475a);
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 0.75rem;
    cursor: pointer;
  }

  .filter-select:focus {
    outline: none;
    border-color: var(--accent, #89b4fa);
  }

  .sort-order-btn {
    background: var(--bg-secondary, #313244);
    border: 1px solid var(--border-color, #45475a);
    color: var(--text-primary, #cdd6f4);
    border-radius: 4px;
    padding: 4px 8px;
    cursor: pointer;
    font-size: 0.8rem;
  }

  .sort-order-btn:hover {
    background: var(--bg-tertiary, #45475a);
  }

  .articles-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .article-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    background: var(--bg-secondary, #1e1e2e);
    border: 1px solid var(--border-color, #313244);
    border-radius: 8px;
  }

  .article-info {
    flex: 1;
  }

  .article-name {
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--text-primary, #cdd6f4);
    margin-bottom: 4px;
  }

  .article-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.75rem;
    color: var(--text-secondary, #a6adc8);
  }

  .article-meta .date {
    font-size: 0.7rem;
    color: var(--text-secondary, #a6adc8);
  }

  .category-tag {
    padding: 2px 8px;
    background: rgba(137, 180, 250, 0.15);
    color: var(--accent, #89b4fa);
    border-radius: 4px;
    font-size: 0.7rem;
    text-transform: capitalize;
  }

  /* Modal Styles */
  .modal-backdrop {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .modal-content {
    background: var(--bg-secondary, #313244);
    border: 1px solid var(--border, #45475a);
    border-radius: 12px;
    width: 90%;
    max-width: 450px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border, #45475a);
  }

  .modal-header h3 {
    margin: 0;
    font-size: 1rem;
    color: var(--text-primary, #cdd6f4);
  }

  .close-btn {
    background: transparent;
    border: none;
    color: var(--text-secondary, #a6adc8);
    font-size: 1.5rem;
    cursor: pointer;
    padding: 0;
    line-height: 1;
  }

  .close-btn:hover {
    color: var(--text-primary, #cdd6f4);
  }

  .modal-body {
    padding: 20px;
  }

  .modal-description {
    margin: 0 0 16px;
    font-size: 0.85rem;
    color: var(--text-secondary, #a6adc8);
    line-height: 1.5;
  }

  .modal-description a {
    color: var(--accent, #89b4fa);
    text-decoration: none;
  }

  .modal-description a:hover {
    text-decoration: underline;
  }

  .form-group {
    margin-bottom: 16px;
  }

  .form-group:last-child {
    margin-bottom: 0;
  }

  .form-group label {
    display: block;
    margin-bottom: 6px;
    font-size: 0.8rem;
    color: var(--text-primary, #cdd6f4);
    font-weight: 500;
  }

  .form-input, .form-select {
    width: 100%;
    padding: 10px 12px;
    background: var(--bg-primary, #1e1e2e);
    border: 1px solid var(--border, #45475a);
    border-radius: 6px;
    color: var(--text-primary, #cdd6f4);
    font-size: 0.9rem;
    box-sizing: border-box;
  }

  .form-input:focus, .form-select:focus {
    outline: none;
    border-color: var(--accent, #89b4fa);
  }

  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 12px;
    padding: 16px 20px;
    border-top: 1px solid var(--border, #45475a);
  }

  .btn-secondary {
    background: transparent;
    border: 1px solid var(--border, #45475a);
    color: var(--text-primary, #cdd6f4);
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .btn-secondary:hover {
    background: var(--bg-tertiary, #45475a);
  }

  .btn-primary {
    background: var(--accent, #89b4fa);
    border: none;
    color: var(--bg-primary, #1e1e2e);
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .btn-primary:hover:not(:disabled) {
    opacity: 0.9;
  }

  .btn-primary:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
</style>

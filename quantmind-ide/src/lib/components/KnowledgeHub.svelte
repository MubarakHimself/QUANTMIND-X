<script lang="ts">
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
  let documents: PDFDocument[] = [];
  let articles: KnowledgeArticle[] = [];
  let namespaces: Namespace[] = [];
  let loading = true;
  let uploading = false;
  let error: string | null = null;
  let articlesLoading = false;

  // Upload state
  let dragOver = false;
  let uploadProgress = 0;
  let currentUpload: string | null = null;
  let indexingStatus: IndexingStatus | null = null;

  // Selected namespace
  let selectedNamespace = "knowledge";

  // Search state
  let searchQuery = "";
  let searchResults: any[] = [];

  // Update breadcrumbs when namespace changes
  $: if (selectedNamespace) {
    navigationStore.navigateToFolder(
      selectedNamespace,
      selectedNamespace.charAt(0).toUpperCase() + selectedNamespace.slice(1),
    );
  }

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
      const response = await fetch("/api/knowledge");
      if (response.ok) {
        articles = await response.json();
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
    await Promise.all([fetchDocuments(), fetchNamespaces(), fetchArticles()]);
    loading = false;
  });
</script>

<div class="knowledge-hub">
  <!-- Header -->
  <div class="hub-header">
    <h2>Knowledge Hub</h2>
    <p class="subtitle">Upload and index PDF documents for AI-powered search</p>
  </div>

  <!-- Error Display -->
  {#if error}
    <div class="error-banner" in:fly={{ y: -20 }}>
      <span class="error-icon">⚠️</span>
      <span>{error}</span>
      <button class="dismiss-btn" on:click={() => (error = null)}>×</button>
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
    on:dragover={handleDragOver}
    on:dragleave={handleDragLeave}
    on:drop={handleDrop}
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
            on:change={handleFileInput}
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
        on:keydown={(e) => e.key === "Enter" && searchDocuments()}
      />
      <button class="search-btn" on:click={searchDocuments}> 🔍 </button>
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
                on:click={() => deleteDocument(doc.id)}
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
    <h3>Scraped Articles ({articles.length})</h3>

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
        {#each articles.slice(0, 20) as article}
          <div class="article-item">
            <div class="article-info">
              <div class="article-name">{article.name}</div>
              <div class="article-meta">
                <span class="category-tag">{article.category}</span>
                <span>•</span>
                <span>{formatSize(article.size_bytes)}</span>
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

  .articles-section h3 {
    margin: 0 0 16px;
    font-size: 1rem;
    color: var(--text-primary, #cdd6f4);
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

  .category-tag {
    padding: 2px 8px;
    background: rgba(137, 180, 250, 0.15);
    color: var(--accent, #89b4fa);
    border-radius: 4px;
    font-size: 0.7rem;
    text-transform: capitalize;
  }
</style>

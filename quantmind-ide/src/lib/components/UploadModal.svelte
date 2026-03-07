<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import {
    X,
    Upload,
    FileText,
    Library,
    File,
  } from "lucide-svelte";

  export let isOpen = false;
  export let uploadType: "book" | "article" | "note" = "article";
  export let dragOver = false;
  export let metadata = {
    title: "",
    author: "",
    category: "",
    url: "",
    content: "",
  };
  export let uploadingFiles: Array<{
    name: string;
    progress: number;
    status: "pending" | "uploading" | "done" | "error";
  }> = [];

  const dispatch = createEventDispatcher();

  function handleClose() {
    dispatch("close");
  }

  function handleFileSelect(e: Event) {
    const target = e.target as HTMLInputElement;
    dispatch("fileSelect", { files: target.files });
  }

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    dispatch("drop", { files: e.dataTransfer?.files });
  }

  function handleDragOver(e: DragEvent) {
    e.preventDefault();
    dispatch("dragOver", { value: true });
  }

  function handleDragLeave() {
    dispatch("dragOver", { value: false });
  }

  function handleNoteSubmit() {
    dispatch("noteSubmit", { metadata });
  }
</script>

{#if isOpen}
  <div
    class="modal-overlay"
    on:click|self={handleClose}
    role="button"
    tabindex="0"
    on:keydown={(e) => e.key === "Enter" && handleClose()}
  >
    <div class="modal upload-modal">
      <div class="modal-header">
        <h2><Upload size={20} /> Upload to Knowledge Hub</h2>
        <button on:click={handleClose}
          ><X size={20} /></button
        >
      </div>
      <div class="modal-body">
        <!-- Upload Type Selector -->
        <div class="form-group">
          <label for="upload-type-selector">Upload Type</label>
          <div
            id="upload-type-selector"
            class="upload-type-selector"
            role="radiogroup"
          >
            <label
              class="type-option"
              class:selected={uploadType === "article"}
            >
              <input
                type="radio"
                bind:group={uploadType}
                value="article"
              />
              <div class="type-content">
                <FileText size={20} />
                <div>
                  <span class="type-name">Article</span>
                  <span class="type-desc"
                    >Blog posts, research papers, tutorials</span
                  >
                </div>
              </div>
            </label>
            <label
              class="type-option"
              class:selected={uploadType === "book"}
            >
              <input
                type="radio"
                bind:group={uploadType}
                value="book"
              />
              <div class="type-content">
                <Library size={20} />
                <div>
                  <span class="type-name">Book (PDF)</span>
                  <span class="type-desc"
                    >PDF indexing with LangMem</span
                  >
                </div>
              </div>
            </label>
            <label
              class="type-option"
              class:selected={uploadType === "note"}
            >
              <input
                type="radio"
                bind:group={uploadType}
                value="note"
              />
              <div class="type-content">
                <FileText size={20} />
                <div>
                  <span class="type-name">Note</span>
                  <span class="type-desc"
                    >Quick notes without indexing</span
                  >
                </div>
              </div>
            </label>
          </div>
        </div>

        <!-- Dynamic form fields based on type -->
        {#if uploadType === "book"}
          <div class="metadata-fields book-fields">
            <div class="form-group">
              <label for="book-title"
                >Title <span class="optional">(optional)</span></label
              >
              <input
                type="text"
                id="book-title"
                placeholder="Book title"
                bind:value={metadata.title}
              />
            </div>
            <div class="form-group">
              <label for="book-author"
                >Author <span class="optional">(optional)</span></label
              >
              <input
                type="text"
                id="book-author"
                placeholder="Author name"
                bind:value={metadata.author}
              />
            </div>
            <div class="form-group">
              <label for="book-category"
                >Category <span class="optional">(optional)</span
                ></label
              >
              <select
                id="book-category"
                bind:value={metadata.category}
              >
                <option value="">Select category...</option>
                <option value="trading">Trading</option>
                <option value="programming">Programming</option>
                <option value="mathematics">Mathematics</option>
                <option value="economics">Economics</option>
                <option value="psychology">Psychology</option>
                <option value="other">Other</option>
              </select>
            </div>
          </div>
        {:else if uploadType === "article"}
          <div class="metadata-fields article-fields">
            <div class="form-group">
              <label for="article-title"
                >Title <span class="optional">(optional)</span></label
              >
              <input
                id="article-title"
                type="text"
                placeholder="Article title"
                bind:value={metadata.title}
              />
            </div>
            <div class="form-group">
              <label for="article-author"
                >Author <span class="optional">(optional)</span></label
              >
              <input
                id="article-author"
                type="text"
                placeholder="Author name"
                bind:value={metadata.author}
              />
            </div>
            <div class="form-group">
              <label for="article-url"
                >Source URL <span class="optional">(optional)</span
                ></label
              >
              <input
                id="article-url"
                type="url"
                placeholder="https://..."
                bind:value={metadata.url}
              />
            </div>
            <div class="form-group">
              <label for="article-category"
                >Category <span class="optional">(optional)</span
                ></label
              >
              <select
                id="article-category"
                bind:value={metadata.category}
              >
                <option value="">Select category...</option>
                <option value="trading-strategies"
                  >Trading Strategies</option
                >
                <option value="technical-analysis"
                  >Technical Analysis</option
                >
                <option value="risk-management">Risk Management</option>
                <option value="programming">Programming</option>
                <option value="market-research">Market Research</option>
                <option value="other">Other</option>
              </select>
            </div>
          </div>
        {:else if uploadType === "note"}
          <div class="metadata-fields note-fields">
            <div class="form-group">
              <label for="note-title">Title *</label>
              <input
                id="note-title"
                type="text"
                placeholder="Note title"
                bind:value={metadata.title}
                required
              />
            </div>
            <div class="form-group">
              <label>Content *</label>
              <textarea
                class="note-content"
                placeholder="Write your note here... (Markdown supported)"
                bind:value={metadata.content}
                rows="6"
              ></textarea>
            </div>
          </div>
        {/if}

        <!-- File upload area (not for notes) -->
        {#if uploadType !== "note"}
          <div
            class="upload-dropzone"
            class:drag-over={dragOver}
            on:dragover={handleDragOver}
            on:dragleave={handleDragLeave}
            on:drop={handleDrop}
          >
            <Upload size={40} />
            <p>
              Drag & drop {uploadType === "book"
                ? "PDF files"
                : "files"} here
            </p>
            <span>or</span>
            <label class="file-input-label">
              <input
                type="file"
                multiple
                accept={uploadType === "book"
                  ? ".pdf"
                  : ".pdf,.md,.txt,.csv,.json,.html"}
                on:change={handleFileSelect}
              />
              Browse Files
            </label>
            <p class="hint">
              Supports: {uploadType === "book"
                ? "PDF (will be indexed)"
                : "PDF, Markdown, TXT, CSV, JSON, HTML"}
            </p>
          </div>
        {:else}
          <div class="note-actions">
            <button class="btn primary" on:click={handleNoteSubmit}>
              <Upload size={14} /> Save Note
            </button>
          </div>
        {/if}

        <!-- Upload progress -->
        {#if uploadingFiles.length > 0}
          <div class="upload-progress-list">
            <h4>Uploading Files</h4>
            {#each uploadingFiles as file}
              <div
                class="upload-item"
                class:done={file.status === "done"}
                class:error={file.status === "error"}
              >
                <File size={14} />
                <span class="filename">{file.name}</span>
                <span
                  class="status-badge"
                  class:uploading={file.status === "uploading"}
                  class:done={file.status === "done"}
                  class:error={file.status === "error"}
                >
                  {file.status === "uploading"
                    ? `Uploading... ${file.progress}%`
                    : file.status === "done"
                      ? "Done"
                      : file.status === "error"
                        ? "Failed"
                        : "Pending"}
                </span>
                {#if file.status === "uploading"}
                  <div class="progress-bar">
                    <div style="width: {file.progress}%"></div>
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        {/if}
      </div>
      <div class="modal-footer">
        <button
          class="btn secondary"
          on:click={handleClose}>Close</button
        >
      </div>
    </div>
  </div>
{/if}

<style>
  .modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
  }

  .modal {
    background: var(--bg-secondary);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
  }

  .upload-modal {
    min-width: 450px;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 16px;
    height: 48px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .modal-header h2 {
    margin: 0;
    font-size: 16px;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .modal-header button {
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
  }

  .modal-body {
    padding: 20px;
  }

  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    padding: 16px 20px;
    border-top: 1px solid var(--border-subtle);
  }

  .form-group {
    margin-bottom: 16px;
  }

  .form-group label {
    display: block;
    margin-bottom: 6px;
    font-size: 12px;
    color: var(--text-muted);
  }

  .form-group input,
  .form-group select {
    width: 100%;
    padding: 10px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
  }

  .optional {
    font-size: 10px;
    color: var(--text-muted);
    font-weight: normal;
  }

  .metadata-fields {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin-top: 16px;
    padding: 16px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .metadata-fields .form-group {
    margin-bottom: 0;
  }

  .note-content {
    width: 100%;
    padding: 10px 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
    font-family: "JetBrains Mono", monospace;
    resize: vertical;
    outline: none;
    min-height: 120px;
  }

  .note-content:focus {
    border-color: var(--accent-primary);
  }

  .upload-type-selector {
    display: flex;
    gap: 12px;
    margin-top: 8px;
  }

  .type-option {
    flex: 1;
    display: flex;
    align-items: center;
    padding: 12px;
    background: var(--bg-tertiary);
    border: 2px solid var(--border-subtle);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .type-option:hover {
    border-color: var(--accent-primary);
    background: rgba(99, 102, 241, 0.05);
  }

  .type-option.selected {
    border-color: var(--accent-primary);
    background: rgba(99, 102, 241, 0.1);
  }

  .type-option input {
    display: none;
  }

  .type-content {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
  }

  .type-content :global(svg) {
    color: var(--accent-primary);
    flex-shrink: 0;
  }

  .type-content div {
    display: flex;
    flex-direction: column;
  }

  .type-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .type-desc {
    font-size: 11px;
    color: var(--text-muted);
  }

  .upload-dropzone {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 40px 20px;
    margin-top: 12px;
    border: 2px dashed var(--border-subtle);
    border-radius: 12px;
    background: var(--bg-primary);
    color: var(--text-muted);
    text-align: center;
    transition: all 0.2s ease;
    cursor: pointer;
  }

  .upload-dropzone:hover,
  .upload-dropzone.drag-over {
    border-color: var(--accent-primary);
    background: rgba(99, 102, 241, 0.05);
    color: var(--accent-primary);
  }

  .upload-dropzone p {
    margin: 0;
    font-size: 14px;
  }

  .upload-dropzone span {
    font-size: 12px;
    color: var(--text-muted);
  }

  .upload-dropzone .hint {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 8px;
  }

  .file-input-label {
    display: inline-block;
    padding: 8px 16px;
    background: var(--accent-primary);
    color: var(--bg-primary);
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: opacity 0.15s ease;
  }

  .file-input-label:hover {
    opacity: 0.9;
  }

  .file-input-label input {
    display: none;
  }

  .note-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    margin-top: 16px;
  }

  .btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 13px;
    cursor: pointer;
  }

  .btn.secondary {
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
  }

  .btn.primary {
    background: var(--accent-primary);
    border: none;
    color: var(--bg-primary);
  }

  .upload-progress-list {
    margin-top: 16px;
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .upload-progress-list h4 {
    margin: 0 0 8px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .upload-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 0;
    font-size: 12px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .upload-item:last-child {
    border-bottom: none;
  }

  .upload-item .filename {
    flex: 1;
    color: var(--text-primary);
  }

  .upload-item .status-badge {
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 500;
  }

  .upload-item .status-badge.uploading {
    background: rgba(99, 102, 241, 0.2);
    color: var(--accent-primary);
  }

  .upload-item .status-badge.done {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .upload-item .status-badge.error {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .upload-item.done {
    opacity: 0.6;
  }

  .upload-item.error .filename {
    color: #ef4444;
  }

  .progress-bar {
    flex: 1;
    height: 4px;
    background: var(--border-subtle);
    border-radius: 2px;
    overflow: hidden;
  }

  .progress-bar div {
    height: 100%;
    background: var(--accent-primary);
  }
</style>

<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { 
    FileText, Code, BookOpen, Image, File,
    ChevronRight, ChevronDown, X, Download, Copy,
    Maximize2, Minimize2, ZoomIn, ZoomOut, RotateCw
  } from 'lucide-svelte';

  export let file: {
    id: string;
    name: string;
    path?: string;
    content?: string;
    type?: string;
    metadata?: {
      size?: number;
      created_at?: string;
      updated_at?: string;
      checksum?: string;
    };
  } | null = null;

  export let showMetadata = true;
  export let allowDownload = true;

  // Flag for external reference (not used internally)
  export const showNavigation = true;

  const dispatch = createEventDispatcher();

  let isMaximized = false;
  let copied = false;
  let content = '';
  let imageZoom = 1;
  let imageRotation = 0;

  const API_BASE = 'http://localhost:8000/api';

  // File type detection
  const fileTypes = {
    image: ['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp'],
    code: ['mq5', 'mqh', 'py', 'ts', 'js', 'json', 'sql', 'yaml', 'yml', 'xml', 'html', 'css'],
    markdown: ['md', 'txt'],
    document: ['pdf', 'doc', 'docx']
  };

  $: fileType = getFileType(file?.name || '');
  $: isImage = fileType === 'image';
  $: isCode = fileType === 'code';
  $: isMarkdown = fileType === 'markdown';
  $: isBinary = fileType === 'binary';

  $: if (file?.content) {
    content = file.content;
  } else if (file && !isImage) {
    loadFileContent();
  }

  function getFileType(filename: string): string {
    const ext = filename.split('.').pop()?.toLowerCase() || '';
    
    if (fileTypes.image.includes(ext)) return 'image';
    if (fileTypes.code.includes(ext)) return 'code';
    if (fileTypes.markdown.includes(ext)) return 'markdown';
    if (fileTypes.document.includes(ext)) return 'document';
    
    return 'text';
  }

  async function loadFileContent() {
    if (!file || isImage) return;
    
    try {
      const endpoint = `${API_BASE}/files/${file.id}/content`;
      const response = await fetch(endpoint);
      if (response.ok) {
        const data = await response.json();
        content = data.content || '';
      }
    } catch (e) {
      console.error('Failed to load file content:', e);
      content = `// Failed to load file content\n// File: ${file.name}`;
    }
  }

  function formatFileSize(bytes?: number): string {
    if (!bytes) return 'Unknown';
    const units = ['B', 'KB', 'MB', 'GB'];
    let unitIndex = 0;
    let size = bytes;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  }

  function formatDate(dateStr?: string): string {
    if (!dateStr) return 'Unknown';
    const date = new Date(dateStr);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  }

  function getFileIcon(filename: string) {
    const ext = filename.split('.').pop()?.toLowerCase();
    if (ext === 'mq5' || ext === 'mqh') return Code;
    if (ext === 'md') return BookOpen;
    if (fileTypes.image.includes(ext || '')) return Image;
    return FileText;
  }

  function copyContent() {
    navigator.clipboard.writeText(content);
    copied = true;
    setTimeout(() => copied = false, 2000);
  }

  function downloadFile() {
    if (!file) return;
    
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = file.name;
    a.click();
    URL.revokeObjectURL(url);
  }

  function toggleMaximize() {
    isMaximized = !isMaximized;
    dispatch('maximize', { maximized: isMaximized });
  }

  function zoomIn() {
    imageZoom = Math.min(imageZoom + 0.25, 3);
  }

  function zoomOut() {
    imageZoom = Math.max(imageZoom - 0.25, 0.25);
  }

  function rotateImage() {
    imageRotation = (imageRotation + 90) % 360;
  }

  function handleClose() {
    dispatch('close');
  }

  // Syntax highlighting for code
  function getHighlightedContent(): string {
    return escapeHtml(content);
  }

  function escapeHtml(text: string): string {
    return text
      .replace(/&/g, '&')
      .replace(/</g, '<')
      .replace(/>/g, '>')
      .replace(/"/g, '"')
      .replace(/'/g, '&#039;');
  }
</script>

<div class="file-viewer" class:maximized={isMaximized}>
  <!-- Header -->
  <div class="viewer-header">
    <div class="header-left">
      <svelte:component this={getFileIcon(file?.name || 'file')} size={20} class="file-icon" />
      <div class="file-info">
        <span class="file-name">{file?.name || 'Untitled'}</span>
        {#if file?.path}
          <span class="file-path">{file.path}</span>
        {/if}
      </div>
    </div>
    
    <div class="header-actions">
      {#if allowDownload && content}
        <button class="action-btn" on:click={copyContent} title="Copy content">
          {#if copied}
            <span class="copied-check">✓</span>
          {:else}
            <Copy size={16} />
          {/if}
        </button>
        <button class="action-btn" on:click={downloadFile} title="Download">
          <Download size={16} />
        </button>
      {/if}
      
      {#if isImage}
        <button class="action-btn" on:click={zoomOut} title="Zoom out">
          <ZoomOut size={16} />
        </button>
        <span class="zoom-level">{Math.round(imageZoom * 100)}%</span>
        <button class="action-btn" on:click={zoomIn} title="Zoom in">
          <ZoomIn size={16} />
        </button>
        <button class="action-btn" on:click={rotateImage} title="Rotate">
          <RotateCw size={16} />
        </button>
      {/if}
      
      <button class="action-btn" on:click={toggleMaximize} title={isMaximized ? 'Minimize' : 'Maximize'}>
        {#if isMaximized}
          <Minimize2 size={16} />
        {:else}
          <Maximize2 size={16} />
        {/if}
      </button>
      
      <button class="action-btn close-btn" on:click={handleClose} title="Close">
        <X size={16} />
      </button>
    </div>
  </div>

  <!-- Metadata Bar -->
  {#if showMetadata && file}
    <div class="metadata-bar">
      <div class="metadata-item">
        <span class="meta-label">Size:</span>
        <span class="meta-value">{formatFileSize(file.metadata?.size)}</span>
      </div>
      <div class="metadata-item">
        <span class="meta-label">Type:</span>
        <span class="meta-value">{fileType.toUpperCase()}</span>
      </div>
      {#if file.metadata?.created_at}
        <div class="metadata-item">
          <span class="meta-label">Created:</span>
          <span class="meta-value">{formatDate(file.metadata.created_at)}</span>
        </div>
      {/if}
      {#if file.metadata?.updated_at}
        <div class="metadata-item">
          <span class="meta-label">Modified:</span>
          <span class="meta-value">{formatDate(file.metadata.updated_at)}</span>
        </div>
      {/if}
      {#if file.metadata?.checksum}
        <div class="metadata-item checksum">
          <span class="meta-label">Checksum:</span>
          <span class="meta-value">{file.metadata.checksum}</span>
        </div>
      {/if}
    </div>
  {/if}

  <!-- Content Area -->
  <div class="viewer-content">
    {#if isImage}
      <!-- Image Viewer -->
      <div class="image-viewer">
        <img 
          src={file?.content || file?.path} 
          alt={file?.name}
          style="transform: rotate({imageRotation}deg) scale({imageZoom});"
          class="preview-image"
        />
      </div>
    {:else if isCode}
      <!-- Code Viewer -->
      <div class="code-viewer">
        <pre class="code-content"><code>{@html getHighlightedContent()}</code></pre>
      </div>
    {:else if isMarkdown}
      <!-- Markdown Viewer -->
      <div class="markdown-viewer">
        <div class="markdown-content">
          {@html renderMarkdown(content)}
        </div>
      </div>
    {:else if isBinary}
      <!-- Binary File Indicator -->
      <div class="binary-viewer">
        <div class="binary-icon">
          <File size={64} />
        </div>
        <h3>Binary File</h3>
        <p>This file type cannot be previewed directly.</p>
        {#if allowDownload}
          <button class="download-btn" on:click={downloadFile}>
            <Download size={16} />
            Download to view
          </button>
        {/if}
      </div>
    {:else}
      <!-- Plain Text Viewer -->
      <div class="text-viewer">
        <pre class="text-content">{content}</pre>
      </div>
    {/if}
  </div>

  <!-- Footer -->
  <div class="viewer-footer">
    <div class="footer-left">
      <span class="line-count">{content.split('\n').length} lines</span>
      <span class="char-count">{content.length} characters</span>
    </div>
    <div class="footer-right">
      <span class="encoding">UTF-8</span>
    </div>
  </div>
</div>

<script context="module" lang="ts">
  // Simple markdown renderer
  function renderMarkdown(text: string): string {
    let html = escapeHtml(text);
    
    // Code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Headers
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    
    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // Italic
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    
    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
    
    // Line breaks
    html = html.replace(/\n/g, '<br>');
    
    return html;
  }

  function escapeHtml(text: string): string {
    return text
      .replace(/&/g, '&')
      .replace(/</g, '<')
      .replace(/>/g, '>')
      .replace(/"/g, '"')
      .replace(/'/g, '&#039;');
  }
</script>

<style>
  .file-viewer {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary, #1a1b26);
    border: 1px solid var(--border-subtle, #2d2d3a);
    border-radius: 8px;
    overflow: hidden;
    transition: all 0.3s ease;
  }

  .file-viewer.maximized {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 1000;
    border-radius: 0;
    border: none;
  }

  /* Header */
  .viewer-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    background: var(--bg-tertiary, #1f1f2e);
    border-bottom: 1px solid var(--border-subtle, #2d2d3a);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .file-icon {
    color: var(--accent-primary, #7aa2f7);
  }

  .file-info {
    display: flex;
    flex-direction: column;
  }

  .file-name {
    font-weight: 600;
    color: var(--text-primary, #c0caf5);
    font-size: 14px;
  }

  .file-path {
    color: var(--text-muted, #565f89);
    font-size: 11px;
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .action-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    padding: 6px 10px;
    background: transparent;
    border: none;
    color: var(--text-secondary, #a9b1d6);
    cursor: pointer;
    border-radius: 4px;
    transition: all 0.15s ease;
  }

  .action-btn:hover {
    background: var(--bg-secondary, #292a3e);
    color: var(--text-primary, #c0caf5);
  }

  .action-btn.close-btn:hover {
    background: #f7768e;
    color: var(--bg-primary, #1a1b26);
  }

  .copied-check {
    color: #9ece6a;
    font-size: 12px;
  }

  .zoom-level {
    color: var(--text-muted, #565f89);
    font-size: 12px;
    min-width: 40px;
    text-align: center;
  }

  /* Metadata Bar */
  .metadata-bar {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 8px 16px;
    background: var(--bg-secondary, #16161e);
    border-bottom: 1px solid var(--border-subtle, #2d2d3a);
    font-size: 11px;
  }

  .metadata-item {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .meta-label {
    color: var(--text-muted, #565f89);
  }

  .meta-value {
    color: var(--text-secondary, #a9b1d6);
  }

  .checksum .meta-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--text-muted, #565f89);
  }

  /* Content Area */
  .viewer-content {
    flex: 1;
    overflow: auto;
    padding: 16px;
  }

  /* Image Viewer */
  .image-viewer {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    overflow: auto;
  }

  .preview-image {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    transition: transform 0.2s ease;
  }

  /* Code Viewer */
  .code-viewer {
    height: 100%;
    overflow: auto;
  }

  .code-content {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    line-height: 1.6;
    color: var(--text-primary, #c0caf5);
    background: transparent;
    margin: 0;
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  /* Markdown Viewer */
  .markdown-viewer {
    height: 100%;
    overflow: auto;
  }

  .markdown-content {
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    line-height: 1.7;
    color: var(--text-primary, #c0caf5);
  }

  .markdown-content :global(h1) {
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 16px;
    color: var(--text-primary, #c0caf5);
  }

  .markdown-content :global(h2) {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 12px;
    color: var(--text-primary, #c0caf5);
  }

  .markdown-content :global(h3) {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 8px;
    color: var(--text-primary, #c0caf5);
  }

  .markdown-content :global(pre) {
    background: var(--bg-secondary, #16161e);
    padding: 12px;
    border-radius: 6px;
    overflow-x: auto;
    margin: 12px 0;
  }

  .markdown-content :global(code) {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    background: var(--bg-tertiary, #1f1f2e);
    padding: 2px 6px;
    border-radius: 4px;
  }

  .markdown-content :global(a) {
    color: var(--accent-primary, #7aa2f7);
    text-decoration: none;
  }

  .markdown-content :global(a:hover) {
    text-decoration: underline;
  }

  /* Binary Viewer */
  .binary-viewer {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    text-align: center;
    color: var(--text-secondary, #a9b1d6);
  }

  .binary-icon {
    color: var(--text-muted, #565f89);
    margin-bottom: 16px;
  }

  .binary-viewer h3 {
    margin-bottom: 8px;
    color: var(--text-primary, #c0caf5);
  }

  .binary-viewer p {
    margin-bottom: 16px;
    color: var(--text-muted, #565f89);
    font-size: 13px;
  }

  .download-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    background: var(--accent-primary, #7aa2f7);
    color: var(--bg-primary, #1a1b26);
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 600;
    transition: background 0.15s ease;
  }

  .download-btn:hover {
    background: var(--accent-secondary, #bb9af7);
  }

  /* Text Viewer */
  .text-viewer {
    height: 100%;
    overflow: auto;
  }

  .text-content {
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    line-height: 1.7;
    color: var(--text-primary, #c0caf5);
    margin: 0;
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  /* Footer */
  .viewer-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 16px;
    background: var(--bg-tertiary, #1f1f2e);
    border-top: 1px solid var(--border-subtle, #2d2d3a);
    font-size: 11px;
    color: var(--text-muted, #565f89);
  }

  .footer-left {
    display: flex;
    gap: 16px;
  }

  .encoding {
    color: var(--text-muted, #565f89);
  }
</style>

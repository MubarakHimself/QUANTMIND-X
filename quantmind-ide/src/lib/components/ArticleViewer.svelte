<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import hljs from 'highlight.js';
  import { theme } from '../stores/themeStore';
  import { FileText, Edit3, X, Copy, Download, ExternalLink } from 'lucide-svelte';

  export let article: any;
  export let onClose: () => void;
  
  const dispatch = createEventDispatcher();
  
  let processedContent = '';
  let currentThemeColors = $theme.colors;

  // Subscribe to theme changes
  $: if ($theme) {
    currentThemeColors = $theme.colors;
    processContent();
  }

  function processContent() {
    if (!article?.content) {
      processedContent = '<p>No content available</p>';
      return;
    }

    let content = article.content;
    
    // Process code blocks with syntax highlighting
    content = content.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
      const language = lang || 'plaintext';
      let highlighted = code;
      
      try {
        if (lang && hljs.getLanguage(lang)) {
          highlighted = hljs.highlight(code, { language: lang }).value;
        } else {
          highlighted = hljs.highlightAuto(code).value;
        }
      } catch {
        highlighted = hljs.highlightAuto(code).value;
      }
      
      return `
        <div class="code-block">
          <div class="code-header">
            <span class="code-language">${lang || 'text'}</span>
            <div class="code-actions">
              <button class="code-btn" onclick="copyCode(this)" title="Copy">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                </svg>
              </button>
              <button class="code-btn" onclick="downloadCode(this)" title="Download">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="7 10 12 15 17 10"></polyline>
                  <line x1="12" y1="15" x2="12" y2="3"></line>
                </svg>
              </button>
            </div>
          </div>
          <pre class="code-content"><code class="hljs ${language}">${highlighted}</code></pre>
        </div>
      `;
    });
    
    // Process inline code
    content = content.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
    
    // Process links
    content = content.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    
    // Process images
    content = content.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" class="article-image" loading="lazy" />');
    
    // Process headers
    content = content.replace(/^### (.*$)/gim, '<h3 class="article-header h3">$1</h3>');
    content = content.replace(/^## (.*$)/gim, '<h2 class="article-header h2">$1</h2>');
    content = content.replace(/^# (.*$)/gim, '<h1 class="article-header h1">$1</h1>');
    
    // Process bold and italic
    content = content.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    content = content.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    
    // Process lists
    content = content.replace(/^\* (.+)$/gim, '<li>$1</li>');
    content = content.replace(/(<li>.*<\/li>)/s, '<ul class="article-list">$1</ul>');
    
    // Process paragraphs
    content = content.replace(/\n\n/g, '</p><p class="article-paragraph">');
    content = '<p class="article-paragraph">' + content + '</p>';
    
    processedContent = content;
  }

  function openInEditor() {
    dispatch('openInEditor', { article });
  }

  function copyCode(button: HTMLButtonElement) {
    const codeBlock = button.closest('.code-block')?.querySelector('.code-content');
    if (codeBlock) {
      navigator.clipboard.writeText(codeBlock.textContent || '');
      const originalText = button.innerHTML;
      button.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>';
      setTimeout(() => {
        button.innerHTML = originalText;
      }, 2000);
    }
  }

  function downloadCode(button: HTMLButtonElement) {
    const codeBlock = button.closest('.code-block')?.querySelector('.code-content');
    const language = button.closest('.code-block')?.querySelector('.code-language')?.textContent || 'code';
    if (codeBlock) {
      const blob = new Blob([codeBlock.textContent || ''], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${article.name.replace(/\.[^/.]+$/, '')}_${language}.${language}`;
      a.click();
      URL.revokeObjectURL(url);
    }
  }

  onMount(() => {
    processContent();
    
    // Add global functions for code buttons
    (window as any).copyCode = copyCode;
    (window as any).downloadCode = downloadCode;
  });
</script>

<div class="article-viewer">
  <div class="article-header">
    <div class="article-title">
      <FileText size={20} />
      <h2>{article.name}</h2>
    </div>
    <div class="article-actions">
      <button class="action-btn" on:click={openInEditor} title="Open in Editor">
        <Edit3 size={16} />
      </button>
      <button class="action-btn" on:click={onClose} title="Close">
        <X size={16} />
      </button>
    </div>
  </div>
  
  <div class="article-content">
    {@html processedContent}
  </div>
</div>

<style>
  .article-viewer {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    overflow: hidden;
  }

  .article-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .article-title {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .article-title h2 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .article-actions {
    display: flex;
    gap: 8px;
  }

  .action-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: transparent;
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.2s;
  }

  .action-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
    border-color: var(--accent-primary);
  }

  .article-content {
    flex: 1;
    padding: 24px;
    overflow-y: auto;
    line-height: 1.6;
    color: var(--text-primary);
  }

  .article-paragraph {
    margin-bottom: 16px;
    font-size: 14px;
    line-height: 1.7;
  }

  .article-header.h1 {
    font-size: 28px;
    font-weight: 700;
    margin: 24px 0 16px 0;
    color: var(--text-primary);
  }

  .article-header.h2 {
    font-size: 22px;
    font-weight: 600;
    margin: 20px 0 12px 0;
    color: var(--text-primary);
  }

  .article-header.h3 {
    font-size: 18px;
    font-weight: 600;
    margin: 16px 0 8px 0;
    color: var(--text-primary);
  }

  .article-list {
    margin: 16px 0;
    padding-left: 20px;
  }

  .article-list li {
    margin-bottom: 8px;
    font-size: 14px;
  }

  .inline-code {
    background: var(--bg-tertiary);
    color: var(--accent-primary);
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
    font-size: 13px;
  }

  .article-image {
    max-width: 100%;
    height: auto;
    border-radius: 8px;
    margin: 16px 0;
  }

  .article-content a {
    color: var(--accent-primary);
    text-decoration: none;
    border-bottom: 1px solid transparent;
    transition: border-color 0.2s;
  }

  .article-content a:hover {
    border-bottom-color: var(--accent-primary);
  }

  /* Code Blocks */
  .code-block {
    margin: 20px 0;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border-subtle);
  }

  .code-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 16px;
    background: var(--bg-tertiary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .code-language {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
  }

  .code-actions {
    display: flex;
    gap: 4px;
  }

  .code-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: transparent;
    border: none;
    border-radius: 4px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.2s;
  }

  .code-btn:hover {
    background: var(--bg-secondary);
    color: var(--text-primary);
  }

  .code-content {
    margin: 0;
    padding: 16px;
    background: var(--bg-secondary);
    font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
    font-size: 13px;
    line-height: 1.5;
    overflow-x: auto;
  }

  .code-content code {
    background: transparent;
    color: var(--text-primary);
  }

  /* Syntax highlighting with theme colors */
  :global(.hljs-keyword) { color: var(--accent-primary); font-weight: 600; }
  :global(.hljs-string) { color: var(--accent-success); }
  :global(.hljs-number) { color: var(--accent-warning); }
  :global(.hljs-comment) { color: var(--text-muted); font-style: italic; }
  :global(.hljs-function) { color: var(--accent-secondary); }
  :global(.hljs-variable) { color: var(--text-primary); }
  :global(.hljs-operator) { color: var(--accent-primary); }
  :global(.hljs-class) { color: var(--accent-secondary); font-weight: 600; }
  :global(.hljs-tag) { color: var(--accent-primary); }
  :global(.hljs-attribute) { color: var(--accent-success); }
  :global(.hljs-built_in) { color: var(--accent-secondary); }
</style>

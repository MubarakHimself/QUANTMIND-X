<script lang="ts">
  import { onMount } from 'svelte';

  // State
  let iframeLoaded = $state(false);
  let currentUrl = $state('https://forge.mql5.io/?lang=en');
  let showGuide = $state(false);

  // Navigation items
  const navItems = [
    { label: 'Home', url: 'https://forge.mql5.io/?lang=en', icon: '🏠' },
    { label: 'Guide', url: 'https://forge.mql5.io/guide', icon: '📖' },
    { label: 'Examples', url: 'https://forge.mql5.io/examples', icon: '📝' },
    { label: 'Documentation', url: 'https://www.mql5.com/en/docs', icon: '📚' }
  ];

  // Navigate to URL
  function navigate(url: string) {
    currentUrl = url;
    iframeLoaded = false;
  }

  // Refresh iframe
  function refresh() {
    iframeLoaded = false;
    // Force reload by appending timestamp
    const separator = currentUrl.includes('?') ? '&' : '?';
    currentUrl = currentUrl.split('&_t=')[0] + `${separator}_t=${Date.now()}`;
  }

  // Open in new tab
  function openExternal() {
    window.open(currentUrl, '_blank');
  }

  // Handle iframe load
  function onIframeLoad() {
    iframeLoaded = true;
  }

  // Toggle guide
  function toggleGuide() {
    showGuide = !showGuide;
  }
</script>

<div class="algoforge-panel">
  <!-- Header -->
  <div class="panel-header">
    <div class="header-left">
      <h3>AlgoForge</h3>
      <span class="badge">Web Preview</span>
    </div>
    <div class="header-actions">
      <button class="action-btn" onclick={refresh} title="Refresh">
        🔄
      </button>
      <button class="action-btn" onclick={openExternal} title="Open in new tab">
        🔗
      </button>
      <button class="action-btn" onclick={toggleGuide} title="Toggle guide">
        ℹ️
      </button>
    </div>
  </div>

  <!-- Navigation -->
  <div class="nav-bar">
    {#each navItems as item}
      <button 
        class="nav-btn"
        class:active={currentUrl === item.url}
        onclick={() => navigate(item.url)}
      >
        <span class="nav-icon">{item.icon}</span>
        <span class="nav-label">{item.label}</span>
      </button>
    {/each}
  </div>

  <!-- Guide Panel -->
  {#if showGuide}
    <div class="guide-panel">
      <h4>Using AlgoForge</h4>
      <ul>
        <li><strong>Home</strong> - Access the main AlgoForge interface for creating EAs</li>
        <li><strong>Guide</strong> - Learn how to use AlgoForge's features</li>
        <li><strong>Examples</strong> - Browse example strategies and code</li>
        <li><strong>Documentation</strong> - MQL5 language reference</li>
      </ul>
      <p class="note">
        Note: AlgoForge is MetaQuotes' official tool for creating Expert Advisors visually.
        Generated code can be used as a reference for your own strategies.
      </p>
    </div>
  {/if}

  <!-- Iframe Container -->
  <div class="iframe-container">
    {#if !iframeLoaded}
      <div class="loading-overlay">
        <div class="spinner"></div>
        <p>Loading AlgoForge...</p>
      </div>
    {/if}
    
    <iframe
      src={currentUrl}
      title="AlgoForge Web Preview"
      sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
      allow="clipboard-read; clipboard-write"
      onload={onIframeLoad}
    ></iframe>
  </div>

  <!-- Footer -->
  <div class="panel-footer">
    <span class="url-display">{currentUrl}</span>
    <span class="powered-by">Powered by MetaQuotes</span>
  </div>
</div>

<style>
  .algoforge-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary, #1e1e2e);
    border-radius: 8px;
    overflow: hidden;
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: var(--bg-secondary, #181825);
    border-bottom: 1px solid var(--border-color, #313244);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .panel-header h3 {
    margin: 0;
    font-size: 1rem;
    color: var(--text-primary, #cdd6f4);
  }

  .badge {
    font-size: 0.625rem;
    padding: 2px 6px;
    background: var(--accent, #89b4fa);
    color: var(--bg-primary, #1e1e2e);
    border-radius: 4px;
    font-weight: 600;
    text-transform: uppercase;
  }

  .header-actions {
    display: flex;
    gap: 8px;
  }

  .action-btn {
    background: transparent;
    border: none;
    padding: 6px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1rem;
    color: var(--text-secondary, #a6adc8);
  }

  .action-btn:hover {
    background: var(--bg-hover, #313244);
  }

  .nav-bar {
    display: flex;
    gap: 4px;
    padding: 8px 16px;
    background: var(--bg-secondary, #181825);
    border-bottom: 1px solid var(--border-color, #313244);
  }

  .nav-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    background: transparent;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    color: var(--text-secondary, #a6adc8);
    font-size: 0.875rem;
    transition: all 0.2s ease;
  }

  .nav-btn:hover {
    background: var(--bg-hover, #313244);
    color: var(--text-primary, #cdd6f4);
  }

  .nav-btn.active {
    background: var(--accent, #89b4fa);
    color: var(--bg-primary, #1e1e2e);
  }

  .nav-icon {
    font-size: 1rem;
  }

  .nav-label {
    font-weight: 500;
  }

  .guide-panel {
    padding: 16px;
    background: var(--bg-tertiary, #313244);
    border-bottom: 1px solid var(--border-color, #45475a);
  }

  .guide-panel h4 {
    margin: 0 0 12px;
    font-size: 0.875rem;
    color: var(--text-primary, #cdd6f4);
  }

  .guide-panel ul {
    margin: 0;
    padding-left: 20px;
    font-size: 0.875rem;
    color: var(--text-secondary, #a6adc8);
  }

  .guide-panel li {
    margin-bottom: 8px;
  }

  .guide-panel .note {
    margin: 12px 0 0;
    padding: 8px;
    background: rgba(137, 180, 250, 0.1);
    border-radius: 4px;
    font-size: 0.75rem;
    color: var(--accent, #89b4fa);
  }

  .iframe-container {
    flex: 1;
    position: relative;
    background: white;
  }

  .loading-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: var(--bg-primary, #1e1e2e);
    z-index: 10;
  }

  .spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--border-color, #45475a);
    border-top-color: var(--accent, #89b4fa);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 12px;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .loading-overlay p {
    color: var(--text-secondary, #a6adc8);
    font-size: 0.875rem;
  }

  iframe {
    width: 100%;
    height: 100%;
    border: none;
  }

  .panel-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 16px;
    background: var(--bg-secondary, #181825);
    border-top: 1px solid var(--border-color, #313244);
    font-size: 0.75rem;
  }

  .url-display {
    color: var(--text-secondary, #a6adc8);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 60%;
  }

  .powered-by {
    color: var(--text-muted, #6c7086);
  }
</style>

<script lang="ts">
  import { stopPropagation } from 'svelte/legacy';

  import { onMount } from "svelte";
  import { fade, fly } from "svelte/transition";

  // Types
  interface Feature {
    id: string;
    title: string;
    description: string;
    agent: string;
    icon: string;
    url?: string;
    created_at?: string;
  }

  interface FeaturesResponse {
    features: Feature[];
  }

  // State
  let features: Feature[] = $state([]);
  let loading = $state(true);
  let error: string | null = $state(null);
  let selectedFeature: Feature | null = $state(null);
  let viewMode: "cards" | "iframe" = $state("cards");

  // API Base URL for feature endpoints
  const FEATURE_API_BASE = "http://localhost:3002/api";

  // Fetch features from the API
  async function fetchFeatures(): Promise<void> {
    loading = true;
    error = null;

    try {
      const response = await fetch(`${FEATURE_API_BASE}/features`);

      if (!response.ok) {
        throw new Error(`Failed to fetch features: ${response.status} ${response.statusText}`);
      }

      const data: FeaturesResponse = await response.json();
      features = data.features || [];
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to fetch features";
      console.error("Error fetching features:", e);
    } finally {
      loading = false;
    }
  }

  // Refresh features
  async function refreshFeatures(): Promise<void> {
    await fetchFeatures();
  }

  // Open feature in new tab
  function openFeatureInNewTab(feature: Feature): void {
    if (feature.url) {
      window.open(feature.url, "_blank");
    } else {
      // Generate URL from feature ID if no explicit URL
      window.open(`${FEATURE_API_BASE}/features/${feature.id}`, "_blank");
    }
  }

  // Open feature in iframe
  function openFeatureInIframe(feature: Feature): void {
    selectedFeature = feature;
    viewMode = "iframe";
  }

  // Close iframe view
  function closeIframe(): void {
    selectedFeature = null;
    viewMode = "cards";
  }

  // Get agent badge color based on agent type
  function getAgentColor(agent: string): string {
    const colors: Record<string, string> = {
      analyst: "var(--accent-finance, #f9e2af)",
      quant: "var(--accent-primary, #89b4fa)",
      coder: "var(--accent-success, #a6e3a1)",
      researcher: "var(--accent-secondary, #cba6f7)",
      trader: "var(--accent-warning, #fab387)",
      default: "var(--text-secondary, #a6adc8)",
    };
    return colors[agent.toLowerCase()] || colors.default;
  }

  // Format date
  function formatDate(isoString: string | undefined): string {
    if (!isoString) return "";
    return new Date(isoString).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  }

  // Lifecycle
  onMount(() => {
    fetchFeatures();
  });
</script>

<div class="feature-catalog">
  <!-- Header -->
  <div class="catalog-header">
    <div class="header-content">
      <h2>Feature Catalog</h2>
      <p class="subtitle">Discover and launch AI-powered trading features</p>
    </div>
    <button
      class="refresh-btn"
      onclick={refreshFeatures}
      disabled={loading}
      title="Refresh features"
    >
      <span class="refresh-icon" class:spinning={loading}>&#8635;</span>
      Refresh
    </button>
  </div>

  <!-- Error Display -->
  {#if error}
    <div class="error-banner" in:fly={{ y: -20 }}>
      <span class="error-icon">&#9888;</span>
      <span>{error}</span>
      <button class="dismiss-btn" onclick={() => (error = null)}>&times;</button>
    </div>
  {/if}

  <!-- Iframe View -->
  {#if viewMode === "iframe" && selectedFeature}
    <div class="iframe-container" in:fade>
      <div class="iframe-header">
        <div class="iframe-title">
          <span class="feature-icon">{selectedFeature.icon}</span>
          <h3>{selectedFeature.title}</h3>
        </div>
        <div class="iframe-actions">
          <button
            class="action-btn"
            onclick={() => openFeatureInNewTab(selectedFeature)}
            title="Open in new tab"
          >
            &#8599; Open External
          </button>
          <button class="close-btn" onclick={closeIframe} title="Close">
            &times;
          </button>
        </div>
      </div>
      <div class="iframe-wrapper">
        <iframe
          src={selectedFeature.url || `${FEATURE_API_BASE}/features/${selectedFeature.id}`}
          title={selectedFeature.title}
          sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
        >
        </iframe>
      </div>
    </div>

  <!-- Cards View -->
  {:else}
    <!-- Loading State -->
    {#if loading}
      <div class="loading-state">
        <div class="spinner"></div>
        <p>Loading features...</p>
      </div>

    <!-- Empty State -->
    {:else if features.length === 0}
      <div class="empty-state" in:fade>
        <span class="empty-icon">&#128270;</span>
        <h3>No features yet</h3>
        <p>Features will appear here when they become available.</p>
        <p class="hint">Check back later or contact your administrator.</p>
      </div>

    <!-- Features Grid -->
    {:else}
      <div class="features-grid">
        {#each features as feature (feature.id)}
          <div
            class="feature-card"
            in:fly={{ y: 20 }}
            role="button"
            tabindex="0"
            onclick={() => openFeatureInNewTab(feature)}
            onkeydown={(e) => e.key === "Enter" && openFeatureInNewTab(feature)}
          >
            <div class="card-header">
              <span class="feature-icon">{feature.icon}</span>
              <span
                class="agent-badge"
                style="color: {getAgentColor(feature.agent)}; border-color: {getAgentColor(feature.agent)}"
              >
                {feature.agent}
              </span>
            </div>

            <div class="card-body">
              <h4 class="feature-title">{feature.title}</h4>
              <p class="feature-description">{feature.description}</p>
            </div>

            <div class="card-footer">
              {#if feature.created_at}
                <span class="feature-date">{formatDate(feature.created_at)}</span>
              {/if}
              <div class="card-actions">
                <button
                  class="action-icon"
                  onclick={stopPropagation(() => openFeatureInIframe(feature))}
                  title="Open in panel"
                >
                  &#9633;
                </button>
                <button
                  class="action-icon"
                  onclick={stopPropagation(() => openFeatureInNewTab(feature))}
                  title="Open in new tab"
                >
                  &#8599;
                </button>
              </div>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  {/if}
</div>

<style>
  .feature-catalog {
    padding: 24px;
    max-width: 1200px;
    margin: 0 auto;
    min-height: 100%;
  }

  /* Header */
  .catalog-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 24px;
    gap: 16px;
  }

  .header-content h2 {
    margin: 0;
    font-size: 1.5rem;
    color: var(--text-primary, #cdd6f4);
    font-weight: 600;
  }

  .subtitle {
    margin: 8px 0 0;
    color: var(--text-secondary, #a6adc8);
    font-size: 0.875rem;
  }

  .refresh-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--bg-tertiary, #313244);
    border: 1px solid var(--border-subtle, #45475a);
    border-radius: 8px;
    padding: 10px 16px;
    color: var(--text-primary, #cdd6f4);
    font-size: 0.875rem;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .refresh-btn:hover:not(:disabled) {
    background: var(--bg-secondary, #1e1e2e);
    border-color: var(--accent-primary, #89b4fa);
  }

  .refresh-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .refresh-icon {
    display: inline-block;
    font-size: 1.125rem;
    transition: transform 0.3s ease;
  }

  .refresh-icon.spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  /* Error Banner */
  .error-banner {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: rgba(243, 139, 168, 0.1);
    border: 1px solid var(--accent-danger, #f38ba8);
    border-radius: 8px;
    margin-bottom: 16px;
    color: var(--accent-danger, #f38ba8);
  }

  .error-icon {
    font-size: 1.125rem;
  }

  .dismiss-btn {
    margin-left: auto;
    background: transparent;
    border: none;
    color: inherit;
    font-size: 1.25rem;
    cursor: pointer;
    padding: 0 4px;
  }

  .dismiss-btn:hover {
    opacity: 0.8;
  }

  /* Loading State */
  .loading-state {
    text-align: center;
    padding: 64px 24px;
    color: var(--text-secondary, #a6adc8);
  }

  .spinner {
    width: 40px;
    height: 40px;
    border: 3px solid var(--border-subtle, #45475a);
    border-top-color: var(--accent-primary, #89b4fa);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto 16px;
  }

  /* Empty State */
  .empty-state {
    text-align: center;
    padding: 64px 24px;
    color: var(--text-secondary, #a6adc8);
  }

  .empty-icon {
    font-size: 4rem;
    display: block;
    margin-bottom: 16px;
    opacity: 0.6;
  }

  .empty-state h3 {
    margin: 0 0 8px;
    font-size: 1.25rem;
    color: var(--text-primary, #cdd6f4);
  }

  .hint {
    font-size: 0.75rem;
    margin-top: 8px;
    opacity: 0.7;
  }

  /* Features Grid */
  .features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 16px;
  }

  /* Feature Card */
  .feature-card {
    background: var(--bg-secondary, #1e1e2e);
    border: 1px solid var(--border-subtle, #313244);
    border-radius: 12px;
    padding: 20px;
    cursor: pointer;
    transition: all 0.2s ease;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .feature-card:hover {
    border-color: var(--accent-primary, #89b4fa);
    background: var(--bg-tertiary, #181825);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
  }

  .feature-card:focus {
    outline: 2px solid var(--accent-primary, #89b4fa);
    outline-offset: 2px;
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
  }

  .feature-icon {
    font-size: 2rem;
    line-height: 1;
  }

  .agent-badge {
    font-size: 0.625rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 4px 8px;
    border-radius: 4px;
    border: 1px solid;
    font-weight: 600;
  }

  .card-body {
    flex: 1;
  }

  .feature-title {
    margin: 0 0 8px;
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary, #cdd6f4);
  }

  .feature-description {
    margin: 0;
    font-size: 0.8125rem;
    color: var(--text-secondary, #a6adc8);
    line-height: 1.5;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .card-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: 12px;
    border-top: 1px solid var(--border-subtle, #313244);
  }

  .feature-date {
    font-size: 0.75rem;
    color: var(--text-muted, #6c7086);
  }

  .card-actions {
    display: flex;
    gap: 4px;
  }

  .action-icon {
    background: transparent;
    border: none;
    color: var(--text-secondary, #a6adc8);
    cursor: pointer;
    padding: 6px 8px;
    border-radius: 4px;
    font-size: 1rem;
    transition: all 0.2s ease;
  }

  .action-icon:hover {
    background: rgba(137, 180, 250, 0.1);
    color: var(--accent-primary, #89b4fa);
  }

  /* Iframe Container */
  .iframe-container {
    display: flex;
    flex-direction: column;
    background: var(--bg-secondary, #1e1e2e);
    border: 1px solid var(--border-subtle, #313244);
    border-radius: 12px;
    overflow: hidden;
    height: calc(100vh - 150px);
    min-height: 500px;
  }

  .iframe-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: var(--bg-tertiary, #313244);
    border-bottom: 1px solid var(--border-subtle, #45475a);
  }

  .iframe-title {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .iframe-title .feature-icon {
    font-size: 1.5rem;
  }

  .iframe-title h3 {
    margin: 0;
    font-size: 1rem;
    color: var(--text-primary, #cdd6f4);
  }

  .iframe-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .action-btn {
    background: var(--bg-secondary, #1e1e2e);
    border: 1px solid var(--border-subtle, #45475a);
    border-radius: 6px;
    padding: 6px 12px;
    color: var(--text-secondary, #a6adc8);
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .action-btn:hover {
    border-color: var(--accent-primary, #89b4fa);
    color: var(--text-primary, #cdd6f4);
  }

  .close-btn {
    background: transparent;
    border: none;
    color: var(--text-secondary, #a6adc8);
    font-size: 1.5rem;
    cursor: pointer;
    padding: 4px 8px;
    border-radius: 4px;
  }

  .close-btn:hover {
    background: rgba(243, 139, 168, 0.1);
    color: var(--accent-danger, #f38ba8);
  }

  .iframe-wrapper {
    flex: 1;
    position: relative;
  }

  .iframe-wrapper iframe {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    border: none;
    background: var(--bg-primary, #11111b);
  }

  /* Responsive */
  @media (max-width: 768px) {
    .feature-catalog {
      padding: 16px;
    }

    .catalog-header {
      flex-direction: column;
      align-items: flex-start;
    }

    .features-grid {
      grid-template-columns: 1fr;
    }

    .iframe-container {
      height: calc(100vh - 200px);
    }
  }
</style>

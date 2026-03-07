<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import {
    Folder,
    FolderOpen,
    Newspaper,
    Library,
    FileText,
    ChevronRight,
  } from "lucide-svelte";

  const dispatch = createEventDispatcher();

  export let articles: Array<any> = [];
  export let filteredArticles: Array<any> = [];
  export let currentFolder: string | null = null;
  export let selectedCategory = "all";
  export let categoryTree: Record<string, { count: number; label: string }> = {};
  export let categories: Array<{ id: string; name: string; count: number }> = [];

  function getCategoryLabel(key: string): string {
    return categoryTree[key]?.label || key.replace(/_/g, " ").toUpperCase();
  }

  function openArticleViewer(article: any) {
    dispatch("openArticleViewer", article);
  }

  function openInEditor(item: any) {
    dispatch("openInEditor", item);
  }

  function navigateTo(folderId: string, folderName: string) {
    dispatch("navigateTo", { folderId, folderName });
  }
</script>

<div class="knowledge-view">
  <!-- Knowledge Hub with Category Sidebar -->
  {#if !currentFolder}
    <div class="knowledge-hub-layout">
      <!-- Category Sidebar -->
      <div class="category-sidebar">
        <h3>Categories</h3>
        <button
          class="category-btn"
          class:active={selectedCategory === "all"}
          on:click={() => (selectedCategory = "all")}
        >
          <Folder size={16} />
          <span>All Articles</span>
          <span class="count">{articles.length}</span>
        </button>

        {#each Object.entries(categoryTree) as [key, cat]}
          <button
            class="category-btn"
            class:active={selectedCategory === key}
            on:click={() => (selectedCategory = key)}
          >
            <FolderOpen size={16} />
            <span>{cat.label}</span>
            <span class="count">{cat.count}</span>
          </button>
        {/each}
      </div>

      <!-- Articles List -->
      <div class="articles-main">
        <div class="articles-header">
          <h3>
            {selectedCategory === "all"
              ? "All Articles"
              : getCategoryLabel(selectedCategory)}
          </h3>
          <span class="article-count"
            >{filteredArticles.length} articles</span
          >
        </div>

        <div class="articles-grid">
          {#if filteredArticles.length === 0}
            <div class="empty-state">
              <Newspaper size={48} />
              <p>No articles found in this category</p>
              <span class="hint"
                >Try selecting a different category or check if the backend is
                running</span
              >
            </div>
          {:else}
            {#each filteredArticles as article}
              <div class="article-card">
                <div class="article-icon">
                  <Newspaper size={20} />
                </div>
                <div class="article-info">
                  <h4>{article.name}</h4>
                  <div class="article-meta">
                    <span
                      class="category-tag"
                      class:expert-advisors={article.category?.includes(
                        "expert_advisors",
                      )}
                      class:integration={article.category?.includes(
                        "integration",
                      )}
                      class:trading={article.category?.includes("trading")}
                      class:trading-systems={article.category?.includes(
                        "trading-systems",
                      )}
                    >
                      {article.category
                        ?.split("/")[1]
                        ?.replace("_", " ")
                        .toUpperCase() || "GENERAL"}
                    </span>
                  </div>
                </div>
                <div class="article-actions">
                  <button
                    class="btn-icon"
                    on:click={() => openArticleViewer(article)}
                    title="View Article"
                  >
                    <FileText size={14} />
                  </button>
                </div>
              </div>
            {/each}
          {/if}
        </div>
      </div>
    </div>
  {:else}
    <!-- Legacy Categories View -->
    <div class="category-cards">
      {#each categories as cat}
        <div
          class="category-card"
          on:click={() => navigateTo(cat.id, cat.name)}
          role="button"
          tabindex="0"
          on:keydown={(e) =>
            e.key === "Enter" && navigateTo(cat.id, cat.name)}
        >
          <Folder size={40} />
          <span class="cat-name">{cat.name}</span>
          <span class="cat-count">{cat.count} items</span>
        </div>
      {/each}
    </div>

    <div class="recent-section">
      <h3>Recent Articles</h3>
      <div class="recent-articles">
        {#each articles.slice(0, 5) as article}
          <div
            class="recent-article-card"
            on:click={() => openArticleViewer(article)}
            role="button"
            tabindex="0"
            on:keydown={(e) =>
              e.key === "Enter" && openArticleViewer(article)}
          >
            <div class="recent-article-icon">
              {#if article.type === "book"}
                <Library size={18} />
              {:else if article.type === "note"}
                <FileText size={18} />
              {:else}
                <Newspaper size={18} />
              {/if}
            </div>
            <div class="recent-article-info">
              <h6>{article.title || article.name}</h6>
              <div class="recent-article-meta">
                {#if article.author}<span>{article.author}</span>{/if}
                {#if article.date}<span
                    >{new Date(
                      article.date,
                    ).toLocaleDateString()}</span
                  >{/if}
              </div>
            </div>
            <ChevronRight size={14} class="recent-article-arrow" />
          </div>
        {/each}
      </div>
    </div>
  {/if}
</div>

<style>
  .knowledge-view {
    height: 100%;
    overflow-y: auto;
  }

  .knowledge-hub-layout {
    display: grid;
    grid-template-columns: 240px 1fr;
    gap: 24px;
    height: 100%;
    padding: 20px;
  }

  .category-sidebar {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding-right: 16px;
    border-right: 1px solid var(--border-subtle, #e5e7eb);
  }

  .category-sidebar h3 {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-secondary, #6b7280);
    margin: 0 0 12px 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .category-btn {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--text-secondary, #6b7280);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s ease;
    text-align: left;
    width: 100%;
  }

  .category-btn:hover {
    background: var(--bg-tertiary, #f3f4f6);
    color: var(--text-primary, #111827);
  }

  .category-btn.active {
    background: var(--accent-primary, #3b82f6);
    color: white;
  }

  .category-btn .count {
    margin-left: auto;
    font-size: 11px;
    background: var(--bg-tertiary, #e5e7eb);
    padding: 2px 6px;
    border-radius: 10px;
  }

  .category-btn.active .count {
    background: rgba(255, 255, 255, 0.2);
    color: white;
  }

  .articles-main {
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .articles-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
  }

  .articles-header h3 {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary, #111827);
    margin: 0;
  }

  .article-count {
    font-size: 13px;
    color: var(--text-muted, #9ca3af);
  }

  .articles-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
    overflow-y: auto;
    flex: 1;
  }

  .empty-state {
    grid-column: 1 / -1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 20px;
    color: var(--text-muted, #9ca3af);
    text-align: center;
  }

  .empty-state p {
    margin: 16px 0 8px;
    font-size: 15px;
    color: var(--text-secondary, #6b7280);
  }

  .empty-state .hint {
    font-size: 13px;
    color: var(--text-muted, #9ca3af);
  }

  .article-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px;
    background: var(--bg-secondary, #f9fafb);
    border: 1px solid var(--border-subtle, #e5e7eb);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .article-card:hover {
    background: var(--bg-tertiary, #f3f4f6);
    border-color: var(--accent-primary, #3b82f6);
    transform: translateY(-1px);
  }

  .article-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    background: var(--bg-tertiary, #e5e7eb);
    border-radius: 8px;
    color: var(--accent-primary, #3b82f6);
  }

  .article-info {
    flex: 1;
    min-width: 0;
  }

  .article-info h4 {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary, #111827);
    margin: 0 0 6px 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .article-meta {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .category-tag {
    font-size: 10px;
    font-weight: 500;
    padding: 2px 6px;
    border-radius: 4px;
    background: var(--bg-tertiary, #e5e7eb);
    color: var(--text-secondary, #6b7280);
    text-transform: uppercase;
  }

  .category-tag.expert-advisors {
    background: rgba(16, 185, 129, 0.1);
    color: #10b981;
  }

  .category-tag.integration {
    background: rgba(59, 130, 246, 0.1);
    color: #3b82f6;
  }

  .category-tag.trading {
    background: rgba(245, 158, 11, 0.1);
    color: #f59e0b;
  }

  .category-tag.trading-systems {
    background: rgba(139, 92, 246, 0.1);
    color: #8b5cf6;
  }

  .article-actions {
    display: flex;
    gap: 4px;
  }

  .btn-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: transparent;
    border: 1px solid var(--border-subtle, #e5e7eb);
    border-radius: 6px;
    color: var(--text-muted, #9ca3af);
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .btn-icon:hover {
    background: var(--accent-primary, #3b82f6);
    border-color: var(--accent-primary, #3b82f6);
    color: white;
  }

  /* Legacy Category Cards */
  .category-cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 20px;
    padding: 20px;
  }

  .category-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    padding: 32px 20px;
    background: var(--bg-secondary, #f9fafb);
    border: 1px solid var(--border-subtle, #e5e7eb);
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .category-card:hover {
    background: var(--bg-tertiary, #f3f4f6);
    border-color: var(--accent-primary, #3b82f6);
    transform: translateY(-2px);
  }

  .cat-name {
    font-size: 15px;
    font-weight: 500;
    color: var(--text-primary, #111827);
  }

  .cat-count {
    font-size: 12px;
    color: var(--text-muted, #9ca3af);
  }

  .recent-section {
    padding: 20px;
    border-top: 1px solid var(--border-subtle, #e5e7eb);
  }

  .recent-section h3 {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary, #111827);
    margin: 0 0 16px 0;
  }

  .recent-articles {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .recent-article-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: var(--bg-secondary, #f9fafb);
    border: 1px solid var(--border-subtle, #e5e7eb);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .recent-article-card:hover {
    background: var(--bg-tertiary, #f3f4f6);
  }

  .recent-article-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    background: var(--bg-tertiary, #e5e7eb);
    border-radius: 6px;
    color: var(--text-secondary, #6b7280);
  }

  .recent-article-info {
    flex: 1;
    min-width: 0;
  }

  .recent-article-info h6 {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary, #111827);
    margin: 0 0 4px 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .recent-article-meta {
    display: flex;
    gap: 12px;
    font-size: 11px;
    color: var(--text-muted, #9ca3af);
  }

  .recent-article-arrow {
    color: var(--text-muted, #9ca3af);
  }
</style>

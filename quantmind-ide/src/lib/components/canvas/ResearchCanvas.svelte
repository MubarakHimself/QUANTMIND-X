<script lang="ts">
  /**
   * Research Canvas
   *
   * Research Department Head workspace.
   * Tab-navigated content grid (Articles, Books, Logs, Personal, YouTube, Dept Tasks)
   *
   * Frosted Terminal aesthetic — amber (#f0a500) dept accent, cyan (#00d4ff) highlights.
   * Svelte 5 runes only.
   */
  import { onMount, onDestroy } from 'svelte';
  import {
    BookOpen,
    FileText,
    ScrollText,
    User,
    Youtube,
    Kanban,
    Newspaper,
    Tag,
    Calendar,
    Clock,
    ExternalLink,
    Loader2,
    Inbox,
    FlaskConical,
    Activity,
    Lightbulb,
    ChevronDown
  } from 'lucide-svelte';
  import { apiFetch } from '$lib/api';
  import DepartmentKanban from '$lib/components/department-kanban/DepartmentKanban.svelte';
  import AgentTilePanel from '$lib/components/AgentTilePanel.svelte';
  import NewsView from '$lib/components/research/NewsView.svelte';
  import { canvasContextService } from '$lib/services/canvasContextService';
  import { pipelineRuns, alphaForgeStore } from '$lib/stores/alpha-forge';
  import type { PipelineRun } from '$lib/stores/alpha-forge';

  // =============================================================================
  // Types
  // =============================================================================

  type ResearchTab = 'articles' | 'books' | 'logs' | 'personal' | 'youtube' | 'news' | 'dept-tasks';

  interface ArticleItem {
    id: string;
    title: string;
    source: string;
    date: string;
    tags: string[];
    excerpt?: string;
  }

  interface BookItem {
    id: string;
    title: string;
    author: string;
    topics: string[];
    year?: number;
  }

  interface LogItem {
    id: string;
    title: string;
    date: string;
    content: string;
    tags: string[];
  }

  interface PersonalItem {
    id: string;
    title: string;
    date: string;
    preview: string;
  }

  interface VideoItem {
    id: string;
    title: string;
    channel: string;
    duration: string;
    thumbnail?: string;
    date?: string;
  }

  // =============================================================================
  // State
  // =============================================================================

  let activeTab = $state<ResearchTab>('articles');

  // Pipeline status bar
  let pipelineRunsData = $state<PipelineRun[]>([]);
  const unsubPipeline = pipelineRuns.subscribe(v => { pipelineRunsData = v; });

  let researchCount = $derived(pipelineRunsData.filter(r => r.current_stage === 'RESEARCH').length);
  let trdCount = $derived(pipelineRunsData.filter(r => r.current_stage === 'TRD').length);
  let researchActiveCount = $derived(pipelineRunsData.filter(r => r.current_stage === 'RESEARCH' && r.stage_status === 'running').length);
  let trdActiveCount = $derived(pipelineRunsData.filter(r => r.current_stage === 'TRD' && r.stage_status === 'running').length);
  let hasPipelineActivity = $derived(researchCount > 0 || trdCount > 0);

  // Content data per tab
  let articles = $state<ArticleItem[]>([]);
  let books = $state<BookItem[]>([]);
  let logs = $state<LogItem[]>([]);
  let personal = $state<PersonalItem[]>([]);
  let videos = $state<VideoItem[]>([]);

  // Loading states per tab
  let loadingTab = $state<ResearchTab | null>(null);

  // Detail sub-page
  type DetailItem = ArticleItem | BookItem | LogItem | PersonalItem | VideoItem | null;
  let selectedItem = $state<DetailItem>(null);
  let selectedItemType = $state<ResearchTab | null>(null);

  // Agent insights strip
  let insightsExpanded = $state(false);
  let insightsUnread = $state(0);

  // =============================================================================
  // Tab config
  // =============================================================================

  const tabs: { id: ResearchTab; label: string; icon: typeof BookOpen }[] = [
    { id: 'articles',   label: 'Articles',    icon: FileText      },
    { id: 'books',      label: 'Books',       icon: BookOpen      },
    { id: 'logs',       label: 'Logs',        icon: ScrollText    },
    { id: 'personal',  label: 'Personal',    icon: User          },
    { id: 'youtube',   label: 'YouTube',     icon: Youtube       },
    { id: 'news',      label: 'News',        icon: Newspaper     },
    { id: 'dept-tasks', label: 'Dept Tasks',  icon: Kanban  },
  ];

  // =============================================================================
  // Lifecycle
  // =============================================================================

  onMount(async () => {
    try {
      await canvasContextService.loadCanvasContext('research');
    } catch {
      // silently ignore — canvas context is optional
    }
    await loadTab('articles');
    // Fetch pipeline status for the status bar (silent — non-critical)
    alphaForgeStore.fetchPipelineStatus().catch(() => {});
  });

  onDestroy(() => {
    unsubPipeline();
  });

  // =============================================================================
  // Data loading
  // =============================================================================

  async function loadTab(tab: ResearchTab) {
    if (tab === 'dept-tasks' || tab === 'news') return;
    if (loadingTab === tab) return;

    loadingTab = tab;
    try {
      switch (tab) {
        case 'articles': {
          const data = await apiFetch<ArticleItem[]>('/knowledge/articles');
          articles = data;
          break;
        }
        case 'books': {
          const data = await apiFetch<BookItem[]>('/knowledge/books');
          books = data;
          break;
        }
        case 'logs': {
          const data = await apiFetch<LogItem[]>('/knowledge/logs');
          logs = data;
          break;
        }
        case 'personal': {
          const data = await apiFetch<PersonalItem[]>('/knowledge/personal');
          personal = data;
          break;
        }
        case 'youtube': {
          const data = await apiFetch<VideoItem[]>('/knowledge/videos');
          videos = data;
          break;
        }
      }
    } catch {
      // empty state — backend not available is non-error
    } finally {
      loadingTab = null;
    }
  }

  async function handleTabChange(tab: ResearchTab) {
    activeTab = tab;
    selectedItem = null;
    selectedItemType = null;
    await loadTab(tab);
  }

  // =============================================================================
  // Detail navigation
  // =============================================================================

  function openDetail(item: DetailItem, type: ResearchTab) {
    selectedItem = item;
    selectedItemType = type;
  }

  function closeDetail() {
    selectedItem = null;
    selectedItemType = null;
  }

  // =============================================================================
  // Helpers
  // =============================================================================

  function isArticle(item: DetailItem): item is ArticleItem {
    return !!item && 'source' in item && 'tags' in item && !('author' in item) && !('channel' in item);
  }
  function isBook(item: DetailItem): item is BookItem {
    return !!item && 'author' in item && 'topics' in item;
  }
  function isLog(item: DetailItem): item is LogItem {
    return !!item && 'content' in item && !('source' in item);
  }
  function isPersonal(item: DetailItem): item is PersonalItem {
    return !!item && 'preview' in item;
  }
  function isVideo(item: DetailItem): item is VideoItem {
    return !!item && 'channel' in item && 'duration' in item;
  }
</script>

<div class="research-canvas" data-dept="research">

  <!-- =========================================================
       Canvas Header
       ========================================================= -->
  <header class="canvas-header">
    {#if selectedItem}
      <button class="back-btn" onclick={closeDetail}>
        <ExternalLink size={13} />
        <span>Back</span>
      </button>
    {/if}
    <div class="header-left">
      <FlaskConical size={18} class="dept-icon" />
      <h1 class="canvas-title">Research</h1>
      <span class="dept-badge">Research Dept</span>
    </div>

    <!-- Tab navigation -->
    {#if !selectedItem}
      <nav class="tab-nav">
        {#each tabs as tab}
          <button
            class="tab-btn"
            class:active={activeTab === tab.id}
            onclick={() => handleTabChange(tab.id)}
          >
            <svelte:component this={tab.icon} size={13} />
            <span>{tab.label}</span>
            {#if loadingTab === tab.id}
              <Loader2 size={11} class="spin" />
            {/if}
          </button>
        {/each}
      </nav>
    {/if}
  </header>

  <!-- =========================================================
       Pipeline status bar (Research & TRD stages only)
       ========================================================= -->
  {#if hasPipelineActivity}
    <div class="pipeline-bar">
      <Activity size={11} class="pipeline-bar-icon" />
      <span class="pipeline-bar-label">AlphaForge</span>
      <div class="pipeline-stages">
        <span class="stage-chip" class:stage-active={researchActiveCount > 0}>
          {#if researchActiveCount > 0}<span class="stage-pulse"></span>{/if}
          RESEARCH
          <span class="stage-count">{researchCount}</span>
        </span>
        <span class="stage-arrow">→</span>
        <span class="stage-chip" class:stage-active={trdActiveCount > 0}>
          {#if trdActiveCount > 0}<span class="stage-pulse"></span>{/if}
          TRD
          <span class="stage-count">{trdCount}</span>
        </span>
      </div>
    </div>
  {/if}

  <!-- =========================================================
       Main content area
       ========================================================= -->
  <div class="canvas-body">

    {#if selectedItem && selectedItemType}
      <!-- ---- Detail sub-page ---- -->
      <div class="detail-page">
        <div class="detail-header">
          <button class="detail-back" onclick={closeDetail}>
            <ExternalLink size={13} />
            Back to {selectedItemType}
          </button>
        </div>

        {#if isArticle(selectedItem)}
          <div class="detail-card">
            <h2 class="detail-title">{selectedItem.title}</h2>
            <div class="detail-meta">
              <span class="meta-chip source-chip">{selectedItem.source}</span>
              <span class="meta-chip">
                <Calendar size={11} />
                {selectedItem.date}
              </span>
            </div>
            <div class="detail-tags">
              {#each selectedItem.tags as tag}
                <span class="tag-chip"><Tag size={10} />{tag}</span>
              {/each}
            </div>
            {#if selectedItem.excerpt}
              <p class="detail-excerpt">{selectedItem.excerpt}</p>
            {/if}
          </div>

        {:else if isBook(selectedItem)}
          <div class="detail-card">
            <h2 class="detail-title">{selectedItem.title}</h2>
            <div class="detail-meta">
              <span class="meta-chip">{selectedItem.author}</span>
              {#if selectedItem.year}
                <span class="meta-chip">{selectedItem.year}</span>
              {/if}
            </div>
            <div class="detail-tags">
              {#each selectedItem.topics as topic}
                <span class="tag-chip"><Tag size={10} />{topic}</span>
              {/each}
            </div>
          </div>

        {:else if isLog(selectedItem)}
          <div class="detail-card">
            <h2 class="detail-title">{selectedItem.title}</h2>
            <div class="detail-meta">
              <span class="meta-chip">
                <Calendar size={11} />
                {selectedItem.date}
              </span>
            </div>
            <div class="detail-tags">
              {#each selectedItem.tags as tag}
                <span class="tag-chip"><Tag size={10} />{tag}</span>
              {/each}
            </div>
            <p class="detail-excerpt">{selectedItem.content}</p>
          </div>

        {:else if isPersonal(selectedItem)}
          <div class="detail-card">
            <h2 class="detail-title">{selectedItem.title}</h2>
            <div class="detail-meta">
              <span class="meta-chip">
                <Calendar size={11} />
                {selectedItem.date}
              </span>
            </div>
            <p class="detail-excerpt">{selectedItem.preview}</p>
          </div>

        {:else if isVideo(selectedItem)}
          <div class="detail-card">
            <div class="video-thumb-placeholder">
              <Youtube size={32} />
            </div>
            <h2 class="detail-title">{selectedItem.title}</h2>
            <div class="detail-meta">
              <span class="meta-chip">{selectedItem.channel}</span>
              <span class="meta-chip">
                <Clock size={11} />
                {selectedItem.duration}
              </span>
              {#if selectedItem.date}
                <span class="meta-chip">
                  <Calendar size={11} />
                  {selectedItem.date}
                </span>
              {/if}
            </div>
          </div>
        {/if}
      </div>

    {:else if activeTab === 'dept-tasks'}
      <!-- ---- Dept Tasks: inline kanban ---- -->
      <div class="kanban-wrapper">
        <DepartmentKanban department="research" />
      </div>

    {:else if activeTab === 'news'}
      <div class="news-wrapper">
        <NewsView />
      </div>

    {:else}
      <!-- ---- Tile grid ---- -->
      <div class="tile-grid-wrapper">

        {#if loadingTab === activeTab}
          <div class="empty-state">
            <Loader2 size={28} class="spin" />
            <span>Loading {activeTab}…</span>
          </div>

        {:else if activeTab === 'articles'}
          {#if articles.length === 0}
            <div class="empty-state">
              <Inbox size={28} />
              <span>No articles indexed yet</span>
            </div>
          {:else}
            <div class="tile-grid">
              {#each articles as article}
                <button class="tile article-tile" onclick={() => openDetail(article, 'articles')}>
                  <div class="tile-top">
                    <FileText size={14} class="tile-icon" />
                    <span class="tile-source">{article.source}</span>
                  </div>
                  <h3 class="tile-title">{article.title}</h3>
                  <div class="tile-meta">
                    <span class="meta-date">
                      <Calendar size={10} />
                      {article.date}
                    </span>
                  </div>
                  {#if article.tags?.length}
                    <div class="tile-tags">
                      {#each article.tags.slice(0, 3) as tag}
                        <span class="tag-chip small">{tag}</span>
                      {/each}
                    </div>
                  {/if}
                </button>
              {/each}
            </div>
          {/if}

        {:else if activeTab === 'books'}
          {#if books.length === 0}
            <div class="empty-state">
              <Inbox size={28} />
              <span>No books indexed yet</span>
            </div>
          {:else}
            <div class="tile-grid">
              {#each books as book}
                <button class="tile book-tile" onclick={() => openDetail(book, 'books')}>
                  <div class="tile-top">
                    <BookOpen size={14} class="tile-icon" />
                    <span class="tile-source">{book.author}</span>
                  </div>
                  <h3 class="tile-title">{book.title}</h3>
                  {#if book.year}
                    <span class="tile-year">{book.year}</span>
                  {/if}
                  {#if book.topics?.length}
                    <div class="tile-tags">
                      {#each book.topics.slice(0, 3) as topic}
                        <span class="tag-chip small">{topic}</span>
                      {/each}
                    </div>
                  {/if}
                </button>
              {/each}
            </div>
          {/if}

        {:else if activeTab === 'logs'}
          {#if logs.length === 0}
            <div class="empty-state">
              <Inbox size={28} />
              <span>No research logs yet</span>
            </div>
          {:else}
            <div class="tile-grid">
              {#each logs as log}
                <button class="tile log-tile" onclick={() => openDetail(log, 'logs')}>
                  <div class="tile-top">
                    <ScrollText size={14} class="tile-icon" />
                    <span class="meta-date">
                      <Calendar size={10} />
                      {log.date}
                    </span>
                  </div>
                  <h3 class="tile-title">{log.title}</h3>
                  {#if log.content}
                    <p class="tile-excerpt">{log.content.slice(0, 100)}{log.content.length > 100 ? '…' : ''}</p>
                  {/if}
                </button>
              {/each}
            </div>
          {/if}

        {:else if activeTab === 'personal'}
          {#if personal.length === 0}
            <div class="empty-state">
              <Inbox size={28} />
              <span>No personal notes yet</span>
            </div>
          {:else}
            <div class="tile-grid">
              {#each personal as note}
                <button class="tile personal-tile" onclick={() => openDetail(note, 'personal')}>
                  <div class="tile-top">
                    <User size={14} class="tile-icon" />
                    <span class="meta-date">
                      <Calendar size={10} />
                      {note.date}
                    </span>
                  </div>
                  <h3 class="tile-title">{note.title}</h3>
                  {#if note.preview}
                    <p class="tile-excerpt">{note.preview.slice(0, 100)}{note.preview.length > 100 ? '…' : ''}</p>
                  {/if}
                </button>
              {/each}
            </div>
          {/if}

        {:else if activeTab === 'youtube'}
          {#if videos.length === 0}
            <div class="empty-state">
              <Inbox size={28} />
              <span>No videos indexed yet</span>
            </div>
          {:else}
            <div class="tile-grid video-grid">
              {#each videos as video}
                <button class="tile video-tile" onclick={() => openDetail(video, 'youtube')}>
                  <div class="video-thumb">
                    {#if video.thumbnail}
                      <img src={video.thumbnail} alt={video.title} />
                    {:else}
                      <div class="thumb-placeholder">
                        <Youtube size={22} />
                      </div>
                    {/if}
                    <span class="duration-badge">{video.duration}</span>
                  </div>
                  <div class="video-info">
                    <h3 class="tile-title">{video.title}</h3>
                    <span class="tile-source">{video.channel}</span>
                    {#if video.date}
                      <span class="meta-date">
                        <Calendar size={10} />
                        {video.date}
                      </span>
                    {/if}
                  </div>
                </button>
              {/each}
            </div>
          {/if}
        {/if}

      </div>
    {/if}

  </div>

  <!-- =========================================================
       Agent Insights strip (always visible, collapsible)
       ========================================================= -->
  <div class="agent-insights-strip" class:expanded={insightsExpanded}>
    <button class="insights-toggle" onclick={() => { insightsExpanded = !insightsExpanded; }}>
      <Lightbulb size={12} />
      <span>Agent Insights</span>
      {#if insightsUnread > 0}
        <span class="unread-badge">{insightsUnread}</span>
      {/if}
      <span class:rotated={insightsExpanded} style="display:inline-flex;align-items:center;transition:transform 0.2s;"><ChevronDown size={12} /></span>
    </button>
    {#if insightsExpanded}
      <AgentTilePanel
        canvas="research"
        maxHeight="200px"
        showHeader={false}
        onUnreadCount={(n: number) => { insightsUnread = n; }}
      />
    {/if}
  </div>

</div>

<style>
  /* =============================================================================
     Shell layout
     ============================================================================= */

  .research-canvas {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: rgba(10, 15, 26, 0.92);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    color: #e0e0e0;
    font-family: 'JetBrains Mono', monospace;
    overflow: hidden;
  }

  /* =============================================================================
     Canvas header
     ============================================================================= */

  .canvas-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 20px 10px;
    border-bottom: 1px solid rgba(240, 165, 0, 0.15);
    flex-shrink: 0;
    flex-wrap: wrap;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
  }

  :global(.dept-icon) {
    color: #f0a500;
  }

  .canvas-title {
    font-size: 18px;
    font-weight: 700;
    color: #f0a500;
    margin: 0;
    letter-spacing: 0.02em;
  }

  .dept-badge {
    padding: 2px 8px;
    background: rgba(240, 165, 0, 0.12);
    border: 1px solid rgba(240, 165, 0, 0.3);
    border-radius: 4px;
    font-size: 10px;
    color: #f0a500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .back-btn {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 5px 10px;
    background: rgba(0, 212, 255, 0.08);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 5px;
    color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
    flex-shrink: 0;
  }

  .back-btn:hover {
    background: rgba(0, 212, 255, 0.15);
    border-color: rgba(0, 212, 255, 0.4);
  }

  /* =============================================================================
     Tab navigation
     ============================================================================= */

  .tab-nav {
    display: flex;
    gap: 4px;
    align-items: center;
    flex-wrap: wrap;
    margin-left: auto;
  }

  .tab-btn {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 5px 11px;
    background: rgba(240, 165, 0, 0.06);
    border: 1px solid rgba(240, 165, 0, 0.15);
    border-radius: 5px;
    color: rgba(224, 224, 224, 0.6);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }

  .tab-btn:hover {
    background: rgba(240, 165, 0, 0.12);
    border-color: rgba(240, 165, 0, 0.3);
    color: #e0e0e0;
  }

  .tab-btn.active {
    background: rgba(240, 165, 0, 0.18);
    border-color: rgba(240, 165, 0, 0.5);
    color: #f0a500;
  }

  /* =============================================================================
     Canvas body (scrollable content area)
     ============================================================================= */

  .canvas-body {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    transition: padding-bottom 0.25s ease;
    min-height: 0;
  }

  /* When kanban-wrapper is present, remove padding so kanban fills edge-to-edge */
  .canvas-body:has(.kanban-wrapper) {
    padding: 0;
    overflow: hidden;
  }

  /* =============================================================================
     Tile grid
     ============================================================================= */

  .tile-grid-wrapper {
    width: 100%;
  }

  .tile-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 14px;
  }

  .video-grid {
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  }

  /* Base tile */
  .tile {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 14px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(240, 165, 0, 0.12);
    border-radius: 8px;
    cursor: pointer;
    text-align: left;
    font-family: 'JetBrains Mono', monospace;
    color: #e0e0e0;
    transition: background 0.15s, border-color 0.15s, transform 0.1s;
  }

  .tile:hover {
    background: rgba(240, 165, 0, 0.08);
    border-color: rgba(240, 165, 0, 0.3);
    transform: translateY(-1px);
  }

  .tile-top {
    display: flex;
    align-items: center;
    gap: 6px;
    justify-content: space-between;
  }

  :global(.tile-icon) {
    color: #f0a500;
    flex-shrink: 0;
  }

  .tile-title {
    font-size: 13px;
    font-weight: 600;
    color: #e8e8e8;
    margin: 0;
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .tile-source {
    font-size: 10px;
    color: #f0a500;
    opacity: 0.85;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .tile-year {
    font-size: 10px;
    color: rgba(224, 224, 224, 0.45);
  }

  .tile-excerpt {
    font-size: 11px;
    color: rgba(224, 224, 224, 0.6);
    margin: 0;
    line-height: 1.5;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .tile-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 2px;
  }

  .meta-date {
    display: flex;
    align-items: center;
    gap: 3px;
    font-size: 10px;
    color: rgba(224, 224, 224, 0.45);
  }

  .tile-meta {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  /* Video tile specifics */
  .video-thumb {
    position: relative;
    width: 100%;
    aspect-ratio: 16 / 9;
    border-radius: 5px;
    overflow: hidden;
    background: rgba(0, 0, 0, 0.4);
  }

  .video-thumb img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .thumb-placeholder {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: rgba(240, 165, 0, 0.4);
  }

  .duration-badge {
    position: absolute;
    bottom: 4px;
    right: 5px;
    background: rgba(0, 0, 0, 0.75);
    color: #e0e0e0;
    font-size: 10px;
    padding: 1px 5px;
    border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
  }

  .video-info {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  /* =============================================================================
     Empty state
     ============================================================================= */

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 60px 20px;
    color: rgba(224, 224, 224, 0.3);
    font-size: 13px;
    text-align: center;
  }

  /* =============================================================================
     Tags / chips
     ============================================================================= */

  .tag-chip {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    padding: 2px 7px;
    background: rgba(240, 165, 0, 0.1);
    border: 1px solid rgba(240, 165, 0, 0.25);
    border-radius: 4px;
    font-size: 10px;
    color: rgba(240, 165, 0, 0.85);
    white-space: nowrap;
  }

  .tag-chip.small {
    font-size: 9px;
    padding: 1px 5px;
  }

  .meta-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    font-size: 11px;
    color: rgba(224, 224, 224, 0.7);
  }

  .source-chip {
    color: #f0a500;
    border-color: rgba(240, 165, 0, 0.25);
    background: rgba(240, 165, 0, 0.08);
  }

  /* =============================================================================
     Detail sub-page
     ============================================================================= */

  .detail-page {
    max-width: 720px;
  }

  .detail-header {
    margin-bottom: 20px;
  }

  .detail-back {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 12px;
    background: rgba(0, 212, 255, 0.07);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 5px;
    color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
    text-transform: capitalize;
  }

  .detail-back:hover {
    background: rgba(0, 212, 255, 0.14);
    border-color: rgba(0, 212, 255, 0.4);
  }

  .detail-card {
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 24px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(240, 165, 0, 0.18);
    border-radius: 10px;
  }

  .detail-title {
    font-size: 18px;
    font-weight: 700;
    color: #f0a500;
    margin: 0;
    line-height: 1.4;
  }

  .detail-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .detail-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }

  .detail-excerpt {
    font-size: 13px;
    color: rgba(224, 224, 224, 0.75);
    line-height: 1.7;
    margin: 0;
    white-space: pre-wrap;
  }

  .video-thumb-placeholder {
    width: 100%;
    max-width: 480px;
    aspect-ratio: 16 / 9;
    background: rgba(0, 0, 0, 0.35);
    border: 1px solid rgba(240, 165, 0, 0.15);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: rgba(240, 165, 0, 0.35);
  }

  /* =============================================================================
     Kanban wrapper
     ============================================================================= */

  .kanban-wrapper {
    display: flex;
    flex-direction: column;
    width: 100%;
    min-width: 0;
    height: 100%;
    min-height: 400px;
  }

  /* =============================================================================
     Spin animation (shared)
     ============================================================================= */

  :global(.spin) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
  }

  /* =============================================================================
     Scrollbar styling
     ============================================================================= */

  .canvas-body::-webkit-scrollbar {
    width: 4px;
  }

  .canvas-body::-webkit-scrollbar-track {
    background: transparent;
  }

  .canvas-body::-webkit-scrollbar-thumb {
    background: rgba(240, 165, 0, 0.2);
    border-radius: 2px;
  }

  .canvas-body::-webkit-scrollbar-thumb:hover {
    background: rgba(240, 165, 0, 0.35);
  }

  /* =============================================================================
     Pipeline status bar
     ============================================================================= */

  .pipeline-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 5px 20px;
    background: rgba(240, 165, 0, 0.05);
    border-bottom: 1px solid rgba(240, 165, 0, 0.1);
    flex-shrink: 0;
  }

  :global(.pipeline-bar-icon) {
    color: rgba(240, 165, 0, 0.5);
    flex-shrink: 0;
  }

  .pipeline-bar-label {
    font-size: 10px;
    color: rgba(240, 165, 0, 0.5);
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    flex-shrink: 0;
  }

  .pipeline-stages {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .stage-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 2px 8px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(240, 165, 0, 0.18);
    border-radius: 4px;
    font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
    color: rgba(224, 224, 224, 0.45);
    letter-spacing: 0.04em;
    transition: all 0.2s;
  }

  .stage-chip.stage-active {
    background: rgba(0, 212, 255, 0.08);
    border-color: rgba(0, 212, 255, 0.3);
    color: #00d4ff;
  }

  .stage-count {
    font-weight: 700;
    min-width: 12px;
    text-align: center;
  }

  .stage-arrow {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.2);
    font-family: 'JetBrains Mono', monospace;
  }

  .stage-pulse {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #00d4ff;
    box-shadow: 0 0 6px #00d4ff;
    animation: pulse-dot 1.4s ease-in-out infinite;
    flex-shrink: 0;
  }

  @keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.7); }
  }

  /* =============================================================================
     Agent Insights strip
     ============================================================================= */

  .agent-insights-strip {
    flex-shrink: 0;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    background: rgba(8, 13, 20, 0.85);
  }

  .insights-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    width: 100%;
    padding: 6px 16px;
    background: transparent;
    border: none;
    color: rgba(224, 224, 224, 0.45);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    transition: color 0.15s, background 0.15s;
  }

  .insights-toggle:hover {
    color: rgba(224, 224, 224, 0.75);
    background: rgba(255, 255, 255, 0.03);
  }

  .agent-insights-strip.expanded .insights-toggle {
    color: #00d4ff;
  }

  .unread-badge {
    padding: 1px 6px;
    background: rgba(0, 212, 255, 0.15);
    border: 1px solid rgba(0, 212, 255, 0.35);
    border-radius: 10px;
    font-size: 10px;
    color: #00d4ff;
    font-weight: 700;
    line-height: 1.4;
  }

  .insights-toggle .rotated {
    transform: rotate(180deg);
  }
</style>

<script lang="ts">
  import { onMount } from 'svelte';
  import { createEventDispatcher } from 'svelte';
  import {
    Bot, FolderOpen, ChevronRight, ChevronDown, Play, Pause,
    Settings, Trash2, Tag, Filter, Search, CheckCircle, XCircle,
    Clock, AlertTriangle, ArrowRight, FileText, History, User,
    Star, Shield, Eye, Download, Upload, Plus, Home, X
  } from 'lucide-svelte';
  import Breadcrumbs from './Breadcrumbs.svelte';
  import { navigationStore } from '../stores/navigationStore';
  import {
    addEaTag,
    listEaBots,
    listEaReviews,
    listEaTags,
    removeEaTag,
    startVideoIngest,
  } from '$lib/services/tradingApi';

  const dispatch = createEventDispatcher();

  // Breadcrumb items
  const breadcrumbItems = [
    { label: 'EA Management', path: '/ea' }
  ];

  function handleBreadcrumbNavigate(path: string) {
    navigationStore.navigateToPath(path);
  }

  // Video ingest state
  let showVideoIngestModal = false;
  let videoUrl = '';
  let strategyName = '';
  let isProcessing = false;
  let processingJobs: Array<{ id: string; name: string; status: string; progress: number }> = [];

  // Bot lifecycle status
  type BotStatus = 'development' | 'review' | 'primal' | 'pending' | 'quarantine' | 'live';

  interface Bot {
    id: string;
    name: string;
    symbol: string;
    status: BotStatus;
    tags: string[];
    version: string;
    lastModified: Date;
    author: string;
    performance: {
      winRate: number;
      profitFactor: number;
      maxDrawdown: number;
      sharpeRatio: number;
    };
    backtestStatus: 'passed' | 'failed' | 'pending';
  }

  // Tag definitions with colors - loaded from API
  interface TagDefinition {
    id: string;
    label: string;
    color: string;
    description: string;
  }

  let availableTags: TagDefinition[] = [];
  let propFirmTags: string[] = [];

  // Load tags from API
  async function loadTags() {
    try {
      const data = await listEaTags<{ available_tags?: TagDefinition[]; prop_firm_tags?: string[] }>();
      availableTags = data.available_tags || [];
      propFirmTags = data.prop_firm_tags || [];
    } catch (e) {
      console.error('Failed to load tags:', e);
      // Fallback to default tags
      availableTags = [
        { id: 'scalper', label: 'Scalper', color: '#3b82f6', description: 'Short-term trades' },
        { id: 'swing', label: 'Swing', color: '#8b5cf6', description: 'Medium-term trades' },
        { id: 'trend', label: 'Trend', color: '#06b6d4', description: 'Trend following' },
        { id: 'mean-reversion', label: 'Mean Reversion', color: '#ec4899', description: 'Reversal strategy' },
        { id: 'breakout', label: 'Breakout', color: '#f59e0b', description: 'Breakout strategy' },
        { id: 'grid', label: 'Grid', color: '#10b981', description: 'Grid trading' },
      ];
    }
  }

  // Add tag to a bot
  async function addTagToBot(botName: string, tagId: string) {
    try {
      const data = await addEaTag<{ tags: string[] }>(botName, tagId);
      bots = bots.map(b =>
        b.name === botName ? { ...b, tags: data.tags } : b
      );
    } catch (e) {
      console.error('Failed to add tag:', e);
    }
  }

  // Remove tag from a bot
  async function removeTagFromBot(botName: string, tagId: string) {
    try {
      const data = await removeEaTag<{ tags: string[] }>(botName, tagId);
      bots = bots.map(b =>
        b.name === botName ? { ...b, tags: data.tags } : b
      );
    } catch (e) {
      console.error('Failed to remove tag:', e);
    }
  }

  // Check if tag is a prop firm tag
  function isPropFirmTag(tagId: string): boolean {
    return tagId.startsWith('@');
  }

  // Default tags fallback
  const defaultTags: TagDefinition[] = [
    { id: 'primal', label: 'Primal', color: '#10b981', description: 'Production ready' },
    { id: 'pending', label: 'Pending', color: '#f59e0b', description: 'Awaiting review' },
    { id: 'quarantine', label: 'Quarantine', color: '#ef4444', description: 'Temporarily disabled' },
    { id: 'scalper', label: 'Scalper', color: '#3b82f6', description: 'Short-term trades' },
    { id: 'swing', label: 'Swing', color: '#8b5cf6', description: 'Medium-term trades' },
    { id: 'trend', label: 'Trend', color: '#06b6d4', description: 'Trend following' },
    { id: 'mean-reversion', label: 'Mean Reversion', color: '#ec4899', description: 'Reversal strategy' }
  ];

  // Computed tags - use API tags if loaded, otherwise default
  $: tags = availableTags.length > 0 ? availableTags : defaultTags;

  // Filter state
  let searchQuery = '';
  let selectedTags: string[] = [];
  let selectedStatus: BotStatus | 'all' = 'all';
  let sortBy: 'name' | 'status' | 'modified' = 'name';
  let showFilters = false;

  // Data loaded from API
  let bots: Bot[] = [];

  // Load bots from API
  async function loadBots() {
    try {
      bots = await listEaBots<Bot[]>();
    } catch (e) {
      console.error('Failed to load bots:', e);
    }
  }

  onMount(() => {
    loadBots();
    loadReviewHistory();
    loadTags();
  });

  // Review state
  let showReviewPanel = false;
  let selectedBotForReview: Bot | null = null;
  let reviewComment = '';

  // Tag manager state
  let showTagManager = false;
  let selectedBotForTags: Bot | null = null;

  function openTagManager(bot: Bot) {
    selectedBotForTags = bot;
    showTagManager = true;
  }

  function closeTagManager() {
    showTagManager = false;
    selectedBotForTags = null;
  }

  async function toggleTag(tagId: string) {
    if (!selectedBotForTags) return;

    const hasTag = selectedBotForTags.tags.includes(tagId);
    if (hasTag) {
      await removeTagFromBot(selectedBotForTags.name, tagId);
    } else {
      await addTagToBot(selectedBotForTags.name, tagId);
    }
    // Refresh selected bot
    selectedBotForTags = bots.find(b => b.name === selectedBotForTags?.name) || null;
  }

  // Review history loaded from API
  let reviewHistory: Array<{botId: string, botName: string, action: string, reviewer: string, date: Date, comment: string}> = [];

  // Load review history from API
  async function loadReviewHistory() {
    try {
      const reviews = await listEaReviews<Array<{botId: string, botName: string, action: string, reviewer: string, date: Date, comment: string}>>();
      reviewHistory = reviews.map((entry) => ({ ...entry, date: new Date(entry.date) }));
    } catch (e) {
      console.error('Failed to load review history:', e);
    }
  }

  // Filter and sort
  $: filteredBots = bots
    .filter(bot => {
      if (searchQuery && !bot.name.toLowerCase().includes(searchQuery.toLowerCase())) return false;
      if (selectedStatus !== 'all' && bot.status !== selectedStatus) return false;
      if (selectedTags.length > 0 && !selectedTags.every(t => bot.tags.includes(t))) return false;
      return true;
    })
    .sort((a, b) => {
      if (sortBy === 'name') return a.name.localeCompare(b.name);
      if (sortBy === 'status') return a.status.localeCompare(b.status);
      return b.lastModified.getTime() - a.lastModified.getTime();
    });

  // Stats
  $: stats = {
    total: bots.length,
    primal: bots.filter(b => b.status === 'primal').length,
    pending: bots.filter(b => b.status === 'review' || b.status === 'pending').length,
    quarantine: bots.filter(b => b.status === 'quarantine').length
  };

  function getStatusColor(status: BotStatus): string {
    const colors: Record<BotStatus, string> = {
      development: '#6b7280',
      review: '#f59e0b',
      primal: '#10b981',
      pending: '#3b82f6',
      quarantine: '#ef4444',
      live: '#8b5cf6'
    };
    return colors[status];
  }

  function getStatusLabel(status: BotStatus): string {
    const labels: Record<BotStatus, string> = {
      development: 'Development',
      review: 'In Review',
      primal: 'Primal',
      pending: 'Pending',
      quarantine: 'Quarantine',
      live: 'Live'
    };
    return labels[status];
  }

  function toggleTag(tagId: string) {
    if (selectedTags.includes(tagId)) {
      selectedTags = selectedTags.filter(t => t !== tagId);
    } else {
      selectedTags = [...selectedTags, tagId];
    }
  }

  function submitForReview(bot: Bot) {
    selectedBotForReview = bot;
    showReviewPanel = true;
  }

  async function confirmReview(action: 'approve' | 'reject') {
    if (!selectedBotForReview) return;

    const newStatus = action === 'approve' ? 'primal' : 'quarantine';
    
    // Update bot status
    bots = bots.map(b => 
      b.id === selectedBotForReview!.id 
        ? { ...b, status: newStatus, tags: action === 'approve' ? [...b.tags.filter(t => t !== 'pending'), 'primal'] : [...b.tags.filter(t => t !== 'pending'), 'quarantine'] }
        : b
    );

    // Add to history
    reviewHistory = [{
      botId: selectedBotForReview.id,
      botName: selectedBotForReview.name,
      action: action === 'approve' ? 'promoted_to_primal' : 'moved_to_quarantine',
      reviewer: 'Current User',
      date: new Date(),
      comment: reviewComment || (action === 'approve' ? 'Approved for production' : 'Needs fixes')
    }, ...reviewHistory];

    showReviewPanel = false;
    selectedBotForReview = null;
    reviewComment = '';
  }

  function formatDate(date: Date): string {
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  // Video ingest functions
  function openVideoIngest() {
    showVideoIngestModal = true;
  }

  function closeVideoIngest() {
    showVideoIngestModal = false;
    videoUrl = '';
    strategyName = '';
  }

  async function submitVideoIngest() {
    if (!videoUrl || !strategyName) {
      alert('Please enter both YouTube URL and Strategy Name');
      return;
    }

    isProcessing = true;

    try {
      const data = await startVideoIngest<{ job_id: string }>({ url: videoUrl, strategy_name: strategyName });
      processingJobs = [
        ...processingJobs,
        { id: data.job_id, name: strategyName, status: 'processing', progress: 0 }
      ];
      videoUrl = '';
      strategyName = '';
      showVideoIngestModal = false;
    } catch (e) {
      console.error('Video ingest error:', e);
      alert(e instanceof Error ? e.message : 'Failed to start video processing');
    } finally {
      isProcessing = false;
    }
  }

  function getJobStatusColor(status: string): string {
    const colors: Record<string, string> = {
      processing: '#f59e0b',
      completed: '#10b981',
      failed: '#ef4444',
      pending: '#6b7280'
    };
    return colors[status] || '#6b7280';
  }
</script>

<div class="ea-manager">
  <!-- Breadcrumbs -->
  <Breadcrumbs
    items={breadcrumbItems}
    onNavigate={handleBreadcrumbNavigate}
  />

  <!-- Header -->
  <div class="manager-header">
    <div class="header-left">
      <Bot size={24} class="header-icon" />
      <div>
        <h2>EA Management</h2>
        <p>Manage Expert Advisors and deployment lifecycle</p>
      </div>
    </div>
    <div class="header-stats">
      <div class="stat">
        <span class="stat-value">{stats.total}</span>
        <span class="stat-label">Total</span>
      </div>
      <div class="stat primal">
        <span class="stat-value">{stats.primal}</span>
        <span class="stat-label">Primal</span>
      </div>
      <div class="stat pending">
        <span class="stat-value">{stats.pending}</span>
        <span class="stat-label">In Review</span>
      </div>
      <div class="stat quarantine">
        <span class="stat-value">{stats.quarantine}</span>
        <span class="stat-label">Quarantine</span>
      </div>
    </div>
    <button class="video-ingest-btn" on:click={openVideoIngest}>
      <Upload size={14} />
      Video Ingest
    </button>
  </div>

  <!-- Toolbar -->
  <div class="toolbar">
    <div class="search-box">
      <Search size={14} />
      <input type="text" bind:value={searchQuery} placeholder="Search EAs..." />
    </div>

    <div class="toolbar-actions">
      <button class="toolbar-btn" class:active={showFilters} on:click={() => showFilters = !showFilters}>
        <Filter size={14} />
        <span>Filters</span>
      </button>

      <select bind:value={selectedStatus} class="status-filter">
        <option value="all">All Status</option>
        <option value="development">Development</option>
        <option value="review">In Review</option>
        <option value="primal">Primal</option>
        <option value="pending">Pending</option>
        <option value="quarantine">Quarantine</option>
        <option value="live">Live</option>
      </select>

      <select bind:value={sortBy} class="sort-filter">
        <option value="name">Sort by Name</option>
        <option value="status">Sort by Status</option>
        <option value="modified">Sort by Modified</option>
      </select>
    </div>
  </div>

  <!-- Tag Filters -->
  {#if showFilters}
    <div class="tag-filters">
      <span class="filter-label">Tags:</span>
      {#each tags as tag}
        <button 
          class="tag-btn" 
          class:selected={selectedTags.includes(tag.id)}
          style="--tag-color: {tag.color}"
          on:click={() => toggleTag(tag.id)}
          title={tag.description}
        >
          {tag.label}
        </button>
      {/each}
    </div>
  {/if}

  <!-- EA Grid -->
  <div class="ea-grid">
    {#each filteredBots as bot}
      <div class="ea-card">
        <div class="ea-header">
          <div class="ea-info">
            <Bot size={18} />
            <div>
              <h3>{bot.name}</h3>
              <span class="ea-meta">{bot.symbol} • v{bot.version}</span>
            </div>
          </div>
          <div class="ea-status" style="--status-color: {getStatusColor(bot.status)}">
            {getStatusLabel(bot.status)}
          </div>
        </div>

        <!-- Tags -->
        <div class="ea-tags">
          {#each bot.tags as tag}
            {@const tagInfo = tags.find(t => t.id === tag)}
            {#if tagInfo}
              <span class="tag" style="--tag-color: {tagInfo.color}" class:prop-firm={isPropFirmTag(tag)}>
                {tagInfo.label}
                <button class="tag-remove" on:click|stopPropagation={() => removeTagFromBot(bot.name, tag)}>×</button>
              </span>
            {:else}
              <span class="tag unknown">{tag}</span>
            {/if}
          {/each}
          <button class="tag-add-btn" on:click|stopPropagation={() => openTagManager(bot)}>
            <Plus size={10} /> Add Tag
          </button>
        </div>

        <!-- Performance -->
        <div class="ea-performance">
          <div class="perf-item">
            <span class="perf-label">Win Rate</span>
            <span class="perf-value">{bot.performance.winRate * 100}%</span>
          </div>
          <div class="perf-item">
            <span class="perf-label">PF</span>
            <span class="perf-value">{bot.performance.profitFactor.toFixed(2)}</span>
          </div>
          <div class="perf-item">
            <span class="perf-label">Max DD</span>
            <span class="perf-value">{(bot.performance.maxDrawdown * 100).toFixed(1)}%</span>
          </div>
          <div class="perf-item">
            <span class="perf-label">Sharpe</span>
            <span class="perf-value">{bot.performance.sharpeRatio.toFixed(2)}</span>
          </div>
        </div>

        <!-- Backtest Status -->
        <div class="backtest-badge" class:passed={bot.backtestStatus === 'passed'} class:failed={bot.backtestStatus === 'failed'}>
          {#if bot.backtestStatus === 'passed'}
            <CheckCircle size={12} />
            <span>Passed</span>
          {:else if bot.backtestStatus === 'failed'}
            <XCircle size={12} />
            <span>Failed</span>
          {:else}
            <Clock size={12} />
            <span>Pending</span>
          {/if}
        </div>

        <!-- Meta -->
        <div class="ea-meta-footer">
          <span class="author"><User size={10} /> {bot.author}</span>
          <span class="modified"><Clock size={10} /> {formatDate(bot.lastModified)}</span>
        </div>

        <!-- Actions -->
        <div class="ea-actions">
          {#if bot.status === 'development'}
            <button class="action-btn" on:click={() => submitForReview(bot)}>
              <Upload size={12} />
              <span>Submit for Review</span>
            </button>
          {:else if bot.status === 'review'}
            <button class="action-btn primary" on:click={() => { selectedBotForReview = bot; showReviewPanel = true; }}>
              <CheckCircle size={12} />
              <span>Review</span>
            </button>
          {:else if bot.status === 'primal'}
            <button class="action-btn">
              <Play size={12} />
              <span>Deploy to Live</span>
            </button>
          {/if}
          <button class="action-btn">
            <Settings size={12} />
            <span>Settings</span>
          </button>
        </div>
      </div>
    {/each}

    {#if filteredBots.length === 0}
      <div class="empty-state">
        <Bot size={48} />
        <p>No EAs found matching your criteria</p>
      </div>
    {/if}
  </div>

  <!-- Review Panel Modal -->
  {#if showReviewPanel && selectedBotForReview}
    <div class="modal-overlay" on:click={() => showReviewPanel = false}>
      <div class="modal review-panel" on:click|stopPropagation>
        <div class="modal-header">
          <h3>Review: {selectedBotForReview.name}</h3>
          <button class="close-btn" on:click={() => showReviewPanel = false}><XCircle size={18} /></button>
        </div>

        <div class="review-content">
          <div class="review-info">
            <div class="info-row">
              <span class="label">Status</span>
              <span class="value status" style="--status-color: {getStatusColor(selectedBotForReview.status)}">
                {getStatusLabel(selectedBotForReview.status)}
              </span>
            </div>
            <div class="info-row">
              <span class="label">Backtest</span>
              <span class="value">{selectedBotForReview.backtestStatus}</span>
            </div>
            <div class="info-row">
              <span class="label">Win Rate</span>
              <span class="value">{selectedBotForReview.performance.winRate * 100}%</span>
            </div>
          </div>

          <div class="review-comment">
            <label>Review Comment</label>
            <textarea bind:value={reviewComment} placeholder="Add your review notes..."></textarea>
          </div>

          <div class="review-actions">
            <button class="btn reject" on:click={() => confirmReview('reject')}>
              <XCircle size={14} />
              <span>Reject to Quarantine</span>
            </button>
            <button class="btn approve" on:click={() => confirmReview('approve')}>
              <CheckCircle size={14} />
              <span>Approve to Primal</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  {/if}

  <!-- Video Ingest Modal -->
  {#if showVideoIngestModal}
    <div class="modal-overlay" on:click={closeVideoIngest}>
      <div class="modal video-ingest-modal" on:click|stopPropagation>
        <div class="modal-header">
          <h3>Video Ingest</h3>
          <button class="close-btn" on:click={closeVideoIngest}><X size={18} /></button>
        </div>

        <div class="modal-content">
          <p class="ingest-description">
            Enter a YouTube URL to extract trading strategies from video content.
            The video will be processed and analyzed to identify potential trading patterns.
          </p>

          <div class="form-group">
            <label for="videoUrl">YouTube URL</label>
            <input
              id="videoUrl"
              type="text"
              bind:value={videoUrl}
              placeholder="https://www.youtube.com/watch?v=..."
              disabled={isProcessing}
            />
          </div>

          <div class="form-group">
            <label for="strategyName">Strategy Name</label>
            <input
              id="strategyName"
              type="text"
              bind:value={strategyName}
              placeholder="Enter a name for this strategy"
              disabled={isProcessing}
            />
          </div>

          {#if processingJobs.length > 0}
            <div class="processing-jobs">
              <h4>Processing Jobs</h4>
              {#each processingJobs as job}
                <div class="job-item">
                  <div class="job-info">
                    <span class="job-name">{job.name}</span>
                    <span class="job-status" style="color: {getJobStatusColor(job.status)}">
                      {job.status}
                    </span>
                  </div>
                  <div class="job-progress">
                    <div class="progress-bar" style="width: {job.progress}%"></div>
                  </div>
                </div>
              {/each}
            </div>
          {/if}

          <div class="modal-actions">
            <button class="btn cancel" on:click={closeVideoIngest} disabled={isProcessing}>
              Cancel
            </button>
            <button class="btn submit" on:click={submitVideoIngest} disabled={isProcessing || !videoUrl || !strategyName}>
              {#if isProcessing}
                Processing...
              {:else}
                Start Processing
              {/if}
            </button>
          </div>
        </div>
      </div>
    </div>
  {/if}

  <!-- Tag Manager Modal -->
  {#if showTagManager && selectedBotForTags}
    <div class="modal-overlay" on:click={closeTagManager}>
      <div class="modal tag-manager-modal" on:click|stopPropagation>
        <div class="modal-header">
          <h3>Manage Tags: {selectedBotForTags.name}</h3>
          <button class="close-btn" on:click={closeTagManager}><XCircle size={18} /></button>
        </div>

="modal-content">
        <div class="tag-manager-content">
          <!-- Strategy Tags -->
          <div class="tag-section">
            <h4>Strategy Tags</h4>
            <div class="tag-grid">
              {#each tags.filter(t => !isPropFirmTag(t.id)) as tag}
                {@const isSelected = selectedBotForTags.tags.includes(tag.id)}
                <button
                  class="tag-option"
                  class:selected={isSelected}
                  style="--tag-color: {tag.color}"
                  on:click={() => toggleTag(tag.id)}
                >
                  <span class="tag-check">{isSelected ? '✓' : ''}</span>
                  <span class="tag-label">{tag.label}</span>
                  <span class="tag-desc">{tag.description}</span>
                </button>
              {/each}
            </div>
          </div>

          <!-- Prop Firm Tags -->
          <div class="tag-section">
            <h4>Prop Firm Tags</h4>
            <div class="tag-grid">
              {#each tags.filter(t => isPropFirmTag(t.id)) as tag}
                {@const isSelected = selectedBotForTags.tags.includes(tag.id)}
                <button
                  class="tag-option prop-firm"
                  class:selected={isSelected}
                  style="--tag-color: {tag.color}"
                  on:click={() => toggleTag(tag.id)}
                >
                  <span class="tag-check">{isSelected ? '✓' : ''}</span>
                  <span class="tag-label">{tag.label}</span>
                  <span class="tag-desc">{tag.description}</span>
                </button>
              {/each}
            </div>
          </div>

          <div class="modal-actions">
            <button class="btn done" on:click={closeTagManager}>Done</button>
          </div>
        </div>
      </div>
    </div>
  {/if}

  <!-- Review History -->
  <div class="review-history">
    <h3><History size={14} /> Review History</h3>
    <div class="history-list">
      {#each reviewHistory as entry}
        <div class="history-item">
          <div class="history-icon" class:approve={entry.action.includes('primal')} class:reject={entry.action.includes('quarantine')}>
            {#if entry.action.includes('primal')}
              <Star size={12} />
            {:else}
              <AlertTriangle size={12} />
            {/if}
          </div>
          <div class="history-content">
            <span class="history-action">{entry.botName} - {entry.action.replace(/_/g, ' ')}</span>
            <span class="history-meta">{entry.reviewer} • {formatDate(entry.date)}</span>
            {#if entry.comment}
              <span class="history-comment">"{entry.comment}"</span>
            {/if}
          </div>
        </div>
      {/each}
    </div>
  </div>
</div>

<style>
  .ea-manager {
    display: flex;
    flex-direction: column;
    height: 100%;
    padding: 20px;
    background: var(--bg-primary);
    overflow-y: auto;
  }

  .manager-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 20px;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .header-icon {
    color: var(--accent-primary);
  }

  .header-left h2 {
    margin: 0;
    font-size: 20px;
    color: var(--text-primary);
  }

  .header-left p {
    margin: 4px 0 0;
    font-size: 12px;
    color: var(--text-muted);
  }

  .header-stats {
    display: flex;
    gap: 16px;
  }

  .stat {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 8px 16px;
    background: var(--bg-secondary);
    border-radius: 8px;
  }

  .stat-value {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .stat-label {
    font-size: 10px;
    text-transform: uppercase;
    color: var(--text-muted);
  }

  .stat.primal .stat-value { color: #10b981; }
  .stat.pending .stat-value { color: #f59e0b; }
  .stat.quarantine .stat-value { color: #ef4444; }

  .toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    flex-wrap: wrap;
    gap: 12px;
  }

  .search-box {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    color: var(--text-muted);
  }

  .search-box input {
    background: transparent;
    border: none;
    color: var(--text-primary);
    font-size: 13px;
    width: 200px;
  }

  .search-box input:focus {
    outline: none;
  }

  .toolbar-actions {
    display: flex;
    gap: 12px;
  }

  .toolbar-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
  }

  .toolbar-btn.active {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: var(--bg-primary);
  }

  .status-filter,
  .sort-filter {
    padding: 8px 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 12px;
  }

  .tag-filters {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
    flex-wrap: wrap;
  }

  .filter-label {
    font-size: 12px;
    color: var(--text-muted);
  }

  .tag-btn {
    padding: 4px 10px;
    background: transparent;
    border: 1px solid var(--tag-color);
    border-radius: 12px;
    color: var(--tag-color);
    font-size: 11px;
    cursor: pointer;
  }

  .tag-btn.selected {
    background: var(--tag-color);
    color: white;
  }

  .ea-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 16px;
    flex: 1;
  }

  .ea-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 16px;
    transition: all 0.2s;
  }

  .ea-card:hover {
    border-color: var(--accent-primary);
  }

  .ea-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 12px;
  }

  .ea-info {
    display: flex;
    gap: 10px;
  }

  .ea-info h3 {
    margin: 0;
    font-size: 14px;
    color: var(--text-primary);
  }

  .ea-meta {
    font-size: 11px;
    color: var(--text-muted);
  }

  .ea-status {
    padding: 4px 10px;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 500;
    background: color-mix(in srgb, var(--status-color) 20%, transparent);
    color: var(--status-color);
  }

  .ea-tags {
    display: flex;
    gap: 6px;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }

  .tag {
    padding: 2px 8px;
    border-radius: 8px;
    font-size: 10px;
    background: color-mix(in srgb, var(--tag-color) 20%, transparent);
    color: var(--tag-color);
  }

  .ea-performance {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-bottom: 12px;
    padding: 10px;
    background: var(--bg-tertiary);
    border-radius: 6px;
  }

  .perf-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
  }

  .perf-label {
    font-size: 9px;
    text-transform: uppercase;
    color: var(--text-muted);
  }

  .perf-value {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .backtest-badge {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    padding: 6px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    font-size: 11px;
    color: var(--text-muted);
    margin-bottom: 12px;
  }

  .backtest-badge.passed {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .backtest-badge.failed {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .ea-meta-footer {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--text-muted);
    margin-bottom: 12px;
  }

  .ea-meta-footer span {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .ea-actions {
    display: flex;
    gap: 8px;
  }

  .action-btn {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    padding: 8px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 11px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .action-btn:hover {
    background: var(--bg-surface);
    color: var(--text-primary);
  }

  .action-btn.primary {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: var(--bg-primary);
  }

  .empty-state {
    grid-column: 1 / -1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px;
    color: var(--text-muted);
    text-align: center;
  }

  .empty-state p {
    margin-top: 12px;
    font-size: 14px;
  }

  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .modal {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    width: 100%;
    max-width: 500px;
    overflow: hidden;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    background: var(--bg-tertiary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .modal-header h3 {
    margin: 0;
    font-size: 16px;
    color: var(--text-primary);
  }

  .close-btn {
    background: transparent;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
  }

  .review-content {
    padding: 20px;
  }

  .review-info {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-bottom: 16px;
    padding: 16px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .info-row {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
  }

  .info-row .label {
    color: var(--text-muted);
  }

  .info-row .value {
    color: var(--text-primary);
    font-weight: 500;
  }

  .info-row .value.status {
    padding: 2px 8px;
    border-radius: 8px;
    background: color-mix(in srgb, var(--status-color) 20%, transparent);
  }

  .review-comment {
    margin-bottom: 16px;
  }

  .review-comment label {
    display: block;
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 8px;
  }

  .review-comment textarea {
    width: 100%;
    padding: 10px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
    resize: vertical;
    min-height: 80px;
  }

  .review-actions {
    display: flex;
    gap: 12px;
  }

  .review-actions .btn {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 12px;
    border: none;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
  }

  .review-actions .reject {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .review-actions .approve {
    background: #10b981;
    color: white;
  }

  .review-actions .reject {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .review-actions .approve {
    background: #10b981;
    color: white;
  }

  .review-history {
    margin-top: 24px;
    padding-top: 20px;
    border-top: 1px solid var(--border-subtle);
  }

  .review-history h3 {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0 0 16px;
    font-size: 14px;
    color: var(--text-primary);
  }

  .history-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .history-item {
    display: flex;
    gap: 12px;
    padding: 12px;
    background: var(--bg-secondary);
    border-radius: 8px;
  }

  .history-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: var(--bg-tertiary);
    color: var(--text-muted);
  }

  .history-icon.approve {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .history-icon.reject {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .history-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .history-action {
    font-size: 13px;
    color: var(--text-primary);
    text-transform: capitalize;
  }

  .history-meta {
    font-size: 11px;
    color: var(--text-muted);
  }

  .history-comment {
    font-size: 12px;
    color: var(--text-secondary);
    font-style: italic;
  }

  /* Video Ingest Button */
  .video-ingest-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    background: var(--accent-primary);
    border: none;
    border-radius: 8px;
    color: var(--bg-primary);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
  }

  .video-ingest-btn:hover {
    filter: brightness(1.1);
    transform: translateY(-1px);
  }

  /* Video Ingest Modal */
  .video-ingest-modal {
    max-width: 500px;
  }

  .modal-content {
    padding: 20px;
  }

  .ingest-description {
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 20px;
    line-height: 1.5;
  }

  .form-group {
    margin-bottom: 16px;
  }

  .form-group label {
    display: block;
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 8px;
  }

  .form-group input {
    width: 100%;
    padding: 10px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
    box-sizing: border-box;
  }

  .form-group input:focus {
    outline: none;
    border-color: var(--accent-primary);
  }

  .form-group input:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .processing-jobs {
    margin: 20px 0;
    padding: 16px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .processing-jobs h4 {
    margin: 0 0 12px;
    font-size: 12px;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .job-item {
    padding: 10px 0;
    border-bottom: 1px solid var(--border-subtle);
  }

  .job-item:last-child {
    border-bottom: none;
  }

  .job-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }

  .job-name {
    font-size: 13px;
    color: var(--text-primary);
    font-weight: 500;
  }

  .job-status {
    font-size: 11px;
    text-transform: uppercase;
    font-weight: 500;
  }

  .job-progress {
    height: 4px;
    background: var(--bg-secondary);
    border-radius: 2px;
    overflow: hidden;
  }

  .progress-bar {
    height: 100%;
    background: var(--accent-primary);
    transition: width 0.3s ease;
  }

  .modal-actions {
    display: flex;
    gap: 12px;
    margin-top: 20px;
  }

  .modal-actions .btn {
    flex: 1;
    padding: 12px;
    border: none;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
  }

  .modal-actions .btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .modal-actions .cancel {
    background: var(--bg-tertiary);
    color: var(--text-secondary);
  }

  .modal-actions .submit {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  .modal-actions .submit:hover:not(:disabled) {
    filter: brightness(1.1);
  }

  /* Tag Manager Modal */
  .tag-manager-modal {
    max-width: 500px;
  }

  .tag-manager-modal .modal-content {
    padding: 20px;
  }

  .tag-section {
    margin-bottom: 20px;
  }

  .tag-section h4 {
    margin: 0 0 12px;
    font-size: 12px;
    text-transform: uppercase;
    color: var(--text-muted);
    letter-spacing: 0.5px;
  }

  .tag-grid {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .tag-option {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.15s;
    text-align: left;
  }

  .tag-option:hover {
    border-color: var(--tag-color);
    background: color-mix(in srgb, var(--tag-color) 10%, transparent);
  }

  .tag-option.selected {
    border-color: var(--tag-color);
    background: color-mix(in srgb, var(--tag-color) 15%, transparent);
  }

  .tag-option.prop-firm {
    border-left: 3px solid var(--tag-color);
  }

  .tag-check {
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 4px;
    background: var(--bg-secondary);
    font-size: 12px;
    color: var(--tag-color);
  }

  .tag-option.selected .tag-check {
    background: var(--tag-color);
    color: white;
  }

  .tag-label {
    flex: 1;
    font-weight: 500;
    color: var(--text-primary);
  }

  .tag-desc {
    font-size: 11px;
    color: var(--text-muted);
  }

  .modal-actions .done {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  /* Tag on EA Card */
  .ea-tags {
    display: flex;
    gap: 6px;
    margin-bottom: 12px;
    flex-wrap: wrap;
    align-items: center;
  }

  .ea-tags .tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 8px;
    font-size: 10px;
    background: color-mix(in srgb, var(--tag-color) 20%, transparent);
    color: var(--tag-color);
  }

  .ea-tags .tag.prop-firm {
    border: 1px dashed var(--tag-color);
  }

  .ea-tags .tag.unknown {
    background: var(--bg-tertiary);
    color: var(--text-muted);
    border: 1px dashed var(--border-subtle);
  }

  .tag-remove {
    display: none;
    background: none;
    border: none;
    color: inherit;
    cursor: pointer;
    padding: 0;
    margin-left: 2px;
    font-size: 12px;
    line-height: 1;
  }

  .ea-tags .tag:hover .tag-remove {
    display: inline;
  }

  .tag-add-btn {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    background: transparent;
    border: 1px dashed var(--border-subtle);
    border-radius: 8px;
    font-size: 10px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s;
  }

  .tag-add-btn:hover {
    border-color: var(--accent-primary);
    color: var(--accent-primary);
  }
</style>

<!-- @migration-task Error while migrating Svelte code: Expected token =
https://svelte.dev/e/expected_token -->
<!-- @migration-task Error while migrating Svelte code: Expected token =
https://svelte.dev/e/expected_token -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { fade, fly } from 'svelte/transition';
  import {
    Share2, Search, Filter, Plus, RefreshCw, Eye, Edit, Trash2,
    ChevronDown, X, BookOpen, Lightbulb, Users, Calendar, Hash
  } from 'lucide-svelte';

  // Types
  interface SharedResource {
    id: string;
    title: string;
    type: 'strategy' | 'knowledge';
    content: string;
    shared_by: string;
    shared_with: string[];
    department: string;
    created_at: string;
    tags?: string[];
  }

  // State
  let resources: SharedResource[] = [];
  let filteredResources: SharedResource[] = [];
  let loading = false;
  let error: string | null = null;
  let expandedId: string | null = null;

  // Filters
  let searchQuery = '';
  let typeFilter = 'all';
  let deptFilter = 'all';

  const API_BASE = 'http://localhost:8000/api';

  onMount(() => {
    loadResources();
  });

  async function loadResources() {
    loading = true;
    error = null;

    try {
      const res = await fetch(`${API_BASE}/shared-resources`);
      if (!res.ok) throw new Error('Failed to load resources');

      const data = await res.json();
      resources = data.resources || [];
      applyFilters();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load resources';
      console.error('Failed to load resources:', e);
    } finally {
      loading = false;
    }
  }

  function applyFilters() {
    filteredResources = resources.filter(r => {
      if (typeFilter !== 'all' && r.type !== typeFilter) return false;
      if (deptFilter !== 'all' && r.department !== deptFilter) return false;
      if (searchQuery) {
        const q = searchQuery.toLowerCase();
        return (
          r.title.toLowerCase().includes(q) ||
          r.content.toLowerCase().includes(q)
        );
      }
      return true;
    });
  }

  function toggleExpanded(id: string) {
    expandedId = expandedId === id ? null : id;
  }

  function formatTimestamp(iso: string) {
    return new Date(iso).toLocaleString();
  }

  function getTypeColor(type: string) {
    return type === 'strategy' ? '#3b82f6' : '#8b5cf6';
  }

  function getDeptColor(dept: string) {
    const colors: Record<string, string> = {
      'development': '#3b82f6',
      'research': '#8b5cf6',
      'risk': '#ef4444',
      'trading': '#f97316',
      'portfolio': '#10b981'
    };
    return colors[dept] || '#6b7280';
  }
</script>

<div class="shared-resources-view">
  <!-- Header -->
  <div class="resources-header">
    <div class="header-left">
      <Share2 size={24} class="icon" />
      <div>
        <h2>Shared Resources</h2>
        <p>Strategies and knowledge shared across departments</p>
      </div>
    </div>
    <div class="header-actions">
      <button class="btn" on:click={loadResources}>
        <RefreshCw size={14} />
        <span>Refresh</span>
      </button>
      <button class="btn primary">
        <Plus size={14} />
        <span>Share Resource</span>
      </button>
    </div>
  </div>

  <!-- Filters -->
  <div class="filter-bar">
    <div class="search-group">
      <Search size={14} />
      <input type="text" placeholder="Search..." bind:value={searchQuery} on:input={applyFilters} />
    </div>
    <div class="filter-group">
      <Filter size={14} />
      <select bind:value={typeFilter} on:change={applyFilters}>
        <option value="all">All Types</option>
        <option value="strategy">Strategies</option>
        <option value="knowledge">Knowledge</option>
      </select>
    </div>
    <div class="stats-summary">
      <span>{filteredResources.length} resources</span>
    </div>
  </div>

  <!-- Content -->
  <div class="resources-content">
    {#if loading}
      <div class="loading-state">
        <RefreshCw size={32} class="spin" />
        <span>Loading...</span>
      </div>
    {:else if filteredResources.length > 0}
      <div class="resources-list">
        {#each filteredResources as resource}
          <div class="resource-card" class:expanded={expandedId === resource.id} in:fly={{ y: 20 }}>
            <div class="resource-header" on:click={() => toggleExpanded(resource.id)}>
              <div class="resource-info">
                <div class="resource-title">
                  {#if resource.type === 'strategy'}
                    <BookOpen size={14} style="color: {getTypeColor('strategy')}" />
                  {:else}
                    <Lightbulb size={14} style="color: {getTypeColor('knowledge')}" />
                  {/if}
                  <span>{resource.title}</span>
                </div>
                <div class="resource-meta">
                  <span class="type-badge" style="background: {getTypeColor(resource.type)}20; color: {getTypeColor(resource.type)}">
                    {resource.type}
                  </span>
                  <span class="dept-badge" style="background: {getDeptColor(resource.department)}20; color: {getDeptColor(resource.department)}">
                    {resource.department}
                  </span>
                  <span class="shared-by">
                    <Users size={10} />
                    <span>{resource.shared_by}</span>
                  </span>
                </div>
              </div>
              <ChevronDown size={16} class:expand-icon" class:expanded={expandedId === resource.id} />
            </div>

            {#if expandedId === resource.id}
              <div class="resource-body">
                <p class="resource-content">{resource.content}</p>
                {#if resource.tags && resource.tags.length > 0}
                  <div class="tags">
                    {#each resource.tags as tag}
                      <span class="tag">{tag}</span>
                    {/each}
                  </div>
                {/if}
                <div class="resource-footer">
                  <span class="timestamp">
                    <Calendar size={10} />
                    {formatTimestamp(resource.created_at)}
                  </span>
                  <span class="shared-with">
                    Shared with: {resource.shared_with.join(', ')}
                  </span>
                </div>
              </div>
            {/if}
          </div>
        {/each}
      </div>
    {:else}
      <div class="empty-state">
        <Share2 size={48} />
        <p>No shared resources yet</p>
        <button class="btn primary">
          <Plus size={14} />
          <span>Share First Resource</span>
        </button>
      </div>
    {/if}
  </div>
</div>

<style>
  .shared-resources-view {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary);
    overflow: hidden;
  }

  .resources-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 24px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .icon {
    color: var(--accent-primary);
  }

  .header-left h2 {
    margin: 0;
    font-size: 18px;
    color: var(--text-primary);
  }

  .header-left p {
    margin: 2px 0 0;
    font-size: 12px;
    color: var(--text-muted);
  }

  .header-actions {
    display: flex;
    gap: 8px;
  }

  .btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
  }

  .btn.primary {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: var(--bg-primary);
  }

  .filter-bar {
    display: flex;
    gap: 12px;
    padding: 16px 24px;
    background: var(--bg-primary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .search-group,
  .filter-group {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
  }

  .search-group input,
  .filter-group select {
    background: transparent;
    border: none;
    color: var(--text-primary);
    font-size: 12px;
  }

  .stats-summary {
    margin-left: auto;
    font-size: 11px;
    color: var(--text-muted);
  }

  .resources-content {
    flex: 1;
    overflow-y: auto;
    padding: 16px 24px;
  }

  .resources-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .resource-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    overflow: hidden;
  }

  .resource-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 16px;
    cursor: pointer;
  }

  .resource-header:hover {
    background: var(--bg-tertiary);
  }

  .resource-info {
    flex: 1;
  }

  .resource-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary);
    margin-bottom: 6px;
  }

  .resource-meta {
    display: flex;
    gap: 12px;
    font-size: 11px;
  }

  .type-badge,
  .dept-badge,
  .shared-by {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 4px;
  }

  .expand-icon {
    color: var(--text-muted);
    transition: transform 0.2s;
  }

  .expand-icon.expanded {
    transform: rotate(180deg);
  }

  .resource-body {
    padding: 0 16px 16px;
  }

  .resource-content {
    margin: 0 0 12px;
    font-size: 13px;
    color: var(--text-secondary);
    white-space: pre-wrap;
  }

  .tags {
    display: flex;
    gap: 6px;
    margin-bottom: 12px;
  }

  .tag {
    padding: 4px 10px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    font-size: 11px;
    color: var(--accent-primary);
  }

  .resource-footer {
    display: flex;
    gap: 16px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .timestamp,
  .shared-with {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 24px;
    color: var(--text-muted);
    gap: 16px;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .spin {
    animation: spin 1s linear infinite;
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 24px;
    color: var(--text-muted);
    text-align: center;
    gap: 16px;
  }
</style>

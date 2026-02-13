<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import {
    Package, Code, Shield, Search, Filter, Plus, Eye, Edit,
    Trash2, ChevronRight, ChevronDown, ChevronUp, X, FileCode, GitBranch,
    CheckCircle, AlertCircle, History, Download, Upload,
    RefreshCw, Settings as SettingsIcon, FolderOpen, Tag, Clock, Zap, Database
  } from 'lucide-svelte';
  import DatabaseView from './DatabaseView.svelte';

  const dispatch = createEventDispatcher();

  // Shared Asset Data Structure
  interface SharedAsset {
    id: string;
    name: string;
    category: 'Indicator' | 'Risk' | 'Utils';
    version: string;
    filesystem_path: string;
    dependencies: string[];
    checksum: string;
    created_by: 'QuantCode' | 'user';
    used_by_count: number;
    created_at: string;
    updated_at: string;
    description?: string;
  }

  // Asset History Entry
  interface AssetHistory {
    version: string;
    checksum: string;
    created_at: string;
    created_by: 'QuantCode' | 'user';
    change_description: string;
  }

  // State
  let assets: SharedAsset[] = [];
  let filteredAssets: SharedAsset[] = [];
  let selectedAsset: SharedAsset | null = null;

  // View state
  let expandedAsset: string | null = null;
  let detailViewOpen = false;
  let addAssetModalOpen = false;
  let editAssetModalOpen = false;
  let historyModalOpen = false;
  let activeTab = 'assets'; // 'assets' or 'database'

  // Filters
  let categoryFilter = 'all';
  let searchQuery = '';

  // New asset form
  let newAsset = {
    name: '',
    category: 'Indicator' as 'Indicator' | 'Risk' | 'Utils',
    code: '',
    description: '',
    dependencies: [] as string[]
  };

  // Asset history
  let assetHistory: AssetHistory[] = [];

  // Permission settings
  let permissionSettings = {
    quantCodeWrite: true,
    copilotWrite: false,
    userWrite: true
  };

  onMount(() => {
    loadAssets();
  });

  async function loadAssets() {
    try {
      const res = await fetch('http://localhost:8000/api/assets/shared');
      if (res.ok) {
        assets = await res.json();
        applyFilters();
      }
    } catch (e) {
      console.error('Failed to load assets:', e);
      // Use mock data for development
      loadMockAssets();
    }
  }

  function loadMockAssets() {
    assets = [
      {
        id: 'asset-1',
        name: 'AdaptiveRSI',
        category: 'Indicator',
        version: '1.2.0',
        filesystem_path: '/mql5/Include/QuantMind/Indicators/AdaptiveRSI.mqh',
        dependencies: ['MovingAverage', 'ATR'],
        checksum: 'a1b2c3d4e5f6',
        created_by: 'QuantCode',
        used_by_count: 5,
        created_at: '2024-01-15T10:30:00Z',
        updated_at: '2024-01-20T14:22:00Z',
        description: 'Adaptive RSI indicator with dynamic period adjustment'
      },
      {
        id: 'asset-2',
        name: 'KellyCalculator',
        category: 'Risk',
        version: '2.0.1',
        filesystem_path: '/mql5/Include/QuantMind/Risk/KellyCalculator.mqh',
        dependencies: ['Statistics', 'Portfolio'],
        checksum: 'f6e5d4c3b2a1',
        created_by: 'QuantCode',
        used_by_count: 8,
        created_at: '2024-01-10T09:00:00Z',
        updated_at: '2024-01-25T16:45:00Z',
        description: 'Kelly criterion calculator for optimal position sizing'
      },
      {
        id: 'asset-3',
        name: 'SocketClient',
        category: 'Utils',
        version: '1.0.5',
        filesystem_path: '/mql5/Include/QuantMind/Utils/SocketClient.mqh',
        dependencies: [],
        checksum: '9z8y7x6w5v4',
        created_by: 'user',
        used_by_count: 3,
        created_at: '2024-01-05T11:20:00Z',
        updated_at: '2024-01-18T13:10:00Z',
        description: 'WebSocket client for real-time communication'
      },
      {
        id: 'asset-4',
        name: 'RegimeDetector',
        category: 'Indicator',
        version: '1.1.0',
        filesystem_path: '/mql5/Include/QuantMind/Indicators/RegimeDetector.mqh',
        dependencies: ['AdaptiveRSI', 'Volatility'],
        checksum: 'm1n2b3v4c5x',
        created_by: 'QuantCode',
        used_by_count: 6,
        created_at: '2024-01-12T15:40:00Z',
        updated_at: '2024-01-22T10:15:00Z',
        description: 'Market regime detection with quality scoring'
      },
      {
        id: 'asset-5',
        name: 'TieredRisk',
        category: 'Risk',
        version: '1.5.0',
        filesystem_path: '/mql5/Include/QuantMind/Risk/TieredRisk.mqh',
        dependencies: ['KellyCalculator', 'Governor'],
        checksum: 'q1w2e3r4t5y',
        created_by: 'user',
        used_by_count: 4,
        created_at: '2024-01-08T14:00:00Z',
        updated_at: '2024-01-19T09:30:00Z',
        description: 'Tiered risk management with prop firm constraints'
      }
    ];
    applyFilters();
  }

  function applyFilters() {
    filteredAssets = assets.filter(asset => {
      if (categoryFilter !== 'all' && asset.category !== categoryFilter) return false;
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        return (
          asset.name.toLowerCase().includes(query) ||
          asset.description?.toLowerCase().includes(query) ||
          asset.filesystem_path.toLowerCase().includes(query)
        );
      }
      return true;
    });
  }

  function toggleExpanded(assetId: string) {
    expandedAsset = expandedAsset === assetId ? null : assetId;
  }

  function selectAsset(asset: SharedAsset) {
    selectedAsset = asset;
    detailViewOpen = true;
  }

  function viewAsset(asset: SharedAsset) {
    selectedAsset = asset;
    detailViewOpen = true;
  }

  function editAsset(asset: SharedAsset) {
    selectedAsset = asset;
    editAssetModalOpen = true;
  }

  async function deleteAsset(asset: SharedAsset) {
    if (!confirm(`Are you sure you want to delete "${asset.name}"?`)) return;

    try {
      const res = await fetch(`http://localhost:8000/api/assets/${asset.id}`, {
        method: 'DELETE'
      });
      if (res.ok) {
        assets = assets.filter(a => a.id !== asset.id);
        applyFilters();
      }
    } catch (e) {
      console.error('Failed to delete asset:', e);
      // For development, remove locally
      assets = assets.filter(a => a.id !== asset.id);
      applyFilters();
    }
  }

  async function createAsset() {
    if (!newAsset.name || !newAsset.code) {
      alert('Please fill in all required fields');
      return;
    }

    try {
      const res = await fetch('http://localhost:8000/api/assets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newAsset)
      });
      if (res.ok) {
        const created = await res.json();
        assets = [created, ...assets];
        applyFilters();
        addAssetModalOpen = false;
        resetNewAssetForm();
      }
    } catch (e) {
      console.error('Failed to create asset:', e);
      // For development, add locally
      const mockAsset: SharedAsset = {
        id: `asset-${Date.now()}`,
        name: newAsset.name,
        category: newAsset.category,
        version: '1.0.0',
        filesystem_path: `/mql5/Include/QuantMind/${newAsset.category}/${newAsset.name}.mqh`,
        dependencies: newAsset.dependencies,
        checksum: 'mock-checksum',
        created_by: 'user',
        used_by_count: 0,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        description: newAsset.description
      };
      assets = [mockAsset, ...assets];
      applyFilters();
      addAssetModalOpen = false;
      resetNewAssetForm();
    }
  }

  function resetNewAssetForm() {
    newAsset = {
      name: '',
      category: 'Indicator',
      code: '',
      description: '',
      dependencies: []
    };
  }

  function viewHistory(asset: SharedAsset) {
    selectedAsset = asset;
    // Mock history data
    assetHistory = [
      {
        version: asset.version,
        checksum: asset.checksum,
        created_at: asset.updated_at,
        created_by: asset.created_by,
        change_description: 'Current version'
      },
      {
        version: '1.1.0',
        checksum: 'old-checksum-1',
        created_at: '2024-01-10T10:00:00Z',
        created_by: 'QuantCode',
        change_description: 'Fixed bug in calculation'
      },
      {
        version: '1.0.0',
        checksum: 'old-checksum-2',
        created_at: '2024-01-05T09:00:00Z',
        created_by: 'user',
        change_description: 'Initial release'
      }
    ];
    historyModalOpen = true;
  }

  async function rollbackToVersion(version: string) {
    if (!selectedAsset || !confirm(`Roll back "${selectedAsset.name}" to version ${version}?`)) return;

    try {
      const res = await fetch(`http://localhost:8000/api/assets/${selectedAsset.id}/rollback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ version })
      });
      if (res.ok) {
        await loadAssets();
        historyModalOpen = false;
      }
    } catch (e) {
      console.error('Failed to rollback:', e);
    }
  }

  function getCategoryColor(category: string) {
    const colors: Record<string, string> = {
      'Indicator': '#3b82f6',
      'Risk': '#f59e0b',
      'Utils': '#8b5cf6'
    };
    return colors[category] || '#6b7280';
  }

  function getCategoryBgColor(category: string) {
    const colors: Record<string, string> = {
      'Indicator': 'rgba(59, 130, 246, 0.1)',
      'Risk': 'rgba(245, 158, 11, 0.1)',
      'Utils': 'rgba(139, 92, 246, 0.1)'
    };
    return colors[category] || 'rgba(107, 114, 128, 0.1)';
  }

  function openInEditor(asset: SharedAsset) {
    // Create a mock file object for the editor
    const fileForEditor = {
      id: asset.id,
      name: asset.name,
      category: `assets/${asset.category.toLowerCase()}`,
      path: asset.filesystem_path,
      content: `// ${asset.name}\n// Category: ${asset.category}\n// Version: ${asset.version}\n// Path: ${asset.filesystem_path}\n\n// Asset code would be loaded here\n\n// This is a ${asset.category} created by ${asset.created_by}\n// Created: ${formatDate(asset.created_at)}\n// Last Updated: ${formatDate(asset.updated_at)}\n\n${asset.description ? `// Description: ${asset.description}\n\n` : ''}// Dependencies:\n${asset.dependencies.map(dep => `// - ${dep}`).join('\n')}\n\n// Asset implementation would go here...`,
      language: asset.filesystem_path.endsWith('.mq5') ? 'cpp' : 'plaintext'
    };
    
    // Dispatch event to open in editor
    dispatch('openInEditor', { article: fileForEditor });
  }

  
  function formatDate(dateStr: string) {
    const date = new Date(dateStr);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  function canWriteAsset(asset: SharedAsset): boolean {
    // QuantCode can always write
    if (asset.created_by === 'QuantCode' && permissionSettings.quantCodeWrite) return true;
    // Users can always write their own assets
    if (asset.created_by === 'user' && permissionSettings.userWrite) return true;
    return false;
  }
</script>

<div class="assets-view">
  <!-- Header -->
  <div class="assets-header">
    <div class="header-left">
      <FolderOpen size={24} class="assets-icon" />
      <div>
        <h2>Shared Assets & Database</h2>
        <p>Reusable MQL5 components and database visualization</p>
      </div>
    </div>
    <div class="header-actions">
      <button class="btn" on:click={loadAssets}>
        <RefreshCw size={14} />
        <span>Refresh</span>
      </button>
      <button class="btn" on:click={() => permissionSettings.copilotWrite = !permissionSettings.copilotWrite}>
        <SettingsIcon size={14} />
        <span>{permissionSettings.copilotWrite ? 'Copilot: ON' : 'Copilot: OFF'}</span>
      </button>
      {#if activeTab === 'assets'}
        <button class="btn primary" on:click={() => addAssetModalOpen = true}>
          <Plus size={14} />
          <span>Add Asset</span>
        </button>
      {/if}
    </div>
  </div>

  <!-- Tabs -->
  <div class="tabs-bar">
    <button class="tab-btn" class:active={activeTab === 'assets'} on:click={() => activeTab = 'assets'}>
      <Package size={14} />
      <span>Shared Assets</span>
    </button>
    <button class="tab-btn" class:active={activeTab === 'database'} on:click={() => activeTab = 'database'}>
      <Database size={14} />
      <span>Database Manager</span>
    </button>
  </div>

  {#if activeTab === 'database'}
    <!-- Database View -->
    <div class="database-view-container">
      <DatabaseView />
    </div>
  {:else}

  <!-- Permission Banner -->
  <div class="permission-banner">
    <div class="permission-info">
      <Shield size={16} />
      <div class="permission-text">
        <strong>Permissions:</strong>
        <span class="perm-read">READ</span> - QuantCode Agent, User (Copilot), Any EA
        <span class="perm-write">WRITE</span> - QuantCode Agent {permissionSettings.quantCodeWrite ? '(enabled)' : '(disabled)'},
        User {permissionSettings.userWrite ? '(enabled)' : '(disabled)'}
        {permissionSettings.copilotWrite ? ', Copilot (delegated)' : ''}
      </div>
    </div>
  </div>

  <!-- Filters -->
  <div class="filters-bar">
    <div class="filter-group">
      <Search size={14} />
      <input
        type="text"
        placeholder="Search by name, description, path..."
        bind:value={searchQuery}
        on:input={applyFilters}
      />
    </div>

    <div class="filter-group">
      <Filter size={14} />
      <select bind:value={categoryFilter} on:change={applyFilters}>
        <option value="all">All Categories</option>
        <option value="Indicator">Indicators</option>
        <option value="Risk">Risk</option>
        <option value="Utils">Utilities</option>
      </select>
    </div>

    <div class="stats-summary">
      <span class="stat-item">{filteredAssets.length} assets</span>
      <span class="stat-item">{assets.filter(a => a.created_by === 'QuantCode').length} by QuantCode</span>
      <span class="stat-item">{assets.filter(a => a.created_by === 'user').length} by User</span>
    </div>
  </div>

  <!-- Assets Table -->
  <div class="assets-table-container">
    <div class="table-header">
      <div class="header-cell sortable">Name</div>
      <div class="header-cell">Category</div>
      <div class="header-cell">Version</div>
      <div class="header-cell">Dependencies</div>
      <div class="header-cell">Used By</div>
      <div class="header-cell">Checksum</div>
      <div class="header-cell">Created By</div>
      <div class="header-cell"></div>
    </div>

    <div class="table-body">
      {#each filteredAssets as asset}
        <div class="table-row" class:expanded={expandedAsset === asset.id}>
          <div class="cell name">
            <div class="name-header">
              <FileCode size={14} class="name-icon" />
              <span class="name-text">{asset.name}</span>
            </div>
            {#if asset.description}
              <span class="description">{asset.description}</span>
            {/if}
          </div>

          <div class="cell category">
            <span
              class="category-badge"
              style="background: {getCategoryBgColor(asset.category)}; color: {getCategoryColor(asset.category)}"
            >
              {asset.category}
            </span>
          </div>

          <div class="cell version">
            <span class="version-text">{asset.version}</span>
          </div>

          <div class="cell dependencies">
            {#if asset.dependencies.length > 0}
              <div class="dep-list">
                {#each asset.dependencies.slice(0, 2) as dep}
                  <span class="dep-tag">{dep}</span>
                {/each}
                {#if asset.dependencies.length > 2}
                  <span class="dep-more">+{asset.dependencies.length - 2}</span>
                {/if}
              </div>
            {:else}
              <span class="no-deps">None</span>
            {/if}
          </div>

          <div class="cell used-by">
            <span class="used-count">{asset.used_by_count}</span>
            <span class="used-label">bots</span>
          </div>

          <div class="cell checksum">
            <span class="checksum-text" title={asset.checksum}>{asset.checksum}</span>
          </div>

          <div class="cell created-by">
            <span class="creator-badge" class:quantcode={asset.created_by === 'QuantCode'} class:user={asset.created_by === 'user'}>
              {asset.created_by === 'QuantCode' ? 'AI' : 'User'}
            </span>
          </div>

          <div class="cell actions">
            <button
              class="icon-btn"
              on:click={() => toggleExpanded(asset.id)}
              title="Toggle details"
            >
              {#if expandedAsset === asset.id}
                <ChevronUp size={14} />
              {:else}
                <ChevronDown size={14} />
              {/if}
            </button>
            <button class="icon-btn" on:click={() => viewAsset(asset)} title="View details">
              <Eye size={14} />
            </button>
            <button
              class="icon-btn"
              on:click={() => editAsset(asset)}
              title="Edit asset"
              disabled={!canWriteAsset(asset)}
              class:disabled={!canWriteAsset(asset)}
            >
              <Edit size={14} />
            </button>
            <button
              class="icon-btn danger"
              on:click={() => deleteAsset(asset)}
              title="Delete asset"
              disabled={!canWriteAsset(asset)}
              class:disabled={!canWriteAsset(asset)}
            >
              <Trash2 size={14} />
            </button>
          </div>
        </div>

        <!-- Expanded Row -->
        {#if expandedAsset === asset.id}
          <div class="expanded-row">
            <div class="expanded-content">
              <!-- File Path -->
              <div class="detail-section">
                <h4><FolderOpen size={14} /> File Path</h4>
                <code class="path-code">{asset.filesystem_path}</code>
              </div>

              <!-- Full Dependencies -->
              {#if asset.dependencies.length > 0}
                <div class="detail-section">
                  <h4><Package size={14} /> Dependencies</h4>
                  <div class="dep-tags">
                    {#each asset.dependencies as dep}
                      <span class="dep-tag-full">{dep}</span>
                    {/each}
                  </div>
                </div>
              {/if}

              <!-- Timestamps -->
              <div class="detail-section">
                <h4><Clock size={14} /> Timestamps</h4>
                <div class="timestamp-grid">
                  <div class="timestamp-item">
                    <span class="label">Created</span>
                    <span class="value">{formatDate(asset.created_at)}</span>
                  </div>
                  <div class="timestamp-item">
                    <span class="label">Updated</span>
                    <span class="value">{formatDate(asset.updated_at)}</span>
                  </div>
                </div>
              </div>

              <!-- Quick Actions -->
              <div class="detail-section">
                <h4><Zap size={14} /> Quick Actions</h4>
                <div class="quick-actions">
                  <button class="action-btn" on:click={() => viewHistory(asset)}>
                    <History size={14} />
                    <span>View History</span>
                  </button>
                  <button class="action-btn">
                    <Download size={14} />
                    <span>Export</span>
                  </button>
                  <button class="action-btn" on:click={() => openInEditor(asset)}>
                    <Edit size={14} />
                    <span>Open in Editor</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        {/if}
      {/each}

      {#if filteredAssets.length === 0}
        <div class="empty-state">
          <Package size={32} />
          <p>No shared assets found</p>
          <button class="btn primary" on:click={() => addAssetModalOpen = true}>
            <Plus size={14} />
            <span>Create First Asset</span>
          </button>
        </div>
      {/if}
    </div>
  </div>
{/if}

<!-- Asset Detail Modal -->
  {#if detailViewOpen && selectedAsset}
    <div class="modal-overlay" on:click|self={() => detailViewOpen = false} role="dialog" aria-modal="true" aria-labelledby="detail-modal-title">
      <div class="modal large">
        <div class="modal-header">
          <div>
            <h3 id="detail-modal-title">{selectedAsset.name}</h3>
            <p class="modal-subtitle">
              <span
                class="category-badge"
                style="background: {getCategoryBgColor(selectedAsset.category)}; color: {getCategoryColor(selectedAsset.category)}"
              >
                {selectedAsset.category}
              </span>
              <span class="version">v{selectedAsset.version}</span>
            </p>
          </div>
          <button class="icon-btn" on:click={() => detailViewOpen = false}>
            <X size={18} />
          </button>
        </div>

        <div class="modal-content">
          <!-- Description -->
          {#if selectedAsset.description}
            <div class="detail-group">
              <h4>Description</h4>
              <p>{selectedAsset.description}</p>
            </div>
          {/if}

          <!-- Metadata -->
          <div class="detail-group">
            <h4>Metadata</h4>
            <div class="metadata-grid">
              <div class="meta-item">
                <span class="label">File Path</span>
                <code class="value">{selectedAsset.filesystem_path}</code>
              </div>
              <div class="meta-item">
                <span class="label">Checksum</span>
                <code class="value">{selectedAsset.checksum}</code>
              </div>
              <div class="meta-item">
                <span class="label">Created By</span>
                <span class="value">{selectedAsset.created_by}</span>
              </div>
              <div class="meta-item">
                <span class="label">Used By</span>
                <span class="value">{selectedAsset.used_by_count} bots</span>
              </div>
            </div>
          </div>

          <!-- Dependencies Graph -->
          {#if selectedAsset.dependencies.length > 0}
            <div class="detail-group">
              <h4>Dependency Graph</h4>
              <div class="dependency-graph">
                <div class="graph-node current">
                  <span>{selectedAsset.name}</span>
                </div>
                {#each selectedAsset.dependencies as dep}
                  <div class="graph-dep-line"></div>
                  <div class="graph-node dependency">
                    <span>{dep}</span>
                  </div>
                {/each}
              </div>
            </div>
          {/if}

          <!-- Usage Statistics -->
          <div class="detail-group">
            <h4>Usage Statistics</h4>
            <div class="usage-stats">
              <div class="stat-card">
                <span class="stat-value">{selectedAsset.used_by_count}</span>
                <span class="stat-label">Active Bots</span>
              </div>
              <div class="stat-card">
                <span class="stat-value">{selectedAsset.version}</span>
                <span class="stat-label">Current Version</span>
              </div>
              <div class="stat-card">
                <span class="stat-value">{selectedAsset.dependencies.length}</span>
                <span class="stat-label">Dependencies</span>
              </div>
            </div>
          </div>

          <!-- Actions -->
          <div class="detail-actions">
            <button class="btn" on:click={() => selectedAsset && viewHistory(selectedAsset)}>
              <GitBranch size={14} />
              <span>Version History</span>
            </button>
            <button class="btn">
              <Download size={14} />
              <span>Export Asset</span>
            </button>
            {#if selectedAsset && canWriteAsset(selectedAsset)}
              <button class="btn" on:click={() => editAsset(selectedAsset)}>
                <Edit size={14} />
                <span>Edit Asset</span>
              </button>
            {/if}
          </div>
        </div>
      </div>
    </div>
  {/if}

  <!-- Add Asset Modal -->
  {#if addAssetModalOpen}
    <div class="modal-overlay" on:click|self={() => addAssetModalOpen = false} on:keydown={(e) => e.key === 'Escape' && (addAssetModalOpen = false)} role="dialog" aria-modal="true" aria-labelledby="add-modal-title">
      <div class="modal">
        <div class="modal-header">
          <div>
            <h3 id="add-modal-title">Add New Shared Asset</h3>
            <p class="modal-subtitle">Create a reusable component for the library</p>
          </div>
          <button class="icon-btn" on:click={() => addAssetModalOpen = false}>
            <X size={18} />
          </button>
        </div>

        <div class="modal-content">
          <div class="form-group">
            <label for="asset-name-input">Asset Name</label>
            <input id="asset-name-input" type="text" bind:value={newAsset.name} placeholder="e.g., AdaptiveRSI" aria-required="true" />
          </div>

          <div class="form-group">
            <label for="asset-category-select">Category</label>
            <select id="asset-category-select" bind:value={newAsset.category}>
              <option value="Indicator">Indicator</option>
              <option value="Risk">Risk</option>
              <option value="Utils">Utilities</option>
            </select>
          </div>

          <div class="form-group">
            <label for="asset-description-textarea">Description</label>
            <textarea id="asset-description-textarea" bind:value={newAsset.description} placeholder="Brief description of the asset..."></textarea>
          </div>

          <div class="form-group">
            <label for="asset-code-textarea">Source Code</label>
            <textarea
              id="asset-code-textarea"
              bind:value={newAsset.code}
              class="code-editor"
              placeholder="// Paste your MQL5 code here..."
              rows="10"
            ></textarea>
          </div>

          <div class="form-group">
            <label for="asset-dependencies-input">Dependencies (comma-separated)</label>
            <input
              id="asset-dependencies-input"
              type="text"
              placeholder="e.g., MovingAverage, ATR, Statistics"
              on:change={(e) => {
                const target = e.target;
                newAsset.dependencies = target.value.split(',').map(s => s.trim()).filter(s => s);
              }}
            />
          </div>
        </div>

        <div class="modal-actions">
          <button class="btn" on:click={() => addAssetModalOpen = false}>Cancel</button>
          <button class="btn primary" on:click={createAsset}>
            <Plus size={14} />
            <span>Create Asset</span>
          </button>
        </div>
      </div>
    </div>
  {/if}

  <!-- History Modal -->
  {#if historyModalOpen && selectedAsset}
    <div class="modal-overlay" on:click|self={() => historyModalOpen = false} on:keydown={(e) => e.key === 'Escape' && (historyModalOpen = false)} role="dialog" aria-modal="true" aria-labelledby="history-modal-title">
      <div class="modal">
        <div class="modal-header">
          <div>
            <h3 id="history-modal-title">Version History</h3>
            <p class="modal-subtitle">{selectedAsset.name}</p>
          </div>
          <button class="icon-btn" on:click={() => historyModalOpen = false}>
            <X size={18} />
          </button>
        </div>

        <div class="modal-content">
          <div class="history-list">
            {#each assetHistory as history, index}
              <div class="history-item" class:current={index === 0}>
                <div class="history-header">
                  <span class="history-version">v{history.version}</span>
                  <span class="history-date">{formatDate(history.created_at)}</span>
                </div>
                <div class="history-details">
                  <span class="history-author">By {history.created_by}</span>
                  <span class="history-checksum" title={history.checksum}>
                    {history.checksum.substring(0, 8)}...
                  </span>
                </div>
                <p class="history-description">{history.change_description}</p>
                {#if index > 0 && canWriteAsset(selectedAsset)}
                  <button class="rollback-btn" on:click={() => rollbackToVersion(history.version)}>
                    <History size={12} />
                    <span>Rollback</span>
                  </button>
                {/if}
              </div>
            {/each}
          </div>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .assets-view {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary);
    overflow: hidden;
  }

  /* Tabs Bar */
  .tabs-bar {
    display: flex;
    gap: 4px;
    padding: 8px 24px 0;
    border-bottom: 1px solid var(--border-subtle);
    background: var(--bg-secondary);
  }

  .tab-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    background: transparent;
    border: none;
    border-radius: 8px 8px 0 0;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s ease;
    font-size: 13px;
    margin-bottom: -1px;
  }

  .tab-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .tab-btn.active {
    background: var(--bg-primary);
    color: var(--accent-primary);
    border-bottom: 2px solid var(--accent-primary);
  }

  .database-view-container {
    flex: 1;
    overflow-y: auto;
    background: var(--bg-primary);
  }

  /* Header */
  .assets-header {
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

  .assets-icon {
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
    transition: all 0.15s;
  }

  .btn:hover {
    background: var(--bg-surface);
  }

  .btn.primary {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: var(--bg-primary);
  }

  .btn.primary:hover {
    opacity: 0.9;
  }

  /* Permission Banner */
  .permission-banner {
    padding: 12px 24px;
    background: var(--bg-surface);
    border-bottom: 1px solid var(--border-subtle);
  }

  .permission-info {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .permission-text {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }

  .perm-read {
    padding: 2px 6px;
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
  }

  .perm-write {
    padding: 2px 6px;
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
  }

  /* Filters */
  .filters-bar {
    display: flex;
    gap: 12px;
    padding: 16px 24px;
    background: var(--bg-primary);
    border-bottom: 1px solid var(--border-subtle);
    flex-wrap: wrap;
    align-items: center;
  }

  .filter-group {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    font-size: 12px;
  }

  .filter-group input,
  .filter-group select {
    background: transparent;
    border: none;
    color: var(--text-primary);
    font-size: 12px;
    outline: none;
  }

  .filter-group input[type="text"] {
    width: 250px;
  }

  .stats-summary {
    margin-left: auto;
    display: flex;
    gap: 16px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .stat-item {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  /* Table */
  .assets-table-container {
    flex: 1;
    overflow-y: auto;
  }

  .table-header {
    display: grid;
    grid-template-columns: 2fr 100px 80px 150px 80px 120px 100px 120px;
    gap: 12px;
    padding: 12px 24px;
    background: var(--bg-tertiary);
    font-size: 11px;
    font-weight: 500;
    color: var(--text-muted);
    position: sticky;
    top: 0;
    z-index: 10;
  }

  .header-cell {
    display: flex;
    align-items: center;
  }

  .table-row {
    display: grid;
    grid-template-columns: 2fr 100px 80px 150px 80px 120px 100px 120px;
    gap: 12px;
    padding: 14px 24px;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 12px;
    transition: background 0.15s;
    align-items: start;
  }

  .table-row:hover {
    background: var(--bg-secondary);
  }

  .table-row.expanded {
    background: var(--bg-secondary);
    border-bottom: none;
  }

  .cell {
    display: flex;
    flex-direction: column;
    color: var(--text-primary);
  }

  .cell.name {
    gap: 4px;
  }

  .name-header {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .name-icon {
    color: var(--text-muted);
  }

  .name-text {
    font-weight: 500;
    color: var(--text-primary);
  }

  .description {
    font-size: 11px;
    color: var(--text-muted);
    line-height: 1.4;
  }

  .category-badge {
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
  }

  .version-text {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
  }

  .dep-list {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }

  .dep-tag {
    padding: 2px 6px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    font-size: 10px;
    color: var(--text-secondary);
  }

  .dep-more {
    padding: 2px 6px;
    background: var(--bg-surface);
    border-radius: 4px;
    font-size: 10px;
    color: var(--text-muted);
  }

  .no-deps {
    color: var(--text-muted);
    font-size: 11px;
  }

  .used-count {
    font-weight: 600;
    color: var(--text-primary);
  }

  .used-label {
    font-size: 10px;
    color: var(--text-muted);
  }

  .checksum-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--text-muted);
    max-width: 100px;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .creator-badge {
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
  }

  .creator-badge.quantcode {
    background: rgba(139, 92, 246, 0.2);
    color: #8b5cf6;
  }

  .creator-badge.user {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .cell.actions {
    flex-direction: row;
    align-items: center;
    gap: 4px;
  }

  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: transparent;
    border: none;
    border-radius: 4px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s;
  }

  .icon-btn:hover:not(.disabled) {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .icon-btn.danger:hover:not(.disabled) {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .icon-btn.disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  /* Expanded Row */
  .expanded-row {
    grid-column: 1 / -1;
    background: var(--bg-surface);
    border-bottom: 1px solid var(--border-subtle);
  }

  .expanded-content {
    padding: 16px 24px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
  }

  .detail-section {
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 12px;
  }

  .detail-section h4 {
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 0 0 10px;
    font-size: 12px;
    color: var(--text-muted);
  }

  .path-code {
    display: block;
    padding: 8px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-primary);
    word-break: break-all;
  }

  .dep-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }

  .dep-tag-full {
    padding: 4px 10px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    font-size: 11px;
    color: var(--text-secondary);
  }

  .timestamp-grid {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .timestamp-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .timestamp-item .label {
    font-size: 10px;
    color: var(--text-muted);
  }

  .timestamp-item .value {
    font-size: 11px;
    color: var(--text-primary);
  }

  .quick-actions {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .action-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
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

  /* Empty State */
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

  .empty-state p {
    margin: 0;
    font-size: 13px;
  }

  /* Modal */
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
    border-radius: 12px;
    width: 600px;
    max-width: 90%;
    max-height: 85vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .modal.large {
    width: 800px;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .modal-header h3 {
    margin: 0;
    font-size: 16px;
    color: var(--text-primary);
  }

  .modal-subtitle {
    margin: 4px 0 0;
    font-size: 11px;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .modal-subtitle .version {
    font-family: 'JetBrains Mono', monospace;
  }

  .modal-content {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
  }

  .detail-group {
    margin-bottom: 20px;
  }

  .detail-group h4 {
    margin: 0 0 10px;
    font-size: 13px;
    color: var(--text-primary);
  }

  .detail-group p {
    margin: 0;
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.5;
  }

  .metadata-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
  }

  .meta-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 10px;
    background: var(--bg-tertiary);
    border-radius: 6px;
  }

  .meta-item .label {
    font-size: 10px;
    color: var(--text-muted);
    text-transform: uppercase;
  }

  .meta-item .value {
    font-size: 12px;
    color: var(--text-primary);
  }

  .meta-item code.value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    word-break: break-all;
  }

  .dependency-graph {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 16px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .graph-node {
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 500;
  }

  .graph-node.current {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  .graph-node.dependency {
    background: var(--bg-surface);
    color: var(--text-primary);
  }

  .graph-dep-line {
    width: 2px;
    height: 12px;
    background: var(--border-subtle);
  }

  .usage-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }

  .stat-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 16px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .stat-card .stat-value {
    font-size: 24px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .stat-card .stat-label {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
  }

  .detail-actions {
    display: flex;
    gap: 8px;
    padding-top: 16px;
    border-top: 1px solid var(--border-subtle);
  }

  .detail-actions .btn {
    flex: 1;
    justify-content: center;
  }

  /* Form */
  .form-group {
    margin-bottom: 16px;
  }

  .form-group label {
    display: block;
    margin-bottom: 6px;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .form-group input,
  .form-group select,
  .form-group textarea {
    width: 100%;
    padding: 10px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
    outline: none;
    transition: border-color 0.15s;
  }

  .form-group input:focus,
  .form-group select:focus,
  .form-group textarea:focus {
    border-color: var(--accent-primary);
  }

  .form-group textarea.code-editor {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    line-height: 1.6;
  }

  .modal-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    padding-top: 16px;
    border-top: 1px solid var(--border-subtle);
  }

  /* History List */
  .history-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .history-item {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
    border: 1px solid var(--border-subtle);
    position: relative;
  }

  .history-item.current {
    border-color: var(--accent-primary);
    background: rgba(99, 102, 241, 0.1);
  }

  .history-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .history-version {
    font-weight: 600;
    color: var(--text-primary);
  }

  .history-date {
    font-size: 11px;
    color: var(--text-muted);
  }

  .history-details {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
  }

  .history-author {
    color: var(--text-muted);
  }

  .history-checksum {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-muted);
  }

  .history-description {
    margin: 0;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .rollback-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 6px 12px;
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 11px;
    cursor: pointer;
    transition: all 0.15s;
    align-self: flex-start;
  }

  .rollback-btn:hover {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: var(--bg-primary);
  }
</style>

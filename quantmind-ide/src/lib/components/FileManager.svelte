<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import {
    Folder, FolderOpen, File, FileCode, ChevronRight, ChevronDown,
    Search, Grid, List, Plus, Trash2, Edit2, Copy, Download,
    Upload, RefreshCw, Home, ArrowLeft, ArrowRight, Star, Clock,
    Eye, MoreVertical, FileText, Image, Music, Video, Archive, X
  } from 'lucide-svelte';
  import Pagination from './Pagination.svelte';
  import FileViewer from './FileViewer.svelte';
  import FileEditor from './FileEditor.svelte';

  const dispatch = createEventDispatcher();

  // File/Folder types
  interface FileNode {
    id: string;
    name: string;
    type: 'folder' | 'file';
    path: string;
    children?: FileNode[];
    parentId?: string;
    expanded?: boolean;
    selected?: boolean;
    metadata?: {
      size?: number;
      created_at?: string;
      updated_at?: string;
      checksum?: string;
      content?: string;
    };
  }

  // State
  let fileTree: FileNode[] = [];
  let currentPath = '/';
  let selectedFile: FileNode | null = null;
  let selectedFileContent: string | null = null;
  
  // View options
  let viewMode: 'tree' | 'list' = 'tree';
  let showHiddenFiles = false;
  let sortBy: 'name' | 'date' | 'size' | 'type' = 'name';
  let sortOrder: 'asc' | 'desc' = 'asc';
  let searchQuery = '';
  let isLoading = false;
  
  // Pagination
  let currentPage = 1;
  let itemsPerPage = 50;
  
  // Preview/Edit
  let showPreview = false;
  let showEditor = false;
  let isEditing = false;

  const API_BASE = 'http://localhost:8000/api';

  // Icons for different file types
  const fileIcons: Record<string, any> = {
    mq5: FileCode,
    mqh: FileCode,
    py: FileCode,
    ts: FileCode,
    js: FileCode,
    json: FileText,
    md: FileText,
    png: Image,
    jpg: Image,
    jpeg: Image,
    gif: Image,
    svg: Image,
    mp3: Music,
    wav: Music,
    mp4: Video,
    avi: Video,
    zip: Archive,
    default: File
  };

  onMount(() => {
    loadFileTree();
  });

  async function loadFileTree() {
    isLoading = true;
    try {
      const response = await fetch(`${API_BASE}/files/tree`);
      if (response.ok) {
        fileTree = await response.json();
        expandToPath(currentPath);
      }
    } catch (e) {
      console.error('Failed to load file tree:', e);
      loadMockFileTree();
    }
    isLoading = false;
  }

  function loadMockFileTree() {
    fileTree = [
      {
        id: 'root-mql5',
        name: 'mql5',
        type: 'folder',
        path: '/mql5',
        expanded: true,
        children: [
          {
            id: 'mql5-experts',
            name: 'Experts',
            type: 'folder',
            path: '/mql5/Experts',
            children: [
              {
                id: 'ea-sma-cross',
                name: 'SMA_Cross.mq5',
                type: 'file',
                path: '/mql5/Experts/SMA_Cross.mq5',
                metadata: {
                  size: 12456,
                  created_at: '2024-01-10T10:00:00Z',
                  updated_at: '2024-01-15T14:30:00Z'
                }
              },
              {
                id: 'ea-rsi-oscillator',
                name: 'RSI_Oscillator.mq5',
                type: 'file',
                path: '/mql5/Experts/RSI_Oscillator.mq5',
                metadata: {
                  size: 15678,
                  created_at: '2024-01-12T09:00:00Z',
                  updated_at: '2024-01-18T16:45:00Z'
                }
              }
            ]
          },
          {
            id: 'mql5-indicators',
            name: 'Indicators',
            type: 'folder',
            path: '/mql5/Indicators',
            children: [
              {
                id: 'ind-adaptive-rsi',
                name: 'AdaptiveRSI.mqh',
                type: 'file',
                path: '/mql5/Indicators/AdaptiveRSI.mqh',
                metadata: {
                  size: 4567,
                  created_at: '2024-01-05T11:00:00Z',
                  updated_at: '2024-01-20T10:15:00Z'
                }
              }
            ]
          },
          {
            id: 'mql5-scripts',
            name: 'Scripts',
            type: 'folder',
            path: '/mql5/Scripts',
            children: []
          }
        ]
      },
      {
        id: 'root-python',
        name: 'python',
        type: 'folder',
        path: '/python',
        children: [
          {
            id: 'py-strategies',
            name: 'strategies',
            type: 'folder',
            path: '/python/strategies',
            children: [
              {
                id: 'py-mean-reversion',
                name: 'mean_reversion.py',
                type: 'file',
                path: '/python/strategies/mean_reversion.py',
                metadata: {
                  size: 8901,
                  created_at: '2024-01-08T14:00:00Z',
                  updated_at: '2024-01-19T11:30:00Z'
                }
              }
            ]
          }
        ]
      },
      {
        id: 'root-data',
        name: 'data',
        type: 'folder',
        path: '/data',
        children: []
      },
      {
        id: 'root-config',
        name: 'config',
        type: 'folder',
        path: '/config',
        children: []
      }
    ];
  }

  function expandToPath(path: string) {
    const parts = path.split('/').filter(p => p);
    let currentLevel = fileTree;
    
    for (const part of parts) {
      const node = currentLevel.find(n => n.name === part && n.type === 'folder');
      if (node) {
        node.expanded = true;
        currentLevel = node.children || [];
      }
    }
  }

  function getIcon(node: FileNode) {
    if (node.type === 'folder') {
      return node.expanded ? FolderOpen : Folder;
    }
    const ext = node.name.split('.').pop()?.toLowerCase() || '';
    return fileIcons[ext] || fileIcons.default;
  }

  function toggleExpand(node: FileNode) {
    if (node.type === 'folder') {
      node.expanded = !node.expanded;
    }
  }

  function selectNode(node: FileNode) {
    selectedFile = node;
    if (node.type === 'file') {
      loadFileContent(node);
    }
  }

  async function loadFileContent(node: FileNode) {
    if (!node.metadata?.content) {
      try {
        const response = await fetch(`${API_BASE}/files/${node.id}/content`);
        if (response.ok) {
          const data = await response.json();
          node.metadata = { ...node.metadata, content: data.content };
          selectedFileContent = data.content;
        }
      } catch (e) {
        console.error('Failed to load file content:', e);
        selectedFileContent = `// Failed to load content for ${node.name}`;
      }
    } else {
      selectedFileContent = node.metadata.content;
    }
  }

  function navigateUp() {
    const parts = currentPath.split('/').filter(p => p);
    if (parts.length > 0) {
      parts.pop();
      currentPath = '/' + parts.join('/');
      expandToPath(currentPath);
    }
  }

  function navigateToFolder(path: string) {
    currentPath = path;
    expandToPath(path);
  }

  function getFilteredFiles(): FileNode[] {
    let files: FileNode[] = [];
    
    // Get all files from expanded folders
    function collectFiles(nodes: FileNode[]) {
      for (const node of nodes) {
        if (node.type === 'folder' && node.expanded) {
          if (node.children) collectFiles(node.children);
        } else if (node.type === 'file') {
          files.push(node);
        }
      }
    }
    
    collectFiles(fileTree);
    
    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      files = files.filter(f => f.name.toLowerCase().includes(query));
    }
    
    // Sort files
    files.sort((a, b) => {
      let comparison = 0;
      switch (sortBy) {
        case 'name':
          comparison = a.name.localeCompare(b.name);
          break;
        case 'date':
          comparison = (a.metadata?.updated_at || '').localeCompare(b.metadata?.updated_at || '');
          break;
        case 'size':
          comparison = (a.metadata?.size || 0) - (b.metadata?.size || 0);
          break;
        case 'type':
          const extA = a.name.split('.').pop() || '';
          const extB = b.name.split('.').pop() || '';
          comparison = extA.localeCompare(extB);
          break;
      }
      return sortOrder === 'asc' ? comparison : -comparison;
    });
    
    return files;
  }

  $: filteredFiles = getFilteredFiles();
  $: totalPages = Math.ceil(filteredFiles.length / itemsPerPage);
  $: paginatedFiles = filteredFiles.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage);

  function handlePreview() {
    showPreview = true;
    showEditor = false;
    isEditing = false;
  }

  function handleEdit() {
    showEditor = true;
    showPreview = false;
    isEditing = true;
  }

  function handleSave(event: CustomEvent) {
    if (selectedFile && selectedFileContent) {
      selectedFileContent = event.detail.content;
      selectedFile.metadata = { ...selectedFile.metadata, content: selectedFileContent || undefined };
    }
  }

  function handleFileAction(action: string, file: FileNode) {
    dispatch(action, { file });
  }

  function formatSize(bytes?: number): string {
    if (!bytes) return '-';
    const units = ['B', 'KB', 'MB'];
    let unitIndex = 0;
    let size = bytes;
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  }

  function formatDate(dateStr?: string): string {
    if (!dateStr) return '-';
    const date = new Date(dateStr);
    return date.toLocaleDateString();
  }

  function getBreadcrumbs(): string[] {
    return currentPath.split('/').filter(p => p);
  }
</script>

<div class="file-manager">
  <!-- Toolbar -->
  <div class="fm-toolbar">
    <div class="toolbar-left">
      <button class="toolbar-btn" on:click={navigateUp} disabled={currentPath === '/'}>
        <ArrowLeft size={16} />
      </button>
      <button class="toolbar-btn" on:click={loadFileTree}>
        <Home size={16} />
      </button>
      <div class="path-bar">
        <span class="path-separator">/</span>
        {#each getBreadcrumbs() as part, i}
          <button 
            class="path-segment" 
            on:click={() => navigateToFolder('/' + getBreadcrumbs().slice(0, i + 1).join('/'))}
          >
            {part}
          </button>
          <span class="path-separator">/</span>
        {/each}
      </div>
    </div>
    
    <div class="toolbar-right">
      <div class="search-box">
        <Search size={14} />
        <input 
          type="text" 
          placeholder="Search files..." 
          bind:value={searchQuery}
        />
      </div>
      
      <div class="view-toggle">
        <button 
          class="toggle-btn" 
          class:active={viewMode === 'tree'}
          on:click={() => viewMode = 'tree'}
          title="Tree view"
        >
          <List size={16} />
        </button>
        <button 
          class="toggle-btn" 
          class:active={viewMode === 'list'}
          on:click={() => viewMode = 'list'}
          title="List view"
        >
          <Grid size={16} />
        </button>
      </div>
      
      <select bind:value={sortBy} class="sort-select">
        <option value="name">Name</option>
        <option value="date">Date</option>
        <option value="size">Size</option>
        <option value="type">Type</option>
      </select>
      
      <button class="toolbar-btn" on:click={() => sortOrder = sortOrder === 'asc' ? 'desc' : 'asc'}>
        {#if sortOrder === 'asc'}
          <ChevronDown size={16} class="rotate-180" />
        {:else}
          <ChevronDown size={16} />
        {/if}
      </button>
    </div>
  </div>

  <!-- Main Content -->
  <div class="fm-content">
    <!-- Sidebar - Folder Tree -->
    <div class="fm-sidebar" class:hidden={viewMode === 'list'}>
      <div class="sidebar-header">
        <span>Folders</span>
      </div>
      <div class="folder-tree">
        {#each fileTree as node}
          <div class="tree-node">
            <button 
              class="node-button"
              class:selected={selectedFile?.id === node.id}
              on:click={() => selectNode(node)}
            >
              <svelte:component this={getIcon(node)} size={14} />
              <span class="node-name">{node.name}</span>
            </button>
            {#if node.type === 'folder' && node.expanded && node.children}
              <div class="node-children">
                {#each node.children as child}
                  <div class="tree-node nested">
                    <button 
                      class="node-button"
                      class:selected={selectedFile?.id === child.id}
                      on:click={() => selectNode(child)}
                    >
                      <svelte:component this={getIcon(child)} size={14} />
                      <span class="node-name">{child.name}</span>
                    </button>
                  </div>
                {/each}
              </div>
            {/if}
          </div>
        {/each}
      </div>
    </div>

    <!-- File List -->
    <div class="fm-main" class:full-width={viewMode === 'list'}>
      <div class="file-list-header">
        <div class="col-name">Name</div>
        <div class="col-size">Size</div>
        <div class="col-date">Modified</div>
        <div class="col-actions"></div>
      </div>
      
      <div class="file-list-body">
        {#if isLoading}
          <div class="loading-state">
            <RefreshCw size={24} class="spinning" />
            <span>Loading files...</span>
          </div>
        {:else if filteredFiles.length === 0}
          <div class="empty-state">
            <Folder size={48} />
            <p>No files found</p>
          </div>
        {:else}
          {#each paginatedFiles as file}
            <div 
              class="file-row"
              class:selected={selectedFile?.id === file.id}
              on:click={() => selectNode(file)}
              on:keypress={(e) => e.key === 'Enter' && selectNode(file)}
              role="button"
              tabindex="0"
            >
              <div class="col-name">
                <svelte:component this={getIcon(file)} size={16} />
                <span class="file-name">{file.name}</span>
              </div>
              <div class="col-size">{formatSize(file.metadata?.size)}</div>
              <div class="col-date">{formatDate(file.metadata?.updated_at)}</div>
              <div class="col-actions">
                <button class="action-icon" on:click|stopPropagation={() => handlePreview()} title="Preview">
                  <Eye size={14} />
                </button>
                {#if file.name.match(/\.(mq5|mqh|py|ts|js|json|md|txt)$/i)}
                  <button class="action-icon" on:click|stopPropagation={() => { selectNode(file); handleEdit(); }} title="Edit">
                    <Edit2 size={14} />
                  </button>
                {/if}
              </div>
            </div>
          {/each}
        {/if}
      </div>

      <!-- Pagination -->
      {#if totalPages > 1}
        <Pagination 
          {currentPage}
          totalPages={totalPages}
          on:pageChange={(e) => currentPage = e.detail}
        />
      {/if}
    </div>

    <!-- Preview Panel -->
    {#if selectedFile && (showPreview || showEditor)}
      <div class="fm-preview">
        <div class="preview-header">
          <span class="preview-title">{selectedFile.name}</span>
          <div class="preview-actions">
            {#if showPreview}
              <button class="preview-btn" on:click={handleEdit}>
                <Edit2 size={14} />
                Edit
              </button>
            {/if}
            <button class="close-preview" on:click={() => { showPreview = false; showEditor = false; }}>
              <X size={16} />
            </button>
          </div>
        </div>
        
        {#if showPreview}
          <FileViewer 
            file={{
              id: selectedFile.id,
              name: selectedFile.name,
              path: selectedFile.path,
              content: selectedFileContent || undefined,
              metadata: selectedFile.metadata
            }}
            on:close={() => showPreview = false}
          />
        {:else if showEditor}
          <FileEditor 
            file={{
              id: selectedFile.id,
              name: selectedFile.name,
              path: selectedFile.path,
              content: selectedFileContent || ''
            }}
            on:save={handleSave}
            on:close={() => showEditor = false}
          />
        {/if}
      </div>
    {/if}
  </div>
</div>

<style>
  .file-manager {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary, #1a1b26);
    border-radius: 8px;
    overflow: hidden;
  }

  /* Toolbar */
  .fm-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    background: var(--bg-tertiary, #1f1f2e);
    border-bottom: 1px solid var(--border-subtle, #2d2d3a);
  }

  .toolbar-left, .toolbar-right {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .toolbar-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: transparent;
    border: none;
    color: var(--text-secondary, #a9b1d6);
    cursor: pointer;
    border-radius: 4px;
    transition: all 0.15s ease;
  }

  .toolbar-btn:hover:not(:disabled) {
    background: var(--bg-secondary, #292a3e);
    color: var(--text-primary, #c0caf5);
  }

  .toolbar-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .path-bar {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    background: var(--bg-secondary, #16161e);
    border-radius: 4px;
  }

  .path-segment {
    padding: 2px 6px;
    background: none;
    border: none;
    color: var(--text-primary, #c0caf5);
    cursor: pointer;
    border-radius: 3px;
    font-size: 12px;
  }

  .path-segment:hover {
    background: var(--bg-tertiary, #1f1f2e);
  }

  .path-separator {
    color: var(--text-muted, #565f89);
    font-size: 12px;
  }

  .search-box {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    background: var(--bg-secondary, #16161e);
    border: 1px solid var(--border-subtle, #2d2d3a);
    border-radius: 4px;
    color: var(--text-muted, #565f89);
  }

  .search-box input {
    background: none;
    border: none;
    color: var(--text-primary, #c0caf5);
    font-size: 12px;
    width: 150px;
    outline: none;
  }

  .view-toggle {
    display: flex;
    background: var(--bg-secondary, #16161e);
    border-radius: 4px;
    padding: 2px;
  }

  .toggle-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: none;
    border: none;
    color: var(--text-muted, #565f89);
    cursor: pointer;
    border-radius: 3px;
  }

  .toggle-btn.active {
    background: var(--bg-tertiary, #1f1f2e);
    color: var(--accent-primary, #7aa2f7);
  }

  .sort-select {
    padding: 4px 8px;
    background: var(--bg-secondary, #16161e);
    border: 1px solid var(--border-subtle, #2d2d3a);
    border-radius: 4px;
    color: var(--text-primary, #c0caf5);
    font-size: 12px;
    cursor: pointer;
  }

  /* Main Content */
  .fm-content {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  /* Sidebar */
  .fm-sidebar {
    width: 220px;
    background: var(--bg-secondary, #16161e);
    border-right: 1px solid var(--border-subtle, #2d2d3a);
    overflow-y: auto;
    flex-shrink: 0;
  }

  .fm-sidebar.hidden {
    display: none;
  }

  .sidebar-header {
    padding: 8px 12px;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted, #565f89);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid var(--border-subtle, #2d2d3a);
  }

  .folder-tree {
    padding: 4px;
  }

  .tree-node {
    margin-bottom: 2px;
  }

  .tree-node.nested {
    margin-left: 16px;
  }

  .node-button {
    display: flex;
    align-items: center;
    gap: 6px;
    width: 100%;
    padding: 6px 8px;
    background: none;
    border: none;
    color: var(--text-secondary, #a9b1d6);
    cursor: pointer;
    border-radius: 4px;
    font-size: 12px;
    text-align: left;
    transition: all 0.1s ease;
  }

  .node-button:hover {
    background: var(--bg-tertiary, #1f1f2e);
    color: var(--text-primary, #c0caf5);
  }

  .node-button.selected {
    background: var(--accent-primary, rgba(122, 162, 247, 0.2));
    color: var(--accent-primary, #7aa2f7);
  }

  .node-children {
    margin-left: 16px;
    border-left: 1px solid var(--border-subtle, #2d2d3a);
    padding-left: 4px;
  }

  /* File List */
  .fm-main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .fm-main.full-width {
    flex: 1;
  }

  .file-list-header {
    display: grid;
    grid-template-columns: 1fr 80px 100px 80px;
    gap: 8px;
    padding: 8px 12px;
    background: var(--bg-secondary, #16161e);
    border-bottom: 1px solid var(--border-subtle, #2d2d3a);
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted, #565f89);
    text-transform: uppercase;
  }

  .col-name {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .file-list-body {
    flex: 1;
    overflow-y: auto;
  }

  .file-row {
    display: grid;
    grid-template-columns: 1fr 80px 100px 80px;
    gap: 8px;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border-subtle, #2d2d3a);
    cursor: pointer;
    transition: background 0.1s ease;
  }

  .file-row:hover {
    background: var(--bg-tertiary, #1f1f2e);
  }

  .file-row.selected {
    background: var(--accent-primary, rgba(122, 162, 247, 0.15));
  }

  .file-row .col-name {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-primary, #c0caf5);
    font-size: 13px;
  }

  .file-row .col-size,
  .file-row .col-date {
    color: var(--text-muted, #565f89);
    font-size: 12px;
    display: flex;
    align-items: center;
  }

  .file-row .col-actions {
    display: flex;
    align-items: center;
    gap: 4px;
    opacity: 0;
    transition: opacity 0.15s ease;
  }

  .file-row:hover .col-actions {
    opacity: 1;
  }

  .action-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: none;
    border: none;
    color: var(--text-muted, #565f89);
    cursor: pointer;
    border-radius: 4px;
  }

  .action-icon:hover {
    background: var(--bg-secondary, #292a3e);
    color: var(--accent-primary, #7aa2f7);
  }

  /* States */
  .loading-state, .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 200px;
    color: var(--text-muted, #565f89);
    gap: 12px;
  }

  .spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  /* Preview Panel */
  .fm-preview {
    width: 400px;
    background: var(--bg-primary, #1a1b26);
    border-left: 1px solid var(--border-subtle, #2d2d3a);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
  }

  .preview-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    background: var(--bg-tertiary, #1f1f2e);
    border-bottom: 1px solid var(--border-subtle, #2d2d3a);
  }

  .preview-title {
    font-weight: 600;
    color: var(--text-primary, #c0caf5);
    font-size: 13px;
  }

  .preview-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .preview-btn {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    background: var(--accent-primary, #7aa2f7);
    color: var(--bg-primary, #1a1b26);
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 11px;
    font-weight: 600;
  }

  .preview-btn:hover {
    background: var(--accent-secondary, #bb9af7);
  }

  .close-preview {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: none;
    border: none;
    color: var(--text-muted, #565f89);
    cursor: pointer;
    border-radius: 4px;
  }

  .close-preview:hover {
    background: #f7768e;
    color: var(--bg-primary, #1a1b26);
  }

  .fm-preview :global(.file-viewer),
  .fm-preview :global(.file-editor) {
    flex: 1;
    border: none;
    border-radius: 0;
  }
</style>

<script lang="ts">
  import { onMount } from 'svelte';
  import {
    Database,
    Search,
    Filter,
    Plus,
    Trash2,
    RefreshCw,
    Brain,
    Clock,
    ChevronDown,
    ChevronRight,
    X,
    Save,
    AlertCircle,
    Network,
    Archive,
    Flame,
    Snowflake,
    Layers,
    Link,
    Sparkles,
    Gauge
  } from 'lucide-svelte';
  import * as graphMemoryApi from '$lib/api/graphMemory';
  import type { GraphMemoryNode, GraphMemoryStats, CompactionStatus } from '$lib/api/graphMemory';

  export let onClose = () => {};

  // State
  let activeTab: 'search' | 'hot' | 'warm' | 'cold' | 'stats' = 'search';
  let searchQuery = '';
  let loading = false;
  let error = '';
  let stats: GraphMemoryStats | null = null;
  let compactionStatus: CompactionStatus | null = null;

  // Search results
  let searchResults: GraphMemoryNode[] = [];

  // Nodes by tier
  let hotNodes: GraphMemoryNode[] = [];
  let warmNodes: GraphMemoryNode[] = [];
  let coldNodes: GraphMemoryNode[] = [];

  // Add memory modal
  let showAddModal = false;
  let newMemory = {
    content: '',
    source: 'user',
    department: '',
    agent_id: '',
    importance: 0.7,
    tags: ''
  };

  // Reflect modal
  let showReflectModal = false;
  let reflectQuery = '';
  let reflectAnswer = '';
  let reflectLoading = false;

  // Selected node for details
  let selectedNode: GraphMemoryNode | null = null;

  // Initialize
  onMount(() => {
    loadStats();
  });

  async function loadStats() {
    loading = true;
    error = '';
    try {
      stats = await graphMemoryApi.getMemoryStats();
      compactionStatus = await graphMemoryApi.getCompactionStatus();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load stats';
    } finally {
      loading = false;
    }
  }

  async function searchMemories() {
    if (!searchQuery.trim()) return;
    loading = true;
    error = '';
    try {
      const result = await graphMemoryApi.searchMemories(searchQuery, 20);
      searchResults = result.nodes;
      activeTab = 'search';
    } catch (e) {
      error = e instanceof Error ? e.message : 'Search failed';
    } finally {
      loading = false;
    }
  }

  async function loadHotNodes() {
    loading = true;
    try {
      hotNodes = await graphMemoryApi.getHotNodes(50);
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load hot nodes';
    } finally {
      loading = false;
    }
  }

  async function loadWarmNodes() {
    loading = true;
    try {
      warmNodes = await graphMemoryApi.getWarmNodes(100);
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load warm nodes';
    } finally {
      loading = false;
    }
  }

  async function loadColdNodes() {
    loading = true;
    try {
      coldNodes = await graphMemoryApi.getColdNodes(100);
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load cold nodes';
    } finally {
      loading = false;
    }
  }

  async function addMemory() {
    if (!newMemory.content.trim()) return;
    loading = true;
    error = '';
    try {
      const tags = newMemory.tags.split(',').map(t => t.trim()).filter(t => t);
      await graphMemoryApi.retainMemory({
        content: newMemory.content,
        source: newMemory.source,
        department: newMemory.department || undefined,
        agent_id: newMemory.agent_id || undefined,
        importance: newMemory.importance,
        tags: tags.length > 0 ? tags : undefined
      });
      showAddModal = false;
      newMemory = { content: '', source: 'user', department: '', agent_id: '', importance: 0.7, tags: '' };
      await loadStats();
      await searchMemories();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to add memory';
    } finally {
      loading = false;
    }
  }

  async function reflect() {
    if (!reflectQuery.trim()) return;
    reflectLoading = true;
    try {
      const result = await graphMemoryApi.reflectOnMemories({ query: reflectQuery });
      reflectAnswer = result.answer;
    } catch (e) {
      error = e instanceof Error ? e.message : 'Reflection failed';
    } finally {
      reflectLoading = false;
    }
  }

  async function triggerCompaction() {
    loading = true;
    try {
      await graphMemoryApi.triggerCompaction();
      await loadStats();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Compaction failed';
    } finally {
      loading = false;
    }
  }

  async function deleteNode(nodeId: string) {
    if (!confirm('Delete this memory?')) return;
    try {
      await graphMemoryApi.deleteMemoryNode(nodeId);
      searchResults = searchResults.filter(n => n.id !== nodeId);
      hotNodes = hotNodes.filter(n => n.id !== nodeId);
      warmNodes = warmNodes.filter(n => n.id !== nodeId);
      coldNodes = coldNodes.filter(n => n.id !== nodeId);
      await loadStats();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Delete failed';
    }
  }

  async function moveNode(nodeId: string, tier: 'hot' | 'warm' | 'cold') {
    try {
      if (tier === 'hot') await graphMemoryApi.moveNodeToHot(nodeId);
      else if (tier === 'warm') await graphMemoryApi.moveNodeToWarm(nodeId);
      else await graphMemoryApi.moveNodeToCold(nodeId);
      await loadStats();
      // Refresh current tab
      if (activeTab === 'hot') await loadHotNodes();
      else if (activeTab === 'warm') await loadWarmNodes();
      else if (activeTab === 'cold') await loadColdNodes();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Move failed';
    }
  }

  function formatDate(dateStr: string): string {
    return new Date(dateStr).toLocaleString();
  }

  function getTierColor(tier: string): string {
    switch (tier) {
      case 'hot': return 'text-orange-500';
      case 'warm': return 'text-blue-500';
      case 'cold': return 'text-slate-500';
      default: return 'text-gray-500';
    }
  }

  function getTierIcon(tier: string) {
    switch (tier) {
      case 'hot': return Flame;
      case 'warm': return Layers;
      case 'cold': return Snowflake;
      default: return Database;
    }
  }

  // Load data when tab changes
  $: if (activeTab === 'hot' && hotNodes.length === 0) loadHotNodes();
  $: if (activeTab === 'warm' && warmNodes.length === 0) loadWarmNodes();
  $: if (activeTab === 'cold' && coldNodes.length === 0) loadColdNodes();
</script>

<div class="flex flex-col h-full bg-gray-900 text-gray-100">
  <!-- Header -->
  <div class="flex items-center justify-between p-4 border-b border-gray-700">
    <div class="flex items-center gap-3">
      <Network class="w-5 h-5 text-cyan-400" />
      <h2 class="text-lg font-semibold">Graph Memory</h2>
    </div>
    <button on:click={onClose} class="p-1 hover:bg-gray-700 rounded">
      <X class="w-5 h-5" />
    </button>
  </div>

  <!-- Tabs -->
  <div class="flex border-b border-gray-700">
    <button
      class="flex-1 px-4 py-2 text-sm font-medium {activeTab === 'search' ? 'text-cyan-400 border-b-2 border-cyan-400' : 'text-gray-400 hover:text-gray-200'}"
      on:click={() => activeTab = 'search'}
    >
      <Search class="w-4 h-4 inline mr-1" /> Search
    </button>
    <button
      class="flex-1 px-4 py-2 text-sm font-medium {activeTab === 'hot' ? 'text-orange-400 border-b-2 border-orange-400' : 'text-gray-400 hover:text-gray-200'}"
      on:click={() => activeTab = 'hot'}
    >
      <Flame class="w-4 h-4 inline mr-1" /> Hot
    </button>
    <button
      class="flex-1 px-4 py-2 text-sm font-medium {activeTab === 'warm' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-400 hover:text-gray-200'}"
      on:click={() => activeTab = 'warm'}
    >
      <Layers class="w-4 h-4 inline mr-1" /> Warm
    </button>
    <button
      class="flex-1 px-4 py-2 text-sm font-medium {activeTab === 'cold' ? 'text-slate-400 border-b-2 border-slate-400' : 'text-gray-400 hover:text-gray-200'}"
      on:click={() => activeTab = 'cold'}
    >
      <Snowflake class="w-4 h-4 inline mr-1" /> Cold
    </button>
    <button
      class="flex-1 px-4 py-2 text-sm font-medium {activeTab === 'stats' ? 'text-purple-400 border-b-2 border-purple-400' : 'text-gray-400 hover:text-gray-200'}"
      on:click={() => { activeTab = 'stats'; loadStats(); }}
    >
      <Gauge class="w-4 h-4 inline mr-1" /> Stats
    </button>
  </div>

  <!-- Content -->
  <div class="flex-1 overflow-auto p-4">
    {#if error}
      <div class="mb-4 p-3 bg-red-900/50 border border-red-700 rounded text-red-200 text-sm">
        {error}
      </div>
    {/if}

    <!-- Search Tab -->
    {#if activeTab === 'search'}
      <div class="space-y-4">
        <div class="flex gap-2">
          <input
            type="text"
            bind:value={searchQuery}
            placeholder="Search memories..."
            class="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm focus:border-cyan-500 focus:outline-none"
            on:keydown={(e) => e.key === 'Enter' && searchMemories()}
          />
          <button
            on:click={searchMemories}
            class="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 rounded text-sm font-medium"
          >
            Search
          </button>
          <button
            on:click={() => { showAddModal = true; showReflectModal = false; }}
            class="px-4 py-2 bg-green-600 hover:bg-green-700 rounded text-sm font-medium"
          >
            <Plus class="w-4 h-4" />
          </button>
        </div>

        <button
          on:click={() => { showReflectModal = true; showAddModal = false; }}
          class="w-full px-4 py-3 bg-purple-600/20 hover:bg-purple-600/30 border border-purple-600/50 rounded-lg flex items-center justify-center gap-2"
        >
          <Sparkles class="w-4 h-4 text-purple-400" />
          <span class="text-purple-300">Reflect on Memories</span>
        </button>

        {#if searchResults.length > 0}
          <div class="space-y-2">
            {#each searchResults as node}
              <div class="p-3 bg-gray-800 rounded-lg border border-gray-700 hover:border-gray-600 cursor-pointer"
                   on:click={() => selectedNode = node}>
                <div class="flex justify-between items-start">
                  <p class="text-sm text-gray-200 line-clamp-2">{node.content}</p>
                  <span class="text-xs {getTierColor(node.tier)}">{node.tier}</span>
                </div>
                <div class="mt-2 flex gap-2 flex-wrap">
                  {#each node.tags as tag}
                    <span class="text-xs px-2 py-0.5 bg-gray-700 rounded">{tag}</span>
                  {/each}
                </div>
                <div class="mt-2 text-xs text-gray-500">
                  {node.agent_id || 'unknown'} • {formatDate(node.created_at)} • {node.importance.toFixed(2)}
                </div>
              </div>
            {/each}
          </div>
        {:else if searchQuery}
          <p class="text-gray-500 text-center py-8">No memories found</p>
        {/if}
      </div>

    <!-- Hot Tab -->
    {:else if activeTab === 'hot'}
      <div class="space-y-2">
        <p class="text-xs text-orange-400 mb-2">Recent memories (&lt; 1 hour)</p>
        {#each hotNodes as node}
          <div class="p-3 bg-gray-800 rounded-lg border border-gray-700 hover:border-gray-600">
            <div class="flex justify-between items-start">
              <p class="text-sm text-gray-200 line-clamp-2">{node.content}</p>
              <button on:click={() => deleteNode(node.id)} class="text-red-400 hover:text-red-300">
                <Trash2 class="w-4 h-4" />
              </button>
            </div>
            <div class="mt-2 flex gap-1">
              <button on:click={() => moveNode(node.id, 'warm')} class="text-xs px-2 py-1 bg-blue-600/50 hover:bg-blue-600 rounded">to Warm</button>
              <button on:click={() => moveNode(node.id, 'cold')} class="text-xs px-2 py-1 bg-slate-600/50 hover:bg-slate-600 rounded">to Cold</button>
            </div>
          </div>
        {/each}
        {#if hotNodes.length === 0}
          <p class="text-gray-500 text-center py-8">No hot memories</p>
        {/if}
      </div>

    <!-- Warm Tab -->
    {:else if activeTab === 'warm'}
      <div class="space-y-2">
        <p class="text-xs text-blue-400 mb-2">Recent memories (&lt; 30 days)</p>
        {#each warmNodes as node}
          <div class="p-3 bg-gray-800 rounded-lg border border-gray-700 hover:border-gray-600">
            <div class="flex justify-between items-start">
              <p class="text-sm text-gray-200 line-clamp-2">{node.content}</p>
              <button on:click={() => deleteNode(node.id)} class="text-red-400 hover:text-red-300">
                <Trash2 class="w-4 h-4" />
              </button>
            </div>
            <div class="mt-2 flex gap-1">
              <button on:click={() => moveNode(node.id, 'hot')} class="text-xs px-2 py-1 bg-orange-600/50 hover:bg-orange-600 rounded">to Hot</button>
              <button on:click={() => moveNode(node.id, 'cold')} class="text-xs px-2 py-1 bg-slate-600/50 hover:bg-slate-600 rounded">to Cold</button>
            </div>
          </div>
        {/each}
        {#if warmNodes.length === 0}
          <p class="text-gray-500 text-center py-8">No warm memories</p>
        {/if}
      </div>

    <!-- Cold Tab -->
    {:else if activeTab === 'cold'}
      <div class="space-y-2">
        <p class="text-xs text-slate-400 mb-2">Archived memories (&gt; 30 days)</p>
        {#each coldNodes as node}
          <div class="p-3 bg-gray-800 rounded-lg border border-gray-700 hover:border-gray-600">
            <div class="flex justify-between items-start">
              <p class="text-sm text-gray-400 line-clamp-2">{node.content}</p>
              <button on:click={() => deleteNode(node.id)} class="text-red-400 hover:text-red-300">
                <Trash2 class="w-4 h-4" />
              </button>
            </div>
            <div class="mt-2 flex gap-1">
              <button on:click={() => moveNode(node.id, 'hot')} class="text-xs px-2 py-1 bg-orange-600/50 hover:bg-orange-600 rounded">to Hot</button>
              <button on:click={() => moveNode(node.id, 'warm')} class="text-xs px-2 py-1 bg-blue-600/50 hover:bg-blue-600 rounded">to Warm</button>
            </div>
          </div>
        {/each}
        {#if coldNodes.length === 0}
          <p class="text-gray-500 text-center py-8">No cold memories</p>
        {/if}
      </div>

    <!-- Stats Tab -->
    {:else if activeTab === 'stats' && stats}
      <div class="space-y-4">
        <!-- Memory counts -->
        <div class="grid grid-cols-4 gap-3">
          <div class="p-4 bg-gray-800 rounded-lg text-center">
            <p class="text-2xl font-bold text-white">{stats.total_nodes}</p>
            <p class="text-xs text-gray-400">Total</p>
          </div>
          <div class="p-4 bg-orange-900/30 rounded-lg text-center border border-orange-700/50">
            <p class="text-2xl font-bold text-orange-400">{stats.hot}</p>
            <p class="text-xs text-orange-400">Hot</p>
          </div>
          <div class="p-4 bg-blue-900/30 rounded-lg text-center border border-blue-700/50">
            <p class="text-2xl font-bold text-blue-400">{stats.warm}</p>
            <p class="text-xs text-blue-400">Warm</p>
          </div>
          <div class="p-4 bg-slate-800 rounded-lg text-center border border-slate-700">
            <p class="text-2xl font-bold text-slate-400">{stats.cold}</p>
            <p class="text-xs text-slate-400">Cold</p>
          </div>
        </div>

        <!-- Compaction status -->
        {#if compactionStatus}
          <div class="p-4 bg-gray-800 rounded-lg border border-gray-700">
            <h3 class="font-medium mb-3">Compaction Status</h3>
            <div class="space-y-2">
              <div class="flex justify-between text-sm">
                <span class="text-gray-400">Context Usage</span>
                <span>{compactionStatus.current_percent.toFixed(1)}%</span>
              </div>
              <div class="w-full bg-gray-700 rounded-full h-2">
                <div class="bg-cyan-500 h-2 rounded-full" style="width: {compactionStatus.current_percent}%"></div>
              </div>
              <div class="flex justify-between text-sm">
                <span class="text-gray-400">Threshold</span>
                <span>{compactionStatus.threshold_percent}%</span>
              </div>
              <div class="flex justify-between text-sm mt-2">
                <span class="text-gray-400">Should Compact</span>
                <span class="{compactionStatus.should_compact ? 'text-red-400' : 'text-green-400'}">
                  {compactionStatus.should_compact ? 'Yes' : 'No'}
                </span>
              </div>
              {#if compactionStatus.should_compact}
                <button
                  on:click={triggerCompaction}
                  class="w-full mt-3 px-4 py-2 bg-cyan-600 hover:bg-cyan-700 rounded text-sm font-medium"
                >
                  Trigger Compaction
                </button>
              {/if}
            </div>
          </div>
        {/if}
      </div>
    {/if}
  </div>

  <!-- Add Memory Modal -->
  {#if showAddModal}
    <div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div class="bg-gray-800 p-6 rounded-lg w-full max-w-md border border-gray-700">
        <h3 class="text-lg font-semibold mb-4">Add Memory</h3>
        <textarea
          bind:value={newMemory.content}
          placeholder="What do you want to remember?"
          class="w-full h-32 px-3 py-2 bg-gray-900 border border-gray-700 rounded text-sm focus:border-cyan-500 focus:outline-none resize-none"
        ></textarea>
        <div class="grid grid-cols-2 gap-3 mt-4">
          <div>
            <label class="text-xs text-gray-400">Source</label>
            <select bind:value={newMemory.source} class="w-full px-2 py-1 bg-gray-900 border border-gray-700 rounded text-sm">
              <option value="user">User</option>
              <option value="observation">Observation</option>
              <option value="reflection">Reflection</option>
              <option value="system">System</option>
            </select>
          </div>
          <div>
            <label class="text-xs text-gray-400">Department</label>
            <input type="text" bind:value={newMemory.department} placeholder="trading" class="w-full px-2 py-1 bg-gray-900 border border-gray-700 rounded text-sm" />
          </div>
          <div>
            <label class="text-xs text-gray-400">Agent ID</label>
            <input type="text" bind:value={newMemory.agent_id} placeholder="agent-1" class="w-full px-2 py-1 bg-gray-900 border border-gray-700 rounded text-sm" />
          </div>
          <div>
            <label class="text-xs text-gray-400">Importance ({newMemory.importance})</label>
            <input type="range" min="0" max="1" step="0.1" bind:value={newMemory.importance} class="w-full" />
          </div>
        </div>
        <div class="mt-4">
          <label class="text-xs text-gray-400">Tags (comma separated)</label>
          <input type="text" bind:value={newMemory.tags} placeholder="trading, strategy, EURUSD" class="w-full px-2 py-1 bg-gray-900 border border-gray-700 rounded text-sm" />
        </div>
        <div class="flex justify-end gap-2 mt-4">
          <button on:click={() => showAddModal = false} class="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm">Cancel</button>
          <button on:click={addMemory} class="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 rounded text-sm">Save</button>
        </div>
      </div>
    </div>
  {/if}

  <!-- Reflect Modal -->
  {#if showReflectModal}
    <div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div class="bg-gray-800 p-6 rounded-lg w-full max-w-lg border border-gray-700">
        <h3 class="text-lg font-semibold mb-4">Reflect on Memories</h3>
        <input
          type="text"
          bind:value={reflectQuery}
          placeholder="What would you like to know from your memories?"
          class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded text-sm focus:border-purple-500 focus:outline-none"
          on:keydown={(e) => e.key === 'Enter' && reflect()}
        />
        <button
          on:click={reflect}
          disabled={reflectLoading}
          class="w-full mt-3 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded text-sm font-medium disabled:opacity-50"
        >
          {reflectLoading ? 'Reflecting...' : 'Ask'}
        </button>
        {#if reflectAnswer}
          <div class="mt-4 p-4 bg-gray-900 rounded-lg border border-purple-700/50">
            <p class="text-sm text-gray-200 whitespace-pre-wrap">{reflectAnswer}</p>
          </div>
        {/if}
        <div class="flex justify-end mt-4">
          <button on:click={() => { showReflectModal = false; reflectAnswer = ''; }} class="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm">Close</button>
        </div>
      </div>
    </div>
  {/if}
</div>

<script lang="ts">
  import { Database, Brain, Clock, Trash2, Search, RefreshCw, BarChart3 } from 'lucide-svelte';
  import { settingsStore } from '../../../stores/settingsStore';
  
  // State
  let searchQuery = '';
  let selectedType: 'all' | 'semantic' | 'episodic' | 'procedural' = 'all';
  
  // Reactive state
  $: memories = $settingsStore.memories;
  $: memoryStats = {
    semantic: Math.floor(Math.random() * 100),
    episodic: Math.floor(Math.random() * 50),
    procedural: Math.floor(Math.random() * 30),
    total: 0
  };
  
  // Calculate total
  $: memoryStats.total = memoryStats.semantic + memoryStats.episodic + memoryStats.procedural;
  
  // Mock memory entries for display
  const mockMemories = [
    { id: '1', type: 'semantic', content: 'User prefers dark theme for trading interfaces', timestamp: new Date() },
    { id: '2', type: 'episodic', content: 'User ran backtest on EURUSD strategy yesterday', timestamp: new Date(Date.now() - 86400000) },
    { id: '3', type: 'procedural', content: 'Standard workflow: analyze → backtest → optimize → deploy', timestamp: new Date(Date.now() - 172800000) }
  ];
  
  // Filter memories
  $: filteredMemories = mockMemories.filter(m => {
    const matchesType = selectedType === 'all' || m.type === selectedType;
    const matchesSearch = !searchQuery || m.content.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesType && matchesSearch;
  });
  
  // Update memory config
  function updateConfig(key: keyof typeof memories, value: boolean | number) {
    settingsStore.updateMemoryConfig({ [key]: value });
  }
  
  // Clear all memories
  function clearAllMemories() {
    if (confirm('Are you sure you want to clear all memories? This cannot be undone.')) {
      // Would call memory manager to clear
      console.log('Clearing all memories...');
    }
  }
  
  // Clear specific type
  function clearMemoryType(type: 'semantic' | 'episodic' | 'procedural') {
    if (confirm(`Clear all ${type} memories?`)) {
      console.log(`Clearing ${type} memories...`);
    }
  }
  
  // Get type color
  function getTypeColor(type: string): string {
    switch (type) {
      case 'semantic': return 'var(--accent-primary)';
      case 'episodic': return 'var(--accent-secondary)';
      case 'procedural': return 'var(--accent-success)';
      default: return 'var(--text-muted)';
    }
  }
  
  // Format timestamp
  function formatTime(date: Date): string {
    return new Date(date).toLocaleString();
  }

  // Helper to get input value as number
  function getInputNumberValue(e: Event): number {
    return parseInt((e.target as HTMLInputElement).value);
  }
</script>

<div class="memories-settings">
  <h3>Agent Memory</h3>
  <p class="description">Manage agent memory systems for persistent context and learning.</p>
  
  <!-- Memory Statistics -->
  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-icon" style="color: var(--accent-primary)">
        <Brain size={20} />
      </div>
      <div class="stat-info">
        <span class="stat-value">{memoryStats.semantic}</span>
        <span class="stat-label">Semantic</span>
      </div>
    </div>
    
    <div class="stat-card">
      <div class="stat-icon" style="color: var(--accent-secondary)">
        <Clock size={20} />
      </div>
      <div class="stat-info">
        <span class="stat-value">{memoryStats.episodic}</span>
        <span class="stat-label">Episodic</span>
      </div>
    </div>
    
    <div class="stat-card">
      <div class="stat-icon" style="color: var(--accent-success)">
        <Database size={20} />
      </div>
      <div class="stat-info">
        <span class="stat-value">{memoryStats.procedural}</span>
        <span class="stat-label">Procedural</span>
      </div>
    </div>
    
    <div class="stat-card total">
      <div class="stat-icon">
        <BarChart3 size={20} />
      </div>
      <div class="stat-info">
        <span class="stat-value">{memoryStats.total}</span>
        <span class="stat-label">Total</span>
      </div>
    </div>
  </div>
  
  <!-- Memory Configuration -->
  <section class="config-section">
    <h4>Memory Configuration</h4>
    
    <div class="config-grid">
      <div class="config-item">
        <div class="config-info">
          <label for="semantic-memory">Semantic Memory</label>
          <span class="config-desc">Facts and concepts about the world</span>
        </div>
        <label class="toggle">
          <input 
            id="semantic-memory"
            type="checkbox" 
            checked={memories.semanticEnabled}
            on:change={() => updateConfig('semanticEnabled', !memories.semanticEnabled)}
          />
          <span class="toggle-slider"></span>
        </label>
      </div>
      
      <div class="config-item">
        <div class="config-info">
          <label for="episodic-memory">Episodic Memory</label>
          <span class="config-desc">Personal experiences and events</span>
        </div>
        <label class="toggle">
          <input 
            id="episodic-memory"
            type="checkbox" 
            checked={memories.episodicEnabled}
            on:change={() => updateConfig('episodicEnabled', !memories.episodicEnabled)}
          />
          <span class="toggle-slider"></span>
        </label>
      </div>
      
      <div class="config-item">
        <div class="config-info">
          <label for="procedural-memory">Procedural Memory</label>
          <span class="config-desc">Skills and how to perform tasks</span>
        </div>
        <label class="toggle">
          <input 
            id="procedural-memory"
            type="checkbox" 
            checked={memories.proceduralEnabled}
            on:change={() => updateConfig('proceduralEnabled', !memories.proceduralEnabled)}
          />
          <span class="toggle-slider"></span>
        </label>
      </div>
    </div>
    
    <div class="config-row">
      <div class="config-item">
        <div class="config-info">
          <label for="max-entries">Max Entries</label>
          <span class="config-desc">Maximum memories to store</span>
        </div>
        <input 
          id="max-entries"
          type="number" 
          value={memories.maxEntries}
          min="100"
          max="10000"
          step="100"
          on:change={(e) => updateConfig('maxEntries', getInputNumberValue(e))}
        />
      </div>
      
      <div class="config-item">
        <div class="config-info">
          <label for="retention-days">Retention Days</label>
          <span class="config-desc">Days before auto-cleanup</span>
        </div>
        <input 
          id="retention-days"
          type="number" 
          value={memories.retentionDays}
          min="7"
          max="365"
          on:change={(e) => updateConfig('retentionDays', getInputNumberValue(e))}
        />
      </div>
    </div>
  </section>
  
  <!-- Memory Browser -->
  <section class="browser-section">
    <div class="browser-header">
      <h4>Memory Browser</h4>
      <div class="browser-actions">
        <button class="btn secondary small" title="Refresh">
          <RefreshCw size={12} />
        </button>
        <button class="btn secondary small danger" on:click={clearAllMemories}>
          <Trash2 size={12} />
          Clear All
        </button>
      </div>
    </div>
    
    <!-- Search and Filter -->
    <div class="search-filter">
      <div class="search-input">
        <Search size={14} />
        <input 
          type="text" 
          placeholder="Search memories..."
          bind:value={searchQuery}
        />
      </div>
      <div class="filter-chips">
        <button 
          class="chip"
          class:active={selectedType === 'all'}
          on:click={() => selectedType = 'all'}
        >
          All
        </button>
        <button 
          class="chip"
          class:active={selectedType === 'semantic'}
          on:click={() => selectedType = 'semantic'}
        >
          Semantic
        </button>
        <button 
          class="chip"
          class:active={selectedType === 'episodic'}
          on:click={() => selectedType = 'episodic'}
        >
          Episodic
        </button>
        <button 
          class="chip"
          class:active={selectedType === 'procedural'}
          on:click={() => selectedType = 'procedural'}
        >
          Procedural
        </button>
      </div>
    </div>
    
    <!-- Memory List -->
    <div class="memory-list">
      {#if filteredMemories.length === 0}
        <div class="empty-state">
          <Database size={24} />
          <p>No memories found</p>
        </div>
      {:else}
        {#each filteredMemories as memory (memory.id)}
          <div class="memory-item">
            <div class="memory-type" style="background: {getTypeColor(memory.type)}">
              {memory.type.slice(0, 1).toUpperCase()}
            </div>
            <div class="memory-content">
              <p>{memory.content}</p>
              <span class="memory-time">{formatTime(memory.timestamp)}</span>
            </div>
            <button class="icon-btn" title="Delete">
              <Trash2 size={12} />
            </button>
          </div>
        {/each}
      {/if}
    </div>
  </section>
  
  <!-- Info Section -->
  <div class="info-section">
    <h4>About Memory Types</h4>
    <ul>
      <li><strong>Semantic:</strong> Facts, concepts, and general knowledge</li>
      <li><strong>Episodic:</strong> Personal experiences and specific events</li>
      <li><strong>Procedural:</strong> Skills, routines, and how-to knowledge</li>
    </ul>
  </div>
</div>

<style>
  .memories-settings {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  
  h3 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
  }
  
  .description {
    margin: 0;
    font-size: 12px;
    color: var(--text-secondary);
  }
  
  /* Stats Grid */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
  }
  
  .stat-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
  }
  
  .stat-card.total {
    background: var(--bg-primary);
    border-color: var(--accent-primary);
  }
  
  .stat-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
  }
  
  .stat-info {
    display: flex;
    flex-direction: column;
  }
  
  .stat-value {
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
  }
  
  .stat-label {
    font-size: 10px;
    color: var(--text-muted);
    text-transform: uppercase;
  }
  
  /* Config Section */
  .config-section {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  
  .config-section h4 {
    margin: 0;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
  }
  
  .config-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }
  
  .config-row {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
  }
  
  .config-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 12px 16px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
  }
  
  .config-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  
  .config-info label {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-primary);
  }
  
  .config-desc {
    font-size: 10px;
    color: var(--text-muted);
  }
  
  .config-item input[type="number"] {
    width: 80px;
    padding: 6px 10px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 12px;
    text-align: center;
  }
  
  /* Toggle Switch */
  .toggle {
    position: relative;
    display: inline-block;
    width: 40px;
    height: 22px;
    flex-shrink: 0;
  }
  
  .toggle input {
    opacity: 0;
    width: 0;
    height: 0;
  }
  
  .toggle-slider {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 22px;
    transition: all 0.2s;
  }
  
  .toggle-slider:before {
    position: absolute;
    content: "";
    height: 16px;
    width: 16px;
    left: 2px;
    bottom: 2px;
    background-color: var(--text-muted);
    border-radius: 50%;
    transition: all 0.2s;
  }
  
  .toggle input:checked + .toggle-slider {
    background-color: var(--accent-primary);
    border-color: var(--accent-primary);
  }
  
  .toggle input:checked + .toggle-slider:before {
    transform: translateX(18px);
    background-color: var(--bg-primary);
  }
  
  /* Browser Section */
  .browser-section {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  
  .browser-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  
  .browser-header h4 {
    margin: 0;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
  }
  
  .browser-actions {
    display: flex;
    gap: 8px;
  }
  
  .btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
    border: none;
  }
  
  .btn.secondary {
    background: var(--bg-primary);
    color: var(--text-secondary);
    border: 1px solid var(--border-subtle);
  }
  
  .btn.secondary:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }
  
  .btn.secondary.danger:hover {
    color: var(--accent-danger);
    border-color: var(--accent-danger);
  }
  
  .btn.small {
    padding: 6px 10px;
    font-size: 11px;
  }
  
  /* Search and Filter */
  .search-filter {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  
  .search-input {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
  }
  
  .search-input input {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--text-primary);
    font-size: 12px;
    outline: none;
  }
  
  .filter-chips {
    display: flex;
    gap: 6px;
  }
  
  .chip {
    padding: 4px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-muted);
    font-size: 11px;
    cursor: pointer;
    transition: all 0.15s;
  }
  
  .chip:hover {
    color: var(--text-primary);
  }
  
  .chip.active {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: var(--bg-primary);
  }
  
  /* Memory List */
  .memory-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-height: 300px;
    overflow-y: auto;
  }
  
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 24px;
    color: var(--text-muted);
    font-size: 12px;
  }
  
  .memory-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
  }
  
  .memory-type {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 4px;
    color: var(--bg-primary);
    font-size: 11px;
    font-weight: 600;
    flex-shrink: 0;
  }
  
  .memory-content {
    flex: 1;
    min-width: 0;
  }
  
  .memory-content p {
    margin: 0;
    font-size: 12px;
    color: var(--text-primary);
    line-height: 1.4;
  }
  
  .memory-time {
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 4px;
    display: block;
  }
  
  .icon-btn {
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
    transition: all 0.15s;
  }
  
  .icon-btn:hover {
    background: var(--bg-primary);
    color: var(--accent-danger);
  }
  
  /* Info Section */
  .info-section {
    padding: 16px;
    background: var(--bg-tertiary);
    border-radius: 8px;
    border: 1px solid var(--border-subtle);
  }
  
  .info-section h4 {
    margin: 0 0 8px;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
  }
  
  .info-section ul {
    margin: 0;
    padding-left: 20px;
    font-size: 11px;
    color: var(--text-secondary);
  }
  
  .info-section li {
    margin-bottom: 4px;
  }
  
  /* Responsive */
  @media (max-width: 600px) {
    .stats-grid {
      grid-template-columns: repeat(2, 1fr);
    }
    
    .config-grid {
      grid-template-columns: 1fr;
    }
    
    .config-row {
      grid-template-columns: 1fr;
    }
  }
</style>

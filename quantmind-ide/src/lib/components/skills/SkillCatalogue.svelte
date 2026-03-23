<script lang="ts">
  import { onMount } from 'svelte';
  import { listSkills, type Skill } from '$lib/api/skillsApi';
  import { Package, Search, Zap, BarChart3, Shield, Code, Database, Settings } from 'lucide-svelte';

  let skills: Skill[] = [];
  let loading = true;
  let error = '';
  let searchQuery = '';

  // Category icon mapping
  const categoryIcons: Record<string, any> = {
    research: Search,
    trading: BarChart3,
    risk: Shield,
    coding: Code,
    data: Database,
    system: Settings,
    general: Package,
  };

  onMount(async () => {
    await loadSkills();
  });

  async function loadSkills() {
    try {
      loading = true;
      skills = await listSkills();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load skills';
    } finally {
      loading = false;
    }
  }

  $: filteredSkills = searchQuery
    ? skills.filter(s =>
        s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        s.description.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : skills;
</script>

<div class="skill-catalogue">
  <div class="header">
    <h2>Skill Catalogue</h2>
    <div class="search-bar">
      <Search size={16} />
      <input
        type="text"
        placeholder="Search skills..."
        bind:value={searchQuery}
      />
    </div>
  </div>

  {#if loading}
    <div class="loading">
      <Zap class="spin" size={24} />
      <span>Loading skills...</span>
    </div>
  {:else if error}
    <div class="error">
      {error}
    </div>
  {:else}
    <div class="skill-grid">
      {#each filteredSkills as skill}
        <div class="skill-card">
          <div class="skill-icon">
            {#if categoryIcons[skill.category || 'general']}
              <svelte:component this={categoryIcons[skill.category || 'general']} size={24} />
            {:else}
              <Package size={24} />
            {/if}
          </div>
          <div class="skill-info">
            <h3>{skill.name}</h3>
            <p class="slash-command">{skill.slash_command}</p>
            <p class="description">{skill.description}</p>
            <div class="meta">
              <span class="version">v{skill.version}</span>
              <span class="usage">Used {skill.usage_count} times</span>
            </div>
          </div>
        </div>
      {/each}
    </div>

    {#if filteredSkills.length === 0}
      <div class="empty">
        {#if searchQuery}
          No skills match "{searchQuery}"
        {:else}
          No skills registered yet
        {/if}
      </div>
    {/if}
  {/if}
</div>

<style>
  .skill-catalogue {
    padding: 1rem;
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    gap: 1rem;
  }

  .header h2 {
    margin: 0;
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--color-text-primary);
  }

  .search-bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    background: var(--color-bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    color: var(--color-text-secondary);
  }

  .search-bar input {
    background: transparent;
    border: none;
    outline: none;
    color: var(--color-text-primary);
    font-size: 0.875rem;
    width: 200px;
  }

  .search-bar input::placeholder {
    color: var(--color-text-secondary);
  }

  .skill-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1rem;
    overflow-y: auto;
  }

  .skill-card {
    display: flex;
    gap: 1rem;
    padding: 1rem;
    background: var(--color-bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    transition: border-color 0.2s;
  }

  .skill-card:hover {
    border-color: var(--accent-color);
  }

  .skill-icon {
    display: flex;
    align-items: flex-start;
    justify-content: center;
    width: 48px;
    height: 48px;
    background: var(--accent-color);
    color: var(--text-inverse);
    border-radius: 8px;
    flex-shrink: 0;
  }

  .skill-info {
    flex: 1;
    min-width: 0;
  }

  .skill-info h3 {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
    color: var(--color-text-primary);
  }

  .slash-command {
    margin: 0.25rem 0;
    font-size: 0.75rem;
    color: var(--accent-color);
    font-family: monospace;
  }

  .description {
    margin: 0.5rem 0;
    font-size: 0.875rem;
    color: var(--color-text-secondary);
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .meta {
    display: flex;
    gap: 1rem;
    font-size: 0.75rem;
    color: var(--color-text-secondary);
  }

  .version {
    color: var(--color-text-secondary);
  }

  .usage {
    color: var(--text-tertiary);
  }

  .loading, .error, .empty {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 2rem;
    color: var(--color-text-secondary);
  }

  .error {
    color: var(--error-color);
  }

  :global(.spin) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>
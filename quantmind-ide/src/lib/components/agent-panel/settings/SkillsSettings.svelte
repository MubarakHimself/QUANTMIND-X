<script lang="ts">
  import { Bot, Code, Wand2, Sparkles, ToggleLeft, ToggleRight, Play, Loader2, CheckCircle, XCircle, Filter } from 'lucide-svelte';
  import { settingsStore } from '../../../stores/settingsStore';
  import type { AgentType, Skill } from '../../../stores/settingsStore';
  import { departmentList, DEPARTMENTS, type DepartmentId } from '../../../stores/departmentChatStore';
  import { onMount } from 'svelte';
  import { buildApiUrl } from '$lib/api';

  // State
  let selectedAgent: AgentType = $state('research');
  let selectedDepartment: DepartmentId | 'all' = $state('all');
  let backendSkills: any[] = $state([]);
  let isLoadingBackend = false;
  let executingSkill: string | null = $state(null);
  let executionResults: Record<string, any> = {};

  // Agent configuration - using department-based agents
  const agents: Array<{ id: AgentType; name: string; icon: any }> = [
    { id: 'research', name: 'Research', icon: Bot },
    { id: 'development', name: 'Development', icon: Code },
    { id: 'trading', name: 'Trading', icon: Wand2 },
    { id: 'risk', name: 'Risk', icon: Sparkles },
    { id: 'portfolio', name: 'Portfolio', icon: Bot }
  ];

  // Department filter options
  const departmentFilters: Array<{ id: DepartmentId | 'all'; name: string; color: string }> = [
    { id: 'all', name: 'All Departments', color: '#6b7280' },
    ...departmentList.map(d => ({ id: d.id, name: d.name, color: d.color }))
  ];

  // Load backend skills on mount
  onMount(async () => {
    await loadBackendSkills();
  });

  // Load skills from backend
  async function loadBackendSkills() {
    isLoadingBackend = true;
    try {
      const res = await fetch(buildApiUrl('/api/settings/skills'));
      if (res.ok) {
        backendSkills = await res.json();
      }
    } catch (e) {
      console.error('Failed to load backend skills:', e);
    } finally {
      isLoadingBackend = false;
    }
  }

  // Reactive state - merge backend skills department info with settings store skills
  let currentSkills = $derived((() => {
    const storeSkills = $settingsStore.skills[selectedAgent]?.skills || [];
    return storeSkills.map(skill => {
      const backendSkill = backendSkills.find((b: any) => b.id === skill.id);
      return {
        ...skill,
        departments: backendSkill?.departments || skill.departments || []
      };
    });
  })());
  let filteredByDepartment = $derived(selectedDepartment === 'all'
    ? currentSkills
    : currentSkills.filter(s => !s.departments || s.departments.length === 0 || s.departments.includes(selectedDepartment)));
  let coreSkills = $derived(filteredByDepartment.filter(s => s.category === 'core'));
  let advancedSkills = $derived(filteredByDepartment.filter(s => s.category === 'advanced'));
  let customSkills = $derived(filteredByDepartment.filter(s => s.category === 'custom'));

  // Get department info for badges
  function getDepartmentInfo(deptId: string) {
    return DEPARTMENTS[deptId as DepartmentId] || { name: deptId, color: '#6b7280' };
  }

  // Toggle skill
  function toggleSkill(skillId: string) {
    settingsStore.toggleSkill(selectedAgent, skillId);
  }

  // Toggle all skills in a category
  function toggleCategory(category: 'core' | 'advanced' | 'custom', enabled: boolean) {
    const skillsToUpdate = currentSkills.filter(s => s.category === category);
    const updatedSkills = currentSkills.map(s =>
      s.category === category ? { ...s, enabled } : s
    );
    settingsStore.updateAgentSkills(selectedAgent, updatedSkills);
  }

  // Check if all skills in category are enabled
  function isCategoryFullyEnabled(category: string): boolean {
    const skills = currentSkills.filter(s => s.category === category);
    return skills.length > 0 && skills.every(s => s.enabled);
  }

  // Get enabled count
  function getEnabledCount(category: string): string {
    const skills = currentSkills.filter(s => s.category === category);
    const enabled = skills.filter(s => s.enabled).length;
    return `${enabled}/${skills.length}`;
  }

  // Execute a skill via backend API
  async function executeSkill(skillId: string) {
    if (executingSkill) return;

    executingSkill = skillId;
    executionResults[skillId] = null;

    try {
      // Get default parameters from backend skill info
      let params = {};
      const backendSkill = backendSkills.find((s: any) => s.id === skillId);
      if (backendSkill?.parameters?.properties) {
        // Use default values for required params
        for (const [key, prop] of Object.entries(backendSkill.parameters.properties as Record<string, any>)) {
          if (prop.default !== undefined) {
            (params as any)[key] = prop.default;
          }
        }
      }

      const res = await fetch(buildApiUrl(`/api/settings/skills/${skillId}/execute`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ params, context: {} })
      });

      if (res.ok) {
        executionResults[skillId] = await res.json();
      } else {
        executionResults[skillId] = { success: false, error: 'Execution failed' };
      }
    } catch (e) {
      console.error('Skill execution error:', e);
      executionResults[skillId] = { success: false, error: String(e) };
    } finally {
      executingSkill = null;
    }
  }

  // Get execution result for a skill
  function getExecutionResult(skillId: string) {
    return executionResults[skillId];
  }

  // Check if a skill has a backend implementation
  function hasBackendSkill(skillId: string): boolean {
    return backendSkills.some((s: any) => s.id === skillId);
  }
</script>

<div class="skills-settings">
  <h3>Agent Skills</h3>
  <p class="description">Configure which skills are available for each agent.</p>
  
  <!-- Agent Selector -->
  <div class="agent-selector">
    {#each agents as agent}
      <button
        class="agent-chip"
        class:active={selectedAgent === agent.id}
        onclick={() => selectedAgent = agent.id}
      >
        <agent.icon size={14} />
        {agent.name}
      </button>
    {/each}
  </div>

  <!-- Department Filter -->
  <div class="department-filter">
    <div class="filter-label">
      <Filter size={14} />
      <span>Filter by Department</span>
    </div>
    <select
      class="department-select"
      bind:value={selectedDepartment}
    >
      {#each departmentFilters as dept}
        <option value={dept.id}>{dept.name}</option>
      {/each}
    </select>
  </div>
  
  <!-- Skills List -->
  <div class="skills-list">
    <!-- Core Skills -->
    {#if coreSkills.length > 0}
      <section class="skill-section">
        <div class="section-header">
          <h4>
            <Sparkles size={14} />
            Core Skills
          </h4>
          <div class="section-actions">
            <span class="count">{getEnabledCount('core')}</span>
            <button 
              class="toggle-all"
              onclick={() => toggleCategory('core', !isCategoryFullyEnabled('core'))}
            >
              {#if isCategoryFullyEnabled('core')}
                <ToggleRight size={18} />
              {:else}
                <ToggleLeft size={18} />
              {/if}
            </button>
          </div>
        </div>
        
        <div class="skill-items">
          {#each coreSkills as skill (skill.id)}
            <div class="skill-item" class:enabled={skill.enabled}>
              <div class="skill-info">
                <div class="skill-header">
                  <span class="skill-name">{skill.name}</span>
                  {#if skill.departments && skill.departments.length > 0}
                    <div class="department-badges">
                      {#each skill.departments.slice(0, 3) as deptId}
                        {@const dept = getDepartmentInfo(deptId)}
                        <span class="department-badge" style="background-color: {dept.color}20; color: {dept.color}; border-color: {dept.color}40;">
                          {dept.name}
                        </span>
                      {/each}
                      {#if skill.departments.length > 3}
                        <span class="department-badge more">+{skill.departments.length - 3}</span>
                      {/if}
                    </div>
                  {/if}
                </div>
                <span class="skill-description">{skill.description}</span>
                {#if getExecutionResult(skill.id)}
                  <div class="execution-result" class:success={getExecutionResult(skill.id)?.success} class:error={!getExecutionResult(skill.id)?.success}>
                    {#if getExecutionResult(skill.id)?.success}
                      <CheckCircle size={12} />
                      <span>Executed in {getExecutionResult(skill.id)?.execution_time_ms?.toFixed(2)}ms</span>
                    {:else}
                      <XCircle size={12} />
                      <span>{getExecutionResult(skill.id)?.error || 'Execution failed'}</span>
                    {/if}
                  </div>
                {/if}
              </div>
              <div class="skill-actions">
                {#if hasBackendSkill(skill.id)}
                  <button
                    class="run-btn"
                    title="Execute skill"
                    disabled={executingSkill === skill.id}
                    onclick={() => executeSkill(skill.id)}
                  >
                    {#if executingSkill === skill.id}
                      <Loader2 size={14} class="spin" />
                    {:else}
                      <Play size={14} />
                    {/if}
                  </button>
                {/if}
                <label class="toggle">
                  <input
                    type="checkbox"
                    checked={skill.enabled}
                    onchange={() => toggleSkill(skill.id)}
                  />
                  <span class="toggle-slider"></span>
                </label>
              </div>
            </div>
          {/each}
        </div>
      </section>
    {/if}
    
    <!-- Advanced Skills -->
    {#if advancedSkills.length > 0}
      <section class="skill-section">
        <div class="section-header">
          <h4>
            <Code size={14} />
            Advanced Skills
          </h4>
          <div class="section-actions">
            <span class="count">{getEnabledCount('advanced')}</span>
            <button 
              class="toggle-all"
              onclick={() => toggleCategory('advanced', !isCategoryFullyEnabled('advanced'))}
            >
              {#if isCategoryFullyEnabled('advanced')}
                <ToggleRight size={18} />
              {:else}
                <ToggleLeft size={18} />
              {/if}
            </button>
          </div>
        </div>
        
        <div class="skill-items">
          {#each advancedSkills as skill (skill.id)}
            <div class="skill-item" class:enabled={skill.enabled}>
              <div class="skill-info">
                <div class="skill-header">
                  <span class="skill-name">{skill.name}</span>
                  {#if skill.departments && skill.departments.length > 0}
                    <div class="department-badges">
                      {#each skill.departments.slice(0, 3) as deptId}
                        {@const dept = getDepartmentInfo(deptId)}
                        <span class="department-badge" style="background-color: {dept.color}20; color: {dept.color}; border-color: {dept.color}40;">
                          {dept.name}
                        </span>
                      {/each}
                      {#if skill.departments.length > 3}
                        <span class="department-badge more">+{skill.departments.length - 3}</span>
                      {/if}
                    </div>
                  {/if}
                </div>
                <span class="skill-description">{skill.description}</span>
                {#if getExecutionResult(skill.id)}
                  <div class="execution-result" class:success={getExecutionResult(skill.id)?.success} class:error={!getExecutionResult(skill.id)?.success}>
                    {#if getExecutionResult(skill.id)?.success}
                      <CheckCircle size={12} />
                      <span>Executed in {getExecutionResult(skill.id)?.execution_time_ms?.toFixed(2)}ms</span>
                    {:else}
                      <XCircle size={12} />
                      <span>{getExecutionResult(skill.id)?.error || 'Execution failed'}</span>
                    {/if}
                  </div>
                {/if}
              </div>
              <div class="skill-actions">
                {#if hasBackendSkill(skill.id)}
                  <button
                    class="run-btn"
                    title="Execute skill"
                    disabled={executingSkill === skill.id}
                    onclick={() => executeSkill(skill.id)}
                  >
                    {#if executingSkill === skill.id}
                      <Loader2 size={14} class="spin" />
                    {:else}
                      <Play size={14} />
                    {/if}
                  </button>
                {/if}
                <label class="toggle">
                  <input
                    type="checkbox"
                    checked={skill.enabled}
                    onchange={() => toggleSkill(skill.id)}
                  />
                  <span class="toggle-slider"></span>
                </label>
              </div>
            </div>
          {/each}
        </div>
      </section>
    {/if}
    
    <!-- Custom Skills -->
    {#if customSkills.length > 0}
      <section class="skill-section">
        <div class="section-header">
          <h4>
            <Wand2 size={14} />
            Custom Skills
          </h4>
          <div class="section-actions">
            <span class="count">{getEnabledCount('custom')}</span>
            <button 
              class="toggle-all"
              onclick={() => toggleCategory('custom', !isCategoryFullyEnabled('custom'))}
            >
              {#if isCategoryFullyEnabled('custom')}
                <ToggleRight size={18} />
              {:else}
                <ToggleLeft size={18} />
              {/if}
            </button>
          </div>
        </div>
        
        <div class="skill-items">
          {#each customSkills as skill (skill.id)}
            <div class="skill-item" class:enabled={skill.enabled}>
              <div class="skill-info">
                <div class="skill-header">
                  <span class="skill-name">{skill.name}</span>
                  {#if skill.departments && skill.departments.length > 0}
                    <div class="department-badges">
                      {#each skill.departments.slice(0, 3) as deptId}
                        {@const dept = getDepartmentInfo(deptId)}
                        <span class="department-badge" style="background-color: {dept.color}20; color: {dept.color}; border-color: {dept.color}40;">
                          {dept.name}
                        </span>
                      {/each}
                      {#if skill.departments.length > 3}
                        <span class="department-badge more">+{skill.departments.length - 3}</span>
                      {/if}
                    </div>
                  {/if}
                </div>
                <span class="skill-description">{skill.description}</span>
                {#if getExecutionResult(skill.id)}
                  <div class="execution-result" class:success={getExecutionResult(skill.id)?.success} class:error={!getExecutionResult(skill.id)?.success}>
                    {#if getExecutionResult(skill.id)?.success}
                      <CheckCircle size={12} />
                      <span>Executed in {getExecutionResult(skill.id)?.execution_time_ms?.toFixed(2)}ms</span>
                    {:else}
                      <XCircle size={12} />
                      <span>{getExecutionResult(skill.id)?.error || 'Execution failed'}</span>
                    {/if}
                  </div>
                {/if}
              </div>
              <div class="skill-actions">
                {#if hasBackendSkill(skill.id)}
                  <button
                    class="run-btn"
                    title="Execute skill"
                    disabled={executingSkill === skill.id}
                    onclick={() => executeSkill(skill.id)}
                  >
                    {#if executingSkill === skill.id}
                      <Loader2 size={14} class="spin" />
                    {:else}
                      <Play size={14} />
                    {/if}
                  </button>
                {/if}
                <label class="toggle">
                  <input
                    type="checkbox"
                    checked={skill.enabled}
                    onchange={() => toggleSkill(skill.id)}
                  />
                  <span class="toggle-slider"></span>
                </label>
              </div>
            </div>
          {/each}
        </div>
      </section>
    {/if}
    
    <!-- Empty State -->
    {#if currentSkills.length === 0}
      <div class="empty-state">
        <Sparkles size={32} />
        <h4>No Skills Configured</h4>
        <p>This agent has no skills configured yet.</p>
      </div>
    {/if}
  </div>
  
  <!-- Info Section -->
  <div class="info-section">
    <h4>About Skills</h4>
    <p>Skills define what capabilities an agent has. Core skills are always available, while advanced skills can be toggled based on your needs.</p>
  </div>
</div>

<style>
  .skills-settings {
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
  
  /* Agent Selector */
  .agent-selector {
    display: flex;
    gap: 8px;
    padding: 4px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }
  
  .agent-chip {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--text-muted);
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
  }
  
  .agent-chip:hover {
    color: var(--text-primary);
  }
  
  .agent-chip.active {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  /* Department Filter */
  .department-filter {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border-radius: 6px;
  }

  .filter-label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .department-select {
    flex: 1;
    padding: 6px 10px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 12px;
    cursor: pointer;
  }

  .department-select:focus {
    outline: none;
    border-color: var(--accent-primary);
  }
  
  /* Skills List */
  .skills-list {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  
  .skill-section {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  
  .section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border-radius: 6px;
  }
  
  .section-header h4 {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
  }
  
  .section-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  
  .count {
    font-size: 11px;
    color: var(--text-muted);
    padding: 2px 8px;
    background: var(--bg-primary);
    border-radius: 4px;
  }
  
  .toggle-all {
    background: transparent;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    padding: 0;
    display: flex;
    align-items: center;
    transition: color 0.15s;
  }
  
  .toggle-all:hover {
    color: var(--accent-primary);
  }
  
  .skill-items {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  
  .skill-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 12px 16px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    transition: all 0.15s;
  }
  
  .skill-item.enabled {
    border-color: var(--accent-primary);
    background: rgba(107, 200, 230, 0.05);
  }
  
  .skill-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
    flex: 1;
  }
  
  .skill-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }
  
  .skill-description {
    font-size: 11px;
    color: var(--text-muted);
  }

  /* Skill Header with Department Badges */
  .skill-header {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }

  .department-badges {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
  }

  .department-badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 6px;
    font-size: 9px;
    font-weight: 500;
    border-radius: 4px;
    border: 1px solid;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }

  .department-badge.more {
    background: var(--bg-tertiary);
    color: var(--text-muted);
    border-color: var(--border-subtle);
  }

  /* Skill Actions */
  .skill-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  /* Run Button */
  .run-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: var(--accent-primary);
    border: none;
    border-radius: 6px;
    color: var(--bg-primary);
    cursor: pointer;
    transition: all 0.15s;
  }

  .run-btn:hover:not(:disabled) {
    background: var(--accent-secondary);
    transform: scale(1.05);
  }

  .run-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .run-btn :global(.spin) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  /* Execution Result */
  .execution-result {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-top: 4px;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 10px;
  }

  .execution-result.success {
    background: rgba(34, 197, 94, 0.15);
    color: #22c55e;
  }

  .execution-result.error {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
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
  
  /* Empty State */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 32px;
    background: var(--bg-tertiary);
    border: 1px dashed var(--border-subtle);
    border-radius: 12px;
    color: var(--text-muted);
  }
  
  .empty-state h4 {
    margin: 12px 0 4px;
    font-size: 14px;
    color: var(--text-primary);
  }
  
  .empty-state p {
    margin: 0;
    font-size: 12px;
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
  
  .info-section p {
    margin: 0;
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.5;
  }
</style>

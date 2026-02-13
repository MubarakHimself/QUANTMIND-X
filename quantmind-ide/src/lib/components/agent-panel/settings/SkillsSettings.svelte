<script lang="ts">
  import { Bot, Code, Wand2, Sparkles, ToggleLeft, ToggleRight } from 'lucide-svelte';
  import { settingsStore } from '../../../stores/settingsStore';
  import type { AgentType, Skill } from '../../../stores/settingsStore';
  
  // State
  let selectedAgent: AgentType = 'copilot';
  
  // Agent configuration
  const agents: Array<{ id: AgentType; name: string; icon: any }> = [
    { id: 'copilot', name: 'Copilot', icon: Bot },
    { id: 'quantcode', name: 'QuantCode', icon: Code },
    { id: 'analyst', name: 'Analyst', icon: Wand2 }
  ];
  
  // Reactive state
  $: currentSkills = $settingsStore.skills[selectedAgent]?.skills || [];
  $: coreSkills = currentSkills.filter(s => s.category === 'core');
  $: advancedSkills = currentSkills.filter(s => s.category === 'advanced');
  $: customSkills = currentSkills.filter(s => s.category === 'custom');
  
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
        on:click={() => selectedAgent = agent.id}
      >
        <svelte:component this={agent.icon} size={14} />
        {agent.name}
      </button>
    {/each}
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
              on:click={() => toggleCategory('core', !isCategoryFullyEnabled('core'))}
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
                <span class="skill-name">{skill.name}</span>
                <span class="skill-description">{skill.description}</span>
              </div>
              <label class="toggle">
                <input 
                  type="checkbox" 
                  checked={skill.enabled}
                  on:change={() => toggleSkill(skill.id)}
                />
                <span class="toggle-slider"></span>
              </label>
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
              on:click={() => toggleCategory('advanced', !isCategoryFullyEnabled('advanced'))}
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
                <span class="skill-name">{skill.name}</span>
                <span class="skill-description">{skill.description}</span>
              </div>
              <label class="toggle">
                <input 
                  type="checkbox" 
                  checked={skill.enabled}
                  on:change={() => toggleSkill(skill.id)}
                />
                <span class="toggle-slider"></span>
              </label>
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
              on:click={() => toggleCategory('custom', !isCategoryFullyEnabled('custom'))}
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
                <span class="skill-name">{skill.name}</span>
                <span class="skill-description">{skill.description}</span>
              </div>
              <label class="toggle">
                <input 
                  type="checkbox" 
                  checked={skill.enabled}
                  on:change={() => toggleSkill(skill.id)}
                />
                <span class="toggle-slider"></span>
              </label>
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

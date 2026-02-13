<script lang="ts">
  import { fade, slide } from 'svelte/transition';
  import { Plus, Play, Trash2, Edit3, Copy, ChevronRight, Clock, CheckCircle } from 'lucide-svelte';
  import { settingsStore } from '../../../stores/settingsStore';
  import type { Workflow, WorkflowStep } from '../../../stores/settingsStore';
  
  // State
  let showAddModal = false;
  let selectedWorkflow: Workflow | null = null;
  
  // Predefined workflow templates
  const workflowTemplates: Workflow[] = [
    {
      id: 'template-1',
      name: 'Strategy Development',
      description: 'Complete workflow for developing a trading strategy',
      category: 'trading',
      isTemplate: true,
      createdAt: new Date(),
      steps: [
        { id: 's1', name: 'Analyze Market', action: 'analyze', params: {}, order: 1 },
        { id: 's2', name: 'Design Strategy', action: 'design', params: {}, order: 2 },
        { id: 's3', name: 'Backtest', action: 'backtest', params: {}, order: 3 },
        { id: 's4', name: 'Optimize', action: 'optimize', params: {}, order: 4 },
        { id: 's5', name: 'Deploy', action: 'deploy', params: {}, order: 5 }
      ]
    },
    {
      id: 'template-2',
      name: 'Code Review',
      description: 'Review and improve code quality',
      category: 'development',
      isTemplate: true,
      createdAt: new Date(),
      steps: [
        { id: 's1', name: 'Analyze Code', action: 'analyze', params: {}, order: 1 },
        { id: 's2', name: 'Identify Issues', action: 'identify', params: {}, order: 2 },
        { id: 's3', name: 'Suggest Fixes', action: 'suggest', params: {}, order: 3 },
        { id: 's4', name: 'Apply Changes', action: 'apply', params: {}, order: 4 }
      ]
    },
    {
      id: 'template-3',
      name: 'Deployment Pipeline',
      description: 'Deploy strategy to production',
      category: 'deployment',
      isTemplate: true,
      createdAt: new Date(),
      steps: [
        { id: 's1', name: 'Validate', action: 'validate', params: {}, order: 1 },
        { id: 's2', name: 'Test', action: 'test', params: {}, order: 2 },
        { id: 's3', name: 'Stage', action: 'stage', params: {}, order: 3 },
        { id: 's4', name: 'Deploy', action: 'deploy', params: {}, order: 4 }
      ]
    }
  ];
  
  // Reactive state
  $: workflows = $settingsStore.workflows;
  $: customWorkflows = workflows.filter(w => !w.isTemplate);
  $: templateWorkflows = workflows.filter(w => w.isTemplate);
  
  // Use templates if no workflows exist
  $: displayTemplates = workflows.length === 0 ? workflowTemplates : templateWorkflows;
  
  // Create workflow from template
  function createFromTemplate(template: Workflow) {
    settingsStore.addWorkflow({
      name: `${template.name} (Copy)`,
      description: template.description,
      category: template.category,
      isTemplate: false,
      steps: template.steps.map(s => ({ ...s, id: `step_${Date.now()}_${Math.random().toString(36).substr(2, 9)}` }))
    });
  }
  
  // Delete workflow
  function deleteWorkflow(workflowId: string) {
    if (confirm('Are you sure you want to delete this workflow?')) {
      settingsStore.removeWorkflow(workflowId);
    }
  }
  
  // Duplicate workflow
  function duplicateWorkflow(workflow: Workflow) {
    settingsStore.addWorkflow({
      name: `${workflow.name} (Copy)`,
      description: workflow.description,
      category: workflow.category,
      isTemplate: false,
      steps: workflow.steps.map(s => ({ ...s, id: `step_${Date.now()}_${Math.random().toString(36).substr(2, 9)}` }))
    });
  }
  
  // Run workflow
  function runWorkflow(workflow: Workflow) {
    console.log('Running workflow:', workflow.name);
    // Would trigger workflow execution
  }
  
  // Format date
  function formatDate(date: Date | undefined): string {
    if (!date) return 'Never';
    return new Date(date).toLocaleDateString();
  }
  
  // Get category color
  function getCategoryColor(category: string): string {
    const colors: Record<string, string> = {
      trading: 'var(--accent-primary)',
      development: 'var(--accent-secondary)',
      deployment: 'var(--accent-success)',
      analysis: 'var(--accent-warning)'
    };
    return colors[category] || 'var(--text-muted)';
  }
</script>

<div class="workflows-settings">
  <div class="header">
    <div class="header-info">
      <h3>Workflows</h3>
      <p class="description">Automate repetitive tasks with predefined workflows.</p>
    </div>
    <button class="btn primary" on:click={() => showAddModal = true}>
      <Plus size={14} />
      Create Workflow
    </button>
  </div>
  
  <!-- Workflow Templates -->
  <section class="templates-section">
    <h4>Templates</h4>
    <div class="workflow-grid">
      {#each workflowTemplates as template (template.id)}
        <div class="workflow-card template">
          <div class="card-header">
            <span class="category-badge" style="background: {getCategoryColor(template.category)}">
              {template.category}
            </span>
            <span class="step-count">{template.steps.length} steps</span>
          </div>
          
          <h5>{template.name}</h5>
          <p class="workflow-desc">{template.description}</p>
          
          <div class="workflow-steps">
            {#each template.steps.slice(0, 3) as step, i}
              <div class="step-preview">
                <span class="step-num">{i + 1}</span>
                <span class="step-name">{step.name}</span>
              </div>
            {/each}
            {#if template.steps.length > 3}
              <span class="more-steps">+{template.steps.length - 3} more</span>
            {/if}
          </div>
          
          <div class="card-actions">
            <button class="btn secondary small" on:click={() => createFromTemplate(template)}>
              <Copy size={12} />
              Use Template
            </button>
          </div>
        </div>
      {/each}
    </div>
  </section>
  
  <!-- Custom Workflows -->
  {#if customWorkflows.length > 0}
    <section class="custom-section">
      <h4>Your Workflows</h4>
      <div class="workflow-list">
        {#each customWorkflows as workflow (workflow.id)}
          <div class="workflow-item">
            <div class="workflow-info">
              <div class="workflow-header">
                <h5>{workflow.name}</h5>
                <span class="category-badge" style="background: {getCategoryColor(workflow.category)}">
                  {workflow.category}
                </span>
              </div>
              <p class="workflow-desc">{workflow.description}</p>
              <div class="workflow-meta">
                <span><Clock size={10} /> Created: {formatDate(workflow.createdAt)}</span>
                <span><CheckCircle size={10} /> Last run: {formatDate(workflow.lastRun)}</span>
                <span>{workflow.steps.length} steps</span>
              </div>
            </div>
            
            <div class="workflow-actions">
              <button class="btn primary small" on:click={() => runWorkflow(workflow)}>
                <Play size={12} />
                Run
              </button>
              <button class="btn secondary small" on:click={() => duplicateWorkflow(workflow)}>
                <Copy size={12} />
              </button>
              <button class="btn secondary small" on:click={() => selectedWorkflow = workflow}>
                <Edit3 size={12} />
              </button>
              <button class="btn secondary small danger" on:click={() => deleteWorkflow(workflow.id)}>
                <Trash2 size={12} />
              </button>
            </div>
          </div>
        {/each}
      </div>
    </section>
  {/if}
  
  <!-- Add Workflow Modal -->
  {#if showAddModal}
    <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
    <div class="modal-overlay" on:click={() => showAddModal = false} transition:fade role="button" tabindex="-1" aria-label="Close dialog">
      <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions a11y-no-noninteractive-element-interactions -->
      <div class="modal" on:click|stopPropagation transition:slide role="dialog" aria-modal="true" aria-labelledby="workflow-modal-title">
        <h4 id="workflow-modal-title">Create New Workflow</h4>
        <p class="modal-desc">Choose a template to start with or create from scratch.</p>
        
        <div class="template-options">
          {#each workflowTemplates as template}
            <button 
              class="template-option"
              on:click={() => { createFromTemplate(template); showAddModal = false; }}
            >
              <span class="template-name">{template.name}</span>
              <span class="template-category">{template.category}</span>
            </button>
          {/each}
          <button class="template-option blank">
            <span class="template-name">Blank Workflow</span>
            <span class="template-category">Start from scratch</span>
          </button>
        </div>
        
        <div class="modal-actions">
          <button class="btn secondary" on:click={() => showAddModal = false}>Cancel</button>
        </div>
      </div>
    </div>
  {/if}
  
  <!-- Workflow Detail Modal -->
  {#if selectedWorkflow}
    <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
    <div class="modal-overlay" on:click={() => selectedWorkflow = null} transition:fade role="button" tabindex="-1" aria-label="Close dialog">
      <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions a11y-no-noninteractive-element-interactions -->
      <div class="modal large" on:click|stopPropagation transition:slide role="dialog" aria-modal="true" aria-labelledby="workflow-detail-title">
        <div class="modal-header">
          <h4 id="workflow-detail-title">{selectedWorkflow.name}</h4>
          <span class="category-badge" style="background: {getCategoryColor(selectedWorkflow.category)}">
            {selectedWorkflow.category}
          </span>
        </div>
        
        <p class="workflow-desc">{selectedWorkflow.description}</p>
        
        <div class="steps-list">
          <h5>Workflow Steps</h5>
          {#each selectedWorkflow.steps.sort((a, b) => a.order - b.order) as step, i}
            <div class="step-item">
              <div class="step-number">{i + 1}</div>
              <div class="step-details">
                <span class="step-name">{step.name}</span>
                <span class="step-action">Action: {step.action}</span>
              </div>
              <ChevronRight size={14} class="step-arrow" />
            </div>
          {/each}
        </div>
        
        <div class="modal-actions">
          <button class="btn secondary" on:click={() => selectedWorkflow = null}>Close</button>
          <button class="btn primary" on:click={() => { if (selectedWorkflow) runWorkflow(selectedWorkflow); }}>
            <Play size={12} />
            Run Workflow
          </button>
        </div>
      </div>
    </div>
  {/if}
  
  <!-- Info Section -->
  <div class="info-section">
    <h4>About Workflows</h4>
    <p>Workflows automate multi-step processes. Each workflow consists of sequential steps that execute in order. Use templates to get started quickly.</p>
  </div>
</div>

<style>
  .workflows-settings {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  
  .header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
  }
  
  .header-info h3 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
  }
  
  .description {
    margin: 4px 0 0;
    font-size: 12px;
    color: var(--text-secondary);
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
  
  .btn.primary {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }
  
  .btn.primary:hover {
    background: var(--accent-secondary);
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
  
  /* Templates Section */
  .templates-section h4,
  .custom-section h4 {
    margin: 0 0 12px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
  }
  
  .workflow-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }
  
  .workflow-card {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 16px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    transition: all 0.15s;
  }
  
  .workflow-card:hover {
    border-color: var(--accent-primary);
  }
  
  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  
  .category-badge {
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 500;
    color: var(--bg-primary);
    text-transform: capitalize;
  }
  
  .step-count {
    font-size: 10px;
    color: var(--text-muted);
  }
  
  .workflow-card h5 {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
  }
  
  .workflow-desc {
    margin: 0;
    font-size: 11px;
    color: var(--text-secondary);
    line-height: 1.4;
  }
  
  .workflow-steps {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 8px 0;
    border-top: 1px solid var(--border-subtle);
  }
  
  .step-preview {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: var(--text-muted);
  }
  
  .step-num {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    background: var(--bg-primary);
    border-radius: 4px;
    font-size: 9px;
    font-weight: 600;
  }
  
  .more-steps {
    font-size: 10px;
    color: var(--text-muted);
    font-style: italic;
  }
  
  .card-actions {
    margin-top: auto;
  }
  
  /* Custom Workflows */
  .workflow-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  
  .workflow-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    padding: 16px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
  }
  
  .workflow-info {
    flex: 1;
    min-width: 0;
  }
  
  .workflow-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
  }
  
  .workflow-header h5 {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
  }
  
  .workflow-meta {
    display: flex;
    gap: 16px;
    margin-top: 8px;
    font-size: 10px;
    color: var(--text-muted);
  }
  
  .workflow-meta span {
    display: flex;
    align-items: center;
    gap: 4px;
  }
  
  .workflow-actions {
    display: flex;
    gap: 6px;
  }
  
  /* Modal */
  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }
  
  .modal {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 24px;
    width: 400px;
    max-width: 90%;
    border: 1px solid var(--border-subtle);
  }
  
  .modal.large {
    width: 520px;
  }
  
  .modal h4 {
    margin: 0 0 8px;
    font-size: 16px;
    color: var(--text-primary);
  }
  
  .modal-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
  }
  
  .modal-header h4 {
    margin: 0;
  }
  
  .modal-desc {
    margin: 0 0 16px;
    font-size: 12px;
    color: var(--text-secondary);
  }
  
  .template-options {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  
  .template-option {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.15s;
    text-align: left;
  }
  
  .template-option:hover {
    border-color: var(--accent-primary);
    background: var(--bg-primary);
  }
  
  .template-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }
  
  .template-category {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: capitalize;
  }
  
  .modal-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    margin-top: 20px;
  }
  
  /* Steps List */
  .steps-list {
    margin-top: 16px;
  }
  
  .steps-list h5 {
    margin: 0 0 12px;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-muted);
  }
  
  .step-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 12px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    margin-bottom: 6px;
  }
  
  .step-number {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: var(--accent-primary);
    border-radius: 6px;
    color: var(--bg-primary);
    font-size: 11px;
    font-weight: 600;
  }
  
  .step-details {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  
  .step-name {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-primary);
  }
  
  .step-action {
    font-size: 10px;
    color: var(--text-muted);
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
  
  /* Responsive */
  @media (max-width: 600px) {
    .workflow-grid {
      grid-template-columns: 1fr;
    }
    
    .workflow-item {
      flex-direction: column;
      align-items: flex-start;
    }
    
    .workflow-actions {
      width: 100%;
      justify-content: flex-end;
      margin-top: 12px;
    }
  }
</style>

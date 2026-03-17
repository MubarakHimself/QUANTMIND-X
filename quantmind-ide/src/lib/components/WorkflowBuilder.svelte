<script lang="ts">
  import { stopPropagation } from 'svelte/legacy';

  import { onMount } from 'svelte';
  import {
    Workflow,
    Plus,
    Trash2,
    Save,
    Play,
    GripVertical,
    ChevronDown,
    ChevronRight,
    Copy,
    FileText,
    Settings,
    CheckCircle,
    XCircle,
    AlertCircle,
    ArrowRight,
    Layers
  } from 'lucide-svelte';
  import { settingsStore, workflowTemplates, type Workflow as WorkflowType, type WorkflowStep } from '$lib/stores/settingsStore';

  // Available workflow step types
  interface StepType {
    id: string;
    name: string;
    description: string;
    icon: string;
    defaultParams: Record<string, unknown>;
  }

  const availableStepTypes: StepType[] = [
    {
      id: 'video_ingest',
      name: 'Video Ingest',
      description: 'Download and process YouTube videos',
      icon: '📹',
      defaultParams: { is_playlist: false }
    },
    {
      id: 'video_analysis',
      name: 'Video Analysis',
      description: 'Analyze video for trading strategies',
      icon: '🔍',
      defaultParams: { extract_indicators: true, extract_rules: true }
    },
    {
      id: 'trd_generation',
      name: 'TRD Generation',
      description: 'Create Trading Requirements Document',
      icon: '📝',
      defaultParams: { template: 'standard' }
    },
    {
      id: 'ea_creation',
      name: 'EA Creation',
      description: 'Generate MQL5 Expert Advisor code',
      icon: '💻',
      defaultParams: { language: 'mql5', include_risk_management: true }
    },
    {
      id: 'backtest',
      name: 'Backtest',
      description: 'Run strategy backtest',
      icon: '📈',
      defaultParams: { period: '1Y', symbols: ['EURUSD'] }
    },
    {
      id: 'validation',
      name: 'Validation',
      description: 'Validate generated EA',
      icon: '✅',
      defaultParams: { strict_mode: true }
    },
    {
      id: 'notification',
      name: 'Notification',
      description: 'Send workflow notifications',
      icon: '🔔',
      defaultParams: { channels: ['email', 'slack'] }
    }
  ];

  // Pre-built templates
  const templates: Partial<WorkflowType>[] = [
    {
      name: 'Video Ingest to EA',
      description: 'Complete pipeline from YouTube video to MQL5 EA',
      category: 'video',
      isTemplate: true,
      steps: [
        { id: '1', name: 'Video Ingest', action: 'video_ingest', params: { is_playlist: false }, order: 0 },
        { id: '2', name: 'Video Analysis', action: 'video_analysis', params: { extract_indicators: true }, order: 1 },
        { id: '3', name: 'TRD Generation', action: 'trd_generation', params: { template: 'standard' }, order: 2 },
        { id: '4', name: 'EA Creation', action: 'ea_creation', params: { language: 'mql5' }, order: 3 },
        { id: '5', name: 'Validation', action: 'validation', params: { strict_mode: true }, order: 4 }
      ]
    },
    {
      name: 'Quick Analysis',
      description: 'Analyze video without EA generation',
      category: 'video',
      isTemplate: true,
      steps: [
        { id: '1', name: 'Video Ingest', action: 'video_ingest', params: { is_playlist: false }, order: 0 },
        { id: '2', name: 'Video Analysis', action: 'video_analysis', params: { extract_indicators: true }, order: 1 }
      ]
    },
    {
      name: 'Full Backtest Pipeline',
      description: 'Complete pipeline with backtesting',
      category: 'backtest',
      isTemplate: true,
      steps: [
        { id: '1', name: 'Video Ingest', action: 'video_ingest', params: { is_playlist: false }, order: 0 },
        { id: '2', name: 'Video Analysis', action: 'video_analysis', params: { extract_indicators: true }, order: 1 },
        { id: '3', name: 'TRD Generation', action: 'trd_generation', params: { template: 'standard' }, order: 2 },
        { id: '4', name: 'EA Creation', action: 'ea_creation', params: { language: 'mql5' }, order: 3 },
        { id: '5', name: 'Backtest', action: 'backtest', params: { period: '1Y' }, order: 4 },
        { id: '6', name: 'Validation', action: 'validation', params: { strict_mode: true }, order: 5 }
      ]
    }
  ];

  // State
  let workflowName = $state('');
  let workflowDescription = $state('');
  let workflowCategory = $state('custom');
  let steps: WorkflowStep[] = $state([]);
  let selectedStepType: string | null = null;
  let expandedStep: string | null = $state(null);
  let editingStep: WorkflowStep | null = null;
  let isEditing = $state(false);
  let editWorkflowId: string | null = null;
  let saving = $state(false);
  let error = $state('');
  let showStepSelector = $state(false);
  let showTemplates = $state(false);
  let dragIndex: number | null = null;
  let runningWorkflow: string | null = $state(null);

  // Run workflow (only for video_ingest_to_ea type)
  async function runWorkflow(workflow: WorkflowType) {
    if (workflow.steps.length === 0) return;

    // Check if this is a video workflow that can be run
    const hasVideoIngest = workflow.steps.some(s => s.action === 'video_ingest');
    if (!hasVideoIngest) {
      alert('This workflow type cannot be run directly. Only workflows with Video Ingest step can be executed.');
      return;
    }

    runningWorkflow = workflow.id;

    try {
      // For now, just navigate to the VideoIngest workflow view
      // In a full implementation, this would create a custom workflow execution
      alert('Workflow execution will open the VideoIngest panel. Please enter a YouTube URL there.');
    } catch (e) {
      console.error('Failed to run workflow:', e);
    } finally {
      runningWorkflow = null;
    }
  }

  // Get step type info
  function getStepType(stepTypeId: string): StepType | undefined {
    return availableStepTypes.find(s => s.id === stepTypeId);
  }

  // Add step from selector
  function addStep(stepType: StepType) {
    const newStep: WorkflowStep = {
      id: crypto.randomUUID(),
      name: stepType.name,
      action: stepType.id,
      params: { ...stepType.defaultParams },
      order: steps.length
    };
    steps = [...steps, newStep];
    showStepSelector = false;
    expandedStep = newStep.id;
  }

  // Remove step
  function removeStep(stepId: string) {
    steps = steps.filter(s => s.id !== stepId);
    // Reorder remaining steps
    steps = steps.map((s, i) => ({ ...s, order: i }));
  }

  // Move step up
  function moveStepUp(index: number) {
    if (index === 0) return;
    const newSteps = [...steps];
    [newSteps[index - 1], newSteps[index]] = [newSteps[index], newSteps[index - 1]];
    steps = newSteps.map((s, i) => ({ ...s, order: i }));
  }

  // Move step down
  function moveStepDown(index: number) {
    if (index === steps.length - 1) return;
    const newSteps = [...steps];
    [newSteps[index], newSteps[index + 1]] = [newSteps[index + 1], newSteps[index]];
    steps = newSteps.map((s, i) => ({ ...s, order: i }));
  }

  // Toggle step expansion
  function toggleStep(stepId: string) {
    expandedStep = expandedStep === stepId ? null : stepId;
  }

  // Update step param
  function updateStepParam(stepId: string, key: string, value: unknown) {
    steps = steps.map(s => {
      if (s.id === stepId) {
        return { ...s, params: { ...s.params, [key]: value } };
      }
      return s;
    });
  }

  // Load template
  function loadTemplate(template: Partial<WorkflowType>) {
    workflowName = template.name || '';
    workflowDescription = template.description || '';
    workflowCategory = template.category || 'custom';
    steps = (template.steps || []).map((s, i) => ({
      ...s,
      id: crypto.randomUUID(),
      order: i
    }));
    showTemplates = false;
  }

  // Load existing workflow for editing
  function editWorkflow(workflow: WorkflowType) {
    editWorkflowId = workflow.id;
    workflowName = workflow.name;
    workflowDescription = workflow.description;
    workflowCategory = workflow.category;
    steps = [...workflow.steps];
    isEditing = true;
  }

  // Clear form
  function clearForm() {
    workflowName = '';
    workflowDescription = '';
    workflowCategory = 'custom';
    steps = [];
    editWorkflowId = null;
    isEditing = false;
    error = '';
    expandedStep = null;
  }

  // Save workflow
  async function saveWorkflow() {
    if (!workflowName.trim()) {
      error = 'Workflow name is required';
      return;
    }
    if (steps.length === 0) {
      error = 'At least one step is required';
      return;
    }

    saving = true;
    error = '';

    try {
      if (isEditing && editWorkflowId) {
        settingsStore.updateWorkflow(editWorkflowId, {
          name: workflowName,
          description: workflowDescription,
          category: workflowCategory,
          steps: steps.map(s => ({
            id: s.id,
            name: s.name,
            action: s.action,
            params: s.params,
            order: s.order
          }))
        });
      } else {
        settingsStore.addWorkflow({
          name: workflowName,
          description: workflowDescription,
          category: workflowCategory,
          isTemplate: false,
          steps: steps.map(s => ({
            id: s.id,
            name: s.name,
            action: s.action,
            params: s.params,
            order: s.order
          }))
        });
      }
      clearForm();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to save workflow';
    } finally {
      saving = false;
    }
  }

  // Delete workflow
  function deleteWorkflow(workflowId: string) {
    if (confirm('Are you sure you want to delete this workflow?')) {
      settingsStore.removeWorkflow(workflowId);
    }
  }

  // Get icon for step action
  function getStepIcon(action: string): string {
    const stepType = getStepType(action);
    return stepType?.icon || '📌';
  }

  // Check if step has issues (missing required params)
  function hasStepIssues(step: WorkflowStep): boolean {
    const stepType = getStepType(step.action);
    if (!stepType) return false;
    // Add validation logic here if needed
    return false;
  }
</script>

<div class="workflow-builder">
  <!-- Header -->
  <div class="builder-header">
    <div class="header-title">
      <Workflow size={20} />
      <h2>Workflow Builder</h2>
    </div>
    <div class="header-actions">
      <button class="btn secondary" onclick={() => showTemplates = !showTemplates}>
        <Layers size={14} />
        {showTemplates ? 'Hide' : 'Templates'}
      </button>
    </div>
  </div>

  <!-- Error Display -->
  {#if error}
    <div class="error-banner">
      <AlertCircle size={16} />
      <span>{error}</span>
      <button onclick={() => error = ''}><XCircle size={16} /></button>
    </div>
  {/if}

  <div class="builder-content">
    <!-- Templates Panel -->
    {#if showTemplates}
      <div class="templates-panel" transition:slide={{ duration: 200 }}>
        <h3>Pre-built Templates</h3>
        <div class="templates-grid">
          {#each templates as template}
            <div class="template-card" onclick={() => loadTemplate(template)}>
              <div class="template-icon">
                <Workflow size={24} />
              </div>
              <div class="template-info">
                <h4>{template.name}</h4>
                <p>{template.description}</p>
                <span class="step-count">{template.steps?.length || 0} steps</span>
              </div>
            </div>
          {/each}
        </div>
      </div>
    {/if}

    <!-- Workflow Form -->
    <div class="workflow-form">
      <!-- Basic Info -->
      <div class="form-section">
        <h3>{isEditing ? 'Edit Workflow' : 'New Workflow'}</h3>

        <div class="form-row">
          <div class="form-group">
            <label for="wf-name">Name</label>
            <input
              id="wf-name"
              type="text"
              bind:value={workflowName}
              placeholder="My Workflow"
            />
          </div>
          <div class="form-group">
            <label for="wf-category">Category</label>
            <select id="wf-category" bind:value={workflowCategory}>
              <option value="custom">Custom</option>
              <option value="video">Video</option>
              <option value="backtest">Backtest</option>
              <option value="trading">Trading</option>
              <option value="analysis">Analysis</option>
            </select>
          </div>
        </div>

        <div class="form-group">
          <label for="wf-desc">Description</label>
          <textarea
            id="wf-desc"
            bind:value={workflowDescription}
            placeholder="Describe what this workflow does..."
            rows={2}
          ></textarea>
        </div>
      </div>

      <!-- Steps Section -->
      <div class="steps-section">
        <div class="steps-header">
          <h3>Workflow Steps</h3>
          <button class="btn primary" onclick={() => showStepSelector = !showStepSelector}>
            <Plus size={14} />
            Add Step
          </button>
        </div>

        <!-- Step Type Selector -->
        {#if showStepSelector}
          <div class="step-selector">
            <div class="selector-header">
              <span>Select Step Type</span>
              <button class="close-btn" onclick={() => showStepSelector = false}>
                <XCircle size={16} />
              </button>
            </div>
            <div class="step-types-grid">
              {#each availableStepTypes as stepType}
                <div
                  class="step-type-option"
                  class:selected={selectedStepType === stepType.id}
                  onclick={() => addStep(stepType)}
                >
                  <span class="step-type-icon">{stepType.icon}</span>
                  <div class="step-type-info">
                    <span class="step-type-name">{stepType.name}</span>
                    <span class="step-type-desc">{stepType.description}</span>
                  </div>
                </div>
              {/each}
            </div>
          </div>
        {/if}

        <!-- Steps List -->
        {#if steps.length === 0}
          <div class="empty-steps">
            <FileText size={32} />
            <p>No steps added yet</p>
            <p class="hint">Click "Add Step" to start building your workflow</p>
          </div>
        {:else}
          <div class="steps-list">
            {#each steps as step, index (step.id)}
              <div
                class="step-item"
                class:expanded={expandedStep === step.id}
                class:has-issues={hasStepIssues(step)}
              >
                <!-- Step Header -->
                <div class="step-header" onclick={() => toggleStep(step.id)}>
                  <div class="step-drag">
                    <GripVertical size={16} />
                  </div>
                  <div class="step-order">
                    {index + 1}
                  </div>
                  <div class="step-icon">
                    {getStepIcon(step.action)}
                  </div>
                  <div class="step-info">
                    <span class="step-name">{step.name}</span>
                    <span class="step-action">{step.action}</span>
                  </div>
                  <div class="step-status">
                    {#if hasStepIssues(step)}
                      <AlertCircle size={14} class="warning" />
                    {:else}
                      <CheckCircle size={14} class="success" />
                    {/if}
                  </div>
                  <div class="step-actions">
                    <button
                      class="action-btn"
                      onclick={stopPropagation(() => moveStepUp(index))}
                      disabled={index === 0}
                      title="Move up"
                    >
                      <ChevronRight size={14} style="transform: rotate(-90deg)" />
                    </button>
                    <button
                      class="action-btn"
                      onclick={stopPropagation(() => moveStepDown(index))}
                      disabled={index === steps.length - 1}
                      title="Move down"
                    >
                      <ChevronRight size={14} style="transform: rotate(90deg)" />
                    </button>
                    <button
                      class="action-btn danger"
                      onclick={stopPropagation(() => removeStep(step.id))}
                      title="Remove step"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>

                <!-- Step Details (Expanded) -->
                {#if expandedStep === step.id}
                  <div class="step-details">
                    <div class="step-params">
                      <h4>Parameters</h4>
                      {#each Object.entries(step.params) as [key, value]}
                        <div class="param-row">
                          <label for="param-{step.id}-{key}">{key}</label>
                          {#if typeof value === 'boolean'}
                            <input
                              type="checkbox"
                              id="param-{step.id}-{key}"
                              checked={value}
                              onchange={(e) => updateStepParam(step.id, key, e.currentTarget.checked)}
                            />
                          {:else if typeof value === 'number'}
                            <input
                              type="number"
                              id="param-{step.id}-{key}"
                              value={value}
                              onchange={(e) => updateStepParam(step.id, key, parseFloat(e.currentTarget.value))}
                            />
                          {:else if Array.isArray(value)}
                            <input
                              type="text"
                              id="param-{step.id}-{key}"
                              value={value.join(', ')}
                              onchange={(e) => updateStepParam(step.id, key, e.currentTarget.value.split(',').map(s => s.trim()))}
                              placeholder="Comma-separated values"
                            />
                          {:else}
                            <input
                              type="text"
                              id="param-{step.id}-{key}"
                              value={value}
                              onchange={(e) => updateStepParam(step.id, key, e.currentTarget.value)}
                            />
                          {/if}
                        </div>
                      {/each}
                    </div>
                  </div>
                {/if}

                <!-- Connection Line -->
                {#if index < steps.length - 1}
                  <div class="step-connector">
                    <ArrowRight size={12} />
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        {/if}
      </div>

      <!-- Form Actions -->
      <div class="form-actions">
        {#if isEditing}
          <button class="btn secondary" onclick={clearForm}>
            Cancel
          </button>
        {/if}
        <button
          class="btn primary"
          onclick={saveWorkflow}
          disabled={saving || !workflowName.trim() || steps.length === 0}
        >
          {#if saving}
            <span class="spinner"></span>
            Saving...
          {:else}
            <Save size={14} />
            {isEditing ? 'Update' : 'Save'} Workflow
          {/if}
        </button>
      </div>
    </div>

    <!-- Saved Workflows Panel -->
    <div class="saved-workflows">
      <h3>Saved Workflows</h3>
      {#if $workflowTemplates.length === 0 && $settingsStore.workflows.length === 0}
        <div class="empty-workflows">
          <Workflow size={32} />
          <p>No saved workflows</p>
          <p class="hint">Create a workflow using the builder above</p>
        </div>
      {:else}
        <div class="workflows-list">
          {#each $settingsStore.workflows as workflow}
            <div class="workflow-card">
              <div class="workflow-info">
                <h4>{workflow.name}</h4>
                <p>{workflow.description}</p>
                <div class="workflow-meta">
                  <span class="category">{workflow.category}</span>
                  <span class="steps-count">{workflow.steps.length} steps</span>
                </div>
              </div>
              <div class="workflow-actions">
                <button
                  class="action-btn success"
                  onclick={() => runWorkflow(workflow)}
                  title="Run Workflow"
                  disabled={runningWorkflow === workflow.id}
                >
                  {#if runningWorkflow === workflow.id}
                    <span class="spinner-sm"></span>
                  {:else}
                    <Play size={14} />
                  {/if}
                </button>
                <button class="action-btn" onclick={() => editWorkflow(workflow)} title="Edit">
                  <Settings size={14} />
                </button>
                <button class="action-btn danger" onclick={() => deleteWorkflow(workflow.id)} title="Delete">
                  <Trash2 size={14} />
                </button>
              </div>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  .workflow-builder {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-secondary, #1e1e2e);
    color: var(--text-primary, #cdd6f4);
    overflow: hidden;
  }

  .builder-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px;
    border-bottom: 1px solid var(--border-color, #313244);
    background: var(--bg-tertiary, #252536);
  }

  .header-title {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .header-title h2 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
  }

  .header-actions {
    display: flex;
    gap: 8px;
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    background: rgba(243, 139, 168, 0.15);
    border-bottom: 1px solid #f38ba8;
    color: #f38ba8;
    font-size: 13px;
  }

  .error-banner button {
    margin-left: auto;
    background: transparent;
    border: none;
    color: inherit;
    cursor: pointer;
    padding: 0;
    display: flex;
  }

  .builder-content {
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 300px;
    gap: 16px;
    padding: 16px;
    overflow: auto;
  }

  .templates-panel {
    grid-column: 1 / -1;
    padding: 16px;
    background: var(--bg-tertiary, #313244);
    border-radius: 8px;
    margin-bottom: 8px;
  }

  .templates-panel h3 {
    margin: 0 0 12px;
    font-size: 14px;
    color: var(--text-secondary, #a6adc8);
  }

  .templates-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px;
  }

  .template-card {
    display: flex;
    gap: 12px;
    padding: 12px;
    background: var(--bg-secondary, #1e1e2e);
    border: 1px solid var(--border-color, #45475a);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .template-card:hover {
    border-color: var(--accent-primary, #89b4fa);
    background: rgba(137, 180, 250, 0.1);
  }

  .template-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 48px;
    height: 48px;
    background: var(--bg-tertiary, #313244);
    border-radius: 8px;
    color: var(--accent-primary, #89b4fa);
  }

  .template-info h4 {
    margin: 0;
    font-size: 14px;
    font-weight: 500;
  }

  .template-info p {
    margin: 4px 0;
    font-size: 12px;
    color: var(--text-secondary, #a6adc8);
  }

  .step-count {
    font-size: 11px;
    color: var(--text-muted, #6c7086);
  }

  .workflow-form {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  .form-section, .steps-section {
    background: var(--bg-tertiary, #313244);
    border-radius: 8px;
    padding: 16px;
  }

  .form-section h3, .steps-section h3 {
    margin: 0 0 16px;
    font-size: 14px;
    font-weight: 500;
    color: var(--text-secondary, #a6adc8);
  }

  .form-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }

  .form-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .form-group label {
    font-size: 12px;
    color: var(--text-secondary, #a6adc8);
    font-weight: 500;
  }

  .form-group input,
  .form-group select,
  .form-group textarea {
    padding: 10px 12px;
    background: var(--bg-secondary, #1e1e2e);
    border: 1px solid var(--border-color, #45475a);
    border-radius: 6px;
    color: var(--text-primary, #cdd6f4);
    font-size: 13px;
    font-family: inherit;
  }

  .form-group input:focus,
  .form-group select:focus,
  .form-group textarea:focus {
    outline: none;
    border-color: var(--accent-primary, #89b4fa);
  }

  .form-group textarea {
    resize: vertical;
    min-height: 60px;
  }

  .steps-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }

  .steps-header h3 {
    margin: 0;
  }

  .step-selector {
    background: var(--bg-secondary, #1e1e2e);
    border: 1px solid var(--border-color, #45475a);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 16px;
  }

  .selector-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    font-size: 13px;
    color: var(--text-secondary, #a6adc8);
  }

  .close-btn {
    background: transparent;
    border: none;
    color: var(--text-muted, #6c7086);
    cursor: pointer;
    padding: 4px;
    display: flex;
  }

  .close-btn:hover {
    color: var(--text-primary, #cdd6f4);
  }

  .step-types-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 8px;
  }

  .step-type-option {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px;
    background: var(--bg-tertiary, #313244);
    border: 1px solid var(--border-color, #45475a);
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .step-type-option:hover {
    border-color: var(--accent-primary, #89b4fa);
  }

  .step-type-option.selected {
    border-color: var(--accent-primary, #89b4fa);
    background: rgba(137, 180, 250, 0.1);
  }

  .step-type-icon {
    font-size: 20px;
  }

  .step-type-name {
    display: block;
    font-size: 12px;
    font-weight: 500;
  }

  .step-type-desc {
    display: block;
    font-size: 10px;
    color: var(--text-muted, #6c7086);
  }

  .empty-steps {
    text-align: center;
    padding: 32px;
    color: var(--text-muted, #6c7086);
  }

  .empty-steps p {
    margin: 8px 0 0;
  }

  .hint {
    font-size: 12px;
    opacity: 0.7;
  }

  .steps-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .step-item {
    position: relative;
    background: var(--bg-secondary, #1e1e2e);
    border: 1px solid var(--border-color, #45475a);
    border-radius: 8px;
    overflow: hidden;
    transition: all 0.2s;
  }

  .step-item.expanded {
    border-color: var(--accent-primary, #89b4fa);
  }

  .step-item.has-issues {
    border-color: var(--accent-warning, #f9e2af);
  }

  .step-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px;
    cursor: pointer;
  }

  .step-drag {
    color: var(--text-muted, #6c7086);
    cursor: grab;
  }

  .step-order {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: var(--accent-primary, #89b4fa);
    color: var(--bg-primary, #1e1e2e);
    border-radius: 50%;
    font-size: 11px;
    font-weight: 600;
  }

  .step-icon {
    font-size: 18px;
  }

  .step-info {
    flex: 1;
  }

  .step-name {
    display: block;
    font-size: 13px;
    font-weight: 500;
  }

  .step-action {
    display: block;
    font-size: 11px;
    color: var(--text-muted, #6c7086);
  }

  .step-status {
    color: var(--accent-success, #a6e3a1);
  }

  .step-status .warning {
    color: var(--accent-warning, #f9e2af);
  }

  .step-actions {
    display: flex;
    gap: 4px;
  }

  .action-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 4px;
    background: transparent;
    border: none;
    color: var(--text-muted, #6c7086);
    cursor: pointer;
    border-radius: 4px;
    transition: all 0.2s;
  }

  .action-btn:hover:not(:disabled) {
    background: var(--bg-tertiary, #313244);
    color: var(--text-primary, #cdd6f4);
  }

  .action-btn:disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }

  .action-btn.danger:hover:not(:disabled) {
    color: #f38ba8;
  }

  .action-btn.success:hover:not(:disabled) {
    color: var(--accent-success, #a6e3a1);
  }

  .spinner-sm {
    width: 12px;
    height: 12px;
    border: 2px solid transparent;
    border-top-color: currentColor;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  .step-details {
    padding: 12px;
    background: var(--bg-tertiary, #313244);
    border-top: 1px solid var(--border-color, #45475a);
  }

  .step-params h4 {
    margin: 0 0 12px;
    font-size: 12px;
    color: var(--text-secondary, #a6adc8);
  }

  .param-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
  }

  .param-row label {
    flex: 0 0 120px;
    font-size: 12px;
    color: var(--text-secondary, #a6adc8);
  }

  .param-row input[type="text"],
  .param-row input[type="number"] {
    flex: 1;
    padding: 6px 10px;
    background: var(--bg-secondary, #1e1e2e);
    border: 1px solid var(--border-color, #45475a);
    border-radius: 4px;
    color: var(--text-primary, #cdd6f4);
    font-size: 12px;
  }

  .param-row input[type="checkbox"] {
    width: 18px;
    height: 18px;
    cursor: pointer;
  }

  .step-connector {
    position: absolute;
    bottom: -16px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: var(--bg-secondary, #1e1e2e);
    border: 1px solid var(--border-color, #45475a);
    border-radius: 50%;
    color: var(--text-muted, #6c7086);
  }

  .form-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    padding-top: 16px;
    border-top: 1px solid var(--border-color, #45475a);
  }

  .btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 10px 16px;
    border: none;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
  }

  .btn.primary {
    background: var(--accent-primary, #89b4fa);
    color: var(--bg-primary, #1e1e2e);
  }

  .btn.primary:hover:not(:disabled) {
    background: #a6c8ff;
  }

  .btn.primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn.secondary {
    background: var(--bg-tertiary, #313244);
    color: var(--text-primary, #cdd6f4);
    border: 1px solid var(--border-color, #45475a);
  }

  .btn.secondary:hover {
    background: var(--bg-secondary, #1e1e2e);
  }

  .spinner {
    width: 14px;
    height: 14px;
    border: 2px solid transparent;
    border-top-color: currentColor;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .saved-workflows {
    background: var(--bg-tertiary, #313244);
    border-radius: 8px;
    padding: 16px;
    overflow-y: auto;
    max-height: calc(100vh - 200px);
  }

  .saved-workflows h3 {
    margin: 0 0 12px;
    font-size: 14px;
    font-weight: 500;
    color: var(--text-secondary, #a6adc8);
  }

  .empty-workflows {
    text-align: center;
    padding: 24px;
    color: var(--text-muted, #6c7086);
  }

  .empty-workflows p {
    margin: 8px 0 0;
    font-size: 12px;
  }

  .workflows-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .workflow-card {
    display: flex;
    justify-content: space-between;
    padding: 12px;
    background: var(--bg-secondary, #1e1e2e);
    border: 1px solid var(--border-color, #45475a);
    border-radius: 8px;
  }

  .workflow-info h4 {
    margin: 0;
    font-size: 13px;
    font-weight: 500;
  }

  .workflow-info p {
    margin: 4px 0;
    font-size: 11px;
    color: var(--text-secondary, #a6adc8);
  }

  .workflow-meta {
    display: flex;
    gap: 8px;
    font-size: 10px;
    color: var(--text-muted, #6c7086);
  }

  .category {
    padding: 2px 6px;
    background: var(--bg-tertiary, #313244);
    border-radius: 4px;
    text-transform: capitalize;
  }

  .workflow-actions {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
</style>

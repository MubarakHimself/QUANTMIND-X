<script lang="ts">
  /**
   * Workflow Panel Component
   *
   * Displays active workflows with progress, controls, and results.
   */
  import { onMount, onDestroy } from 'svelte';
  import workflowStore, {
    type Workflow,
    type WorkflowStep,
    type WorkflowTemplate,
    WORKFLOW_TEMPLATES,
    activeWorkflows,
    selectedWorkflow,
    loading,
    error,
  } from '$lib/stores/workflowStore';

  // Props
  export let showHistory = false;

  // State
  let refreshInterval: number | null = null;
  let showCreateModal = false;
  let selectedTemplate: WorkflowTemplate = 'video_ingest_full';
  let trdContent = '';

  // Computed
  $: hasActiveWorkflows = $activeWorkflows.length > 0;

  // Methods
  function getStatusColor(status: string): string {
    const colors: Record<string, string> = {
      pending: 'text-gray-500',
      running: 'text-blue-500',
      paused: 'text-yellow-500',
      completed: 'text-green-500',
      failed: 'text-red-500',
      cancelled: 'text-gray-400',
    };
    return colors[status] || 'text-gray-500';
  }

  function getStatusIcon(status: string): string {
    const icons: Record<string, string> = {
      pending: '⏳',
      running: '🔄',
      paused: '⏸️',
      completed: '✅',
      failed: '❌',
      cancelled: '🚫',
    };
    return icons[status] || '❓';
  }

  function formatDuration(seconds: number | undefined): string {
    if (!seconds) return '-';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
    return `${Math.round(seconds / 3600)}h`;
  }

  async function handleCancel(workflowId: string) {
    if (confirm('Are you sure you want to cancel this workflow?')) {
      await workflowStore.cancelWorkflow(workflowId);
    }
  }

  async function handlePause(workflowId: string) {
    await workflowStore.pauseWorkflow(workflowId);
  }

  async function handleResume(workflowId: string) {
    await workflowStore.resumeWorkflow(workflowId);
  }

  function selectWorkflow(workflow: Workflow) {
    workflowStore.selectWorkflow(workflow);
  }

  function openCreateModal() {
    showCreateModal = true;
    selectedTemplate = 'video_ingest_full';
    trdContent = '';
  }

  function closeCreateModal() {
    showCreateModal = false;
  }

  async function handleCreateWorkflow() {
    if (selectedTemplate === 'trd_to_ea' && !trdContent.trim()) {
      alert('Please enter TRD content');
      return;
    }

    let workflowId: string | null = null;

    if (selectedTemplate === 'trd_to_ea') {
      workflowId = await workflowStore.startTRDToEA(trdContent);
    } else {
      // For video_ingest_full and video_ingest_ea, we need video content
      // For demo purposes, we'll create a placeholder content
      const placeholderContent = JSON.stringify({
        title: 'Demo Strategy',
        description: 'Strategy created from UI',
        template: selectedTemplate,
        created_at: new Date().toISOString(),
      });
      workflowId = await workflowStore.startWorkflowFromTemplate(selectedTemplate, {
        video_ingest_content: placeholderContent,
      });
    }

    if (workflowId) {
      closeCreateModal();
      // Refresh workflows to show the new one
      workflowStore.fetchWorkflows();
    }
  }

  // Lifecycle
  onMount(() => {
    workflowStore.fetchWorkflows();

    // Auto-refresh active workflows
    refreshInterval = window.setInterval(() => {
      $activeWorkflows.forEach((workflow) => {
        if (workflow.status === 'running') {
          workflowStore.fetchWorkflow(workflow.workflow_id);
        }
      });
    }, 3000);
  });

  onDestroy(() => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }
  });
</script>

<div class="workflow-panel h-full flex flex-col">
  <!-- Header -->
  <div class="flex items-center justify-between p-4 border-b border-gray-700">
    <h2 class="text-lg font-semibold">Workflows</h2>
    <div class="flex items-center gap-2">
      {#if $loading}
        <span class="text-sm text-gray-500">Loading...</span>
      {/if}
      <button
        class="px-3 py-1 text-sm bg-green-600 rounded hover:bg-green-700"
        on:click={openCreateModal}
      >
        + Create Workflow
      </button>
      <button
        class="px-3 py-1 text-sm bg-blue-600 rounded hover:bg-blue-700"
        on:click={() => workflowStore.fetchWorkflows()}
      >
        Refresh
      </button>
    </div>
  </div>

  <!-- Error -->
  {#if $error}
    <div class="p-4 bg-red-900/50 text-red-300 text-sm">
      {$error}
      <button class="ml-2 underline" on:click={() => workflowStore.clearError()}>
        Dismiss
      </button>
    </div>
  {/if}

  <!-- Active Workflows -->
  <div class="flex-1 overflow-auto p-4">
    {#if hasActiveWorkflows}
      <div class="space-y-4">
        <h3 class="text-sm font-medium text-gray-400 uppercase tracking-wider">
          Active Workflows
        </h3>

        {#each $activeWorkflows as workflow (workflow.workflow_id)}
          <div
            class="p-4 bg-gray-800 rounded-lg border border-gray-700 hover:border-gray-600 cursor-pointer"
            class:border-blue-500={$selectedWorkflow?.workflow_id === workflow.workflow_id}
            on:click={() => selectWorkflow(workflow)}
          >
            <div class="flex items-center justify-between mb-2">
              <div class="flex items-center gap-2">
                <span class="text-lg">{getStatusIcon(workflow.status)}</span>
                <span class="font-medium">{workflow.workflow_type}</span>
              </div>
              <span class="text-sm {getStatusColor(workflow.status)}">
                {workflow.status}
              </span>
            </div>

            <!-- Progress -->
            <div class="mb-3">
              <div class="flex justify-between text-xs text-gray-500 mb-1">
                <span>Progress</span>
                <span>{workflow.progress_percent.toFixed(0)}%</span>
              </div>
              <div class="h-2 bg-gray-700 rounded-full overflow-hidden">
                <div
                  class="h-full bg-blue-500 transition-all duration-300"
                  style="width: {workflow.progress_percent}%"
                ></div>
              </div>
            </div>

            <!-- Steps -->
            {#if workflow.steps.length > 0}
              <div class="space-y-1 text-sm">
                {#each workflow.steps as step, i}
                  <div class="flex items-center gap-2 text-gray-400">
                    <span class="w-4 text-center">
                      {#if step.status === 'completed'}
                        ✓
                      {:else if step.status === 'running'}
                        🔄
                      {:else if step.status === 'failed'}
                        ❌
                      {:else}
                        ○
                      {/if}
                    </span>
                    <span class={step.status === 'running' ? 'text-white' : ''}>
                      {step.name}
                    </span>
                    {#if step.duration_seconds}
                      <span class="text-xs text-gray-500">
                        {formatDuration(step.duration_seconds)}
                      </span>
                    {/if}
                  </div>
                {/each}
              </div>
            {/if}

            <!-- Controls -->
            {#if workflow.status === 'running' || workflow.status === 'paused'}
              <div class="mt-3 flex gap-2">
                {#if workflow.status === 'running'}
                  <button
                    class="px-3 py-1 text-xs bg-yellow-600 rounded hover:bg-yellow-700"
                    on:click|stopPropagation={() => handlePause(workflow.workflow_id)}
                  >
                    Pause
                  </button>
                {:else}
                  <button
                    class="px-3 py-1 text-xs bg-green-600 rounded hover:bg-green-700"
                    on:click|stopPropagation={() => handleResume(workflow.workflow_id)}
                  >
                    Resume
                  </button>
                {/if}
                <button
                  class="px-3 py-1 text-xs bg-red-600 rounded hover:bg-red-700"
                  on:click|stopPropagation={() => handleCancel(workflow.workflow_id)}
                >
                  Cancel
                </button>
              </div>
            {/if}

            <!-- Error -->
            {#if workflow.error}
              <div class="mt-2 text-sm text-red-400">
                {workflow.error}
              </div>
            {/if}
          </div>
        {/each}
      </div>
    {:else}
      <div class="text-center text-gray-500 py-8">
        <p>No active workflows</p>
        <p class="text-sm mt-1">Start a workflow from the VideoIngest panel</p>
      </div>
    {/if}
  </div>

  <!-- Selected Workflow Details -->
  {#if $selectedWorkflow}
    <div class="border-t border-gray-700 p-4 bg-gray-800/50">
      <h4 class="font-medium mb-2">Workflow Details</h4>
      <div class="text-sm space-y-1">
        <p><span class="text-gray-500">ID:</span> {$selectedWorkflow.workflow_id}</p>
        <p><span class="text-gray-500">Type:</span> {$selectedWorkflow.workflow_type}</p>
        <p><span class="text-gray-500">Duration:</span> {formatDuration($selectedWorkflow.duration_seconds)}</p>
      </div>

      {#if Object.keys($selectedWorkflow.final_result).length > 0}
        <div class="mt-3">
          <h5 class="text-sm font-medium text-gray-400 mb-1">Results</h5>
          <pre class="text-xs bg-gray-900 p-2 rounded overflow-auto max-h-32">
{JSON.stringify($selectedWorkflow.final_result, null, 2)}
          </pre>
        </div>
      {/if}
    </div>
  {/if}
</div>

<!-- Create Workflow Modal -->
{#if showCreateModal}
  <div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50" on:click={closeCreateModal}>
    <div class="bg-gray-800 rounded-lg p-6 w-full max-w-md" on:click|stopPropagation>
      <h3 class="text-lg font-semibold mb-4">Create New Workflow</h3>

      <!-- Template Selection -->
      <div class="mb-4">
        <label class="block text-sm font-medium text-gray-400 mb-2">
          Select Workflow Template
        </label>
        <select
          bind:value={selectedTemplate}
          class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
        >
          {#each Object.entries(WORKFLOW_TEMPLATES) as [key, template]}
            <option value={key}>{template.label}</option>
          {/each}
        </select>
        <p class="mt-1 text-sm text-gray-500">
          {WORKFLOW_TEMPLATES[selectedTemplate]?.description}
        </p>
      </div>

      <!-- TRD Content (only for TRD to EA template) -->
      {#if selectedTemplate === 'trd_to_ea'}
        <div class="mb-4">
          <label class="block text-sm font-medium text-gray-400 mb-2">
            TRD Content
          </label>
          <textarea
            bind:value={trdContent}
            rows="6"
            placeholder="Paste your TRD (Trading Robot Description) content here..."
            class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white placeholder-gray-500"
          ></textarea>
        </div>
      {/if}

      <!-- Action Buttons -->
      <div class="flex justify-end gap-3">
        <button
          class="px-4 py-2 text-sm bg-gray-600 rounded hover:bg-gray-700"
          on:click={closeCreateModal}
        >
          Cancel
        </button>
        <button
          class="px-4 py-2 text-sm bg-green-600 rounded hover:bg-green-700"
          on:click={handleCreateWorkflow}
          disabled={$loading}
        >
          {$loading ? 'Creating...' : 'Create Workflow'}
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .workflow-panel {
    min-width: 300px;
  }
</style>

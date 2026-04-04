<script lang="ts">
  /**
   * FlowForgeNodeGraph Component
   *
   * SVG dependency graph for Prefect workflow tasks.
   * Features:
   * - Task boxes colored by state
   * - Directed edges (dependencies)
   * - Zoom + pan controls
   * - Minimap
   * - Node selection with detail tooltip
   * - Failed node red highlight with error detail panel (Story 11.8 AC7)
   */

  import type { PrefectWorkflow, PrefectTask, NodeErrorInfo } from '$lib/stores/flowforge';
  import { X, ZoomIn, ZoomOut, Maximize2, Map, AlertTriangle, RefreshCw } from 'lucide-svelte';
  import { onMount } from 'svelte';

  interface Props {
    workflow: PrefectWorkflow;
    onClose: () => void;
    nodeErrors?: NodeErrorInfo[];  // Story 11.8: Errors from SSE events
  }

  let { workflow, onClose, nodeErrors = [] }: Props = $props();

  // SVG dimensions
  const SVG_WIDTH = 1000;
  const SVG_HEIGHT = 400;
  const NODE_WIDTH = 120;
  const NODE_HEIGHT = 40;

  // Viewbox state
  let viewBox = $state({ x: 0, y: 0, width: SVG_WIDTH, height: SVG_HEIGHT });
  let scale = $state(1);
  let isPanning = $state(false);
  let panStart = $state({ x: 0, y: 0 });

  // Selected node
  let selectedTask = $state<PrefectTask | null>(null);
  let tooltipPosition = $state({ x: 0, y: 0 });

  // Show/hide minimap
  let showMinimap = $state(true);

  // Story 11.8 AC7: Error panel state
  let showErrorPanel = $state(false);
  let selectedError = $state<NodeErrorInfo | null>(null);

  // State colors
  const stateColors: Record<string, string> = {
    PENDING: '#94a3b8',
    RUNNING: '#06b6d4',
    COMPLETED: '#22c55e',
    CANCELLED: '#ef4444',
    FAILED: '#dc2626',
  };

  // Story 11.8 AC7: Get error for a specific task
  function getErrorForTask(taskId: string): NodeErrorInfo | undefined {
    return nodeErrors.find((e) => e.taskName === taskId || e.taskName === workflow.tasks.find((t) => t.id === taskId)?.name);
  }

  // Story 11.8 AC7: Check if a task has an error
  function hasError(taskId: string): boolean {
    return getErrorForTask(taskId) !== undefined;
  }

  // Story 11.8 AC7: Open error panel for a task
  function openErrorPanel(task: PrefectTask, event: MouseEvent | KeyboardEvent) {
    const error = getErrorForTask(task.id);
    if (error) {
      selectedError = error;
      showErrorPanel = true;
      // Prevent node selection when clicking error
      event.stopPropagation();
    } else {
      // Normal node selection
      handleNodeClick(task, event);
    }
  }

  // Calculate viewBox based on scale - properly center content
  function updateViewBox() {
    const newWidth = SVG_WIDTH / scale;
    const newHeight = SVG_HEIGHT / scale;
    // Center the view based on current center point
    const centerX = viewBox.x + viewBox.width / 2;
    const centerY = viewBox.y + viewBox.height / 2;
    const newX = centerX - newWidth / 2;
    const newY = centerY - newHeight / 2;
    viewBox = { x: newX, y: newY, width: newWidth, height: newHeight };
  }

  // Zoom controls
  function zoomIn() {
    scale = Math.min(scale * 1.3, 4);
    updateViewBox();
  }

  function zoomOut() {
    scale = Math.max(scale / 1.3, 0.5);
    updateViewBox();
  }

  function resetZoom() {
    scale = 1;
    viewBox = { x: 0, y: 0, width: SVG_WIDTH, height: SVG_HEIGHT };
  }

  // Pan handling
  function handleMouseDown(e: MouseEvent) {
    if (e.button === 0) {
      isPanning = true;
      panStart = { x: e.clientX, y: e.clientY };
    }
  }

  function handleMouseMove(e: MouseEvent) {
    if (isPanning) {
      const dx = (e.clientX - panStart.x) * scale;
      const dy = (e.clientY - panStart.y) * scale;
      viewBox = {
        ...viewBox,
        x: viewBox.x - dx,
        y: viewBox.y - dy,
      };
      panStart = { x: e.clientX, y: e.clientY };
    }
  }

  function handleMouseUp() {
    isPanning = false;
  }

  // Node selection - handle both mouse and keyboard
  function handleNodeClick(task: PrefectTask, event: MouseEvent | KeyboardEvent) {
    event.stopPropagation();
    selectedTask = task;
    // Get the target element regardless of event type
    const target = event.currentTarget as Element;
    const rect = target.getBoundingClientRect();
    tooltipPosition = {
      x: rect.left + rect.width / 2,
      y: rect.top - 10,
    };
  }

  function handleNodeKeydown(task: PrefectTask, e: KeyboardEvent) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleNodeClick(task, e);
    }
  }

  function handleCanvasClick() {
    selectedTask = null;
  }

  // Generate edge path between tasks
  function getEdgePath(fromTask: PrefectTask, toTask: PrefectTask): string {
    const fromX = fromTask.x + NODE_WIDTH;
    const fromY = fromTask.y + NODE_HEIGHT / 2;
    const toX = toTask.x;
    const toY = toTask.y + NODE_HEIGHT / 2;

    // Curved path
    const midX = (fromX + toX) / 2;
    return `M ${fromX} ${fromY} C ${midX} ${fromY}, ${midX} ${toY}, ${toX} ${toY}`;
  }

  // Find task by ID
  function getTaskById(taskId: string): PrefectTask | undefined {
    return workflow.tasks.find((t) => t.id === taskId);
  }

  // Minimap calculations
  const minimapScale = 0.1;
  const minimapWidth = SVG_WIDTH * minimapScale;
  const minimapHeight = SVG_HEIGHT * minimapScale;
  const minimapViewBox = $derived(
    ` ${viewBox.x * minimapScale} ${viewBox.y * minimapScale} ${viewBox.width * minimapScale} ${minimapHeight}`
  );

  onMount(() => {
    // Auto-fit to content on mount
    if (workflow.tasks.length > 0) {
      const minX = Math.min(...workflow.tasks.map((t) => t.x)) - 50;
      const maxX = Math.max(...workflow.tasks.map((t) => t.x + NODE_WIDTH)) + 50;
      const contentWidth = maxX - minX;
      if (contentWidth > SVG_WIDTH) {
        scale = SVG_WIDTH / contentWidth;
        updateViewBox();
      }
    }
  });
</script>

<div class="node-graph-overlay" onclick={onClose} role="dialog" aria-modal="true">
  <div class="node-graph-container" onclick={(e) => e.stopPropagation()}>
    <!-- Header -->
    <div class="graph-header">
      <div class="header-info">
        <h3>{workflow.name}</h3>
        <span class="department">{workflow.department}</span>
      </div>
      <div class="header-actions">
        <button class="icon-btn" onclick={() => (showMinimap = !showMinimap)} title="Toggle Minimap">
          <Map size={18} />
        </button>
        <button class="icon-btn" onclick={onClose} title="Close">
          <X size={20} />
        </button>
      </div>
    </div>

    <!-- Toolbar -->
    <div class="graph-toolbar">
      <button class="toolbar-btn" onclick={zoomIn} title="Zoom In">
        <ZoomIn size={16} />
      </button>
      <span class="zoom-level">{Math.round(scale * 100)}%</span>
      <button class="toolbar-btn" onclick={zoomOut} title="Zoom Out">
        <ZoomOut size={16} />
      </button>
      <button class="toolbar-btn" onclick={resetZoom} title="Reset View">
        <Maximize2 size={16} />
      </button>
    </div>

    <!-- SVG Graph -->
    <div
      class="graph-canvas"
      onmousedown={handleMouseDown}
      onmousemove={handleMouseMove}
      onmouseup={handleMouseUp}
      onmouseleave={handleMouseUp}
      onclick={handleCanvasClick}
      role="img"
      aria-label="Workflow node graph"
    >
      <svg
        viewBox="{viewBox.x} {viewBox.y} {viewBox.width} {viewBox.height}"
        preserveAspectRatio="xMidYMid meet"
      >
        <!-- Edges (Dependencies) -->
        <g class="edges">
          {#each workflow.dependencies as dep}
            {@const fromTask = getTaskById(dep.from)}
            {@const toTask = getTaskById(dep.to)}
            {#if fromTask && toTask}
              <path
                class="edge"
                d={getEdgePath(fromTask, toTask)}
                stroke={stateColors[toTask.state] || '#94a3b8'}
                stroke-width="2"
                fill="none"
                opacity="0.6"
              />
            {/if}
          {/each}
        </g>

        <!-- Nodes (Tasks) -->
        <g class="nodes">
          {#each workflow.tasks as task}
            {@const hasError = hasError(task.id)}
            <g
              class="task-node"
              class:selected={selectedTask?.id === task.id}
              class:has-error={hasError}
              transform="translate({task.x}, {task.y})"
              onclick={(e) => openErrorPanel(task, e)}
              onkeydown={(e) => handleNodeKeydown(task, e)}
              role="button"
              tabindex="0"
            >
              <!-- Story 11.8 AC7: Failed node red highlight -->
              <rect
                width={NODE_WIDTH}
                height={NODE_HEIGHT}
                rx="6"
                class="task-rect state-{task.state.toLowerCase()}"
                class:error-rect={hasError}
                fill="rgba(30, 32, 40, 0.9)"
                stroke={hasError ? '#ef4444' : stateColors[task.state]}
                stroke-width={hasError ? '3' : '2'}
              />
              <text
                x={NODE_WIDTH / 2}
                y={NODE_HEIGHT / 2 + 4}
                text-anchor="middle"
                class="task-label"
                fill="#f1f5f9"
              >
                {task.name.length > 14 ? task.name.slice(0, 12) + '...' : task.name}
              </text>

              <!-- Running animation -->
              {#if task.state === 'RUNNING'}
                <rect
                  x="0"
                  y="0"
                  width={NODE_WIDTH}
                  height={NODE_HEIGHT}
                  rx="6"
                  class="running-pulse"
                  stroke="#06b6d4"
                  stroke-width="2"
                  fill="none"
                />
              {/if}

              <!-- Story 11.8 AC7: Error indicator -->
              {#if hasError}
                <g class="error-indicator" transform="translate({NODE_WIDTH - 20}, -8)">
                  <circle r="10" fill="#ef4444" />
                  <AlertTriangle size="12" x="-6" y="-6" color="white" />
                </g>
              {/if}
            </g>
          {/each}
        </g>
      </svg>

      <!-- Tooltip for selected node -->
      {#if selectedTask}
        <div
          class="node-tooltip"
          style="left: {tooltipPosition.x}px; top: {tooltipPosition.y}px"
        >
          <div class="tooltip-header">
            <span class="task-name">{selectedTask.name}</span>
            <span class="task-state" style="color: {stateColors[selectedTask.state]}">
              {selectedTask.state}
            </span>
          </div>
          <div class="tooltip-body">
            <span>ID: {selectedTask.id}</span>
          </div>
        </div>
      {/if}

      <!-- Story 11.8 AC7: Error detail panel -->
      {#if showErrorPanel && selectedError}
        <div class="error-panel">
          <div class="error-panel-header">
            <div class="error-title">
              <AlertTriangle size={16} color="#ef4444" />
              <span>Task Error</span>
            </div>
            <button class="close-btn" onclick={() => { showErrorPanel = false; selectedError = null; }}>
              <X size={16} />
            </button>
          </div>
          <div class="error-panel-body">
            <div class="error-field">
              <span class="error-label">Task:</span>
              <span class="error-value">{selectedError.taskName}</span>
            </div>
            <div class="error-field">
              <span class="error-label">Error Type:</span>
              <span class="error-value error-type">{selectedError.errorType}</span>
            </div>
            <div class="error-field">
              <span class="error-label">Message:</span>
              <span class="error-value error-message">{selectedError.errorMessage}</span>
            </div>
            <div class="error-field">
              <span class="error-label">Retry Count:</span>
              <span class="error-value">
                <RefreshCw size={12} />
                {selectedError.retryCount}
              </span>
            </div>
          </div>
        </div>
      {/if}
    </div>

    <!-- Minimap -->
    {#if showMinimap}
      <div class="minimap">
        <svg viewBox={minimapViewBox} preserveAspectRatio="xMidYMid meet">
          <!-- Minimap content -->
          <g class="minimap-nodes">
            {#each workflow.tasks as task}
              <rect
                x={task.x * minimapScale}
                y={task.y * minimapScale}
                width={NODE_WIDTH * minimapScale}
                height={NODE_HEIGHT * minimapScale}
                rx="2"
                fill={stateColors[task.state]}
                opacity="0.7"
              />
            {/each}
          </g>
        </svg>
        <div class="minimap-viewport" style="
          left: {(-viewBox.x / SVG_WIDTH) * 100}%;
          top: {(-viewBox.y / SVG_HEIGHT) * 100}%;
          width: {(viewBox.width / SVG_WIDTH) * 100}%;
          height: {(viewBox.height / SVG_HEIGHT) * 100}%;
        "></div>
      </div>
    {/if}
  </div>
</div>

<style>
  .node-graph-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.8);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .node-graph-container {
    background: rgba(20, 22, 28, 0.95);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    width: 90%;
    max-width: 1100px;
    height: 80%;
    max-height: 600px;
    display: flex;
    flex-direction: column;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
  }

  .graph-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  }

  .header-info h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    color: #f1f5f9;
  }

  .header-info .department {
    font-size: 12px;
    color: #94a3b8;
  }

  .header-actions {
    display: flex;
    gap: 8px;
  }

  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: rgba(255, 255, 255, 0.08);
    border: none;
    border-radius: 6px;
    color: #94a3b8;
    cursor: pointer;
    transition: all 0.2s;
  }

  .icon-btn:hover {
    background: rgba(255, 255, 255, 0.12);
    color: #f1f5f9;
  }

  .graph-toolbar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background: rgba(255, 255, 255, 0.04);
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  }

  .toolbar-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: rgba(255, 255, 255, 0.08);
    border: none;
    border-radius: 4px;
    color: #94a3b8;
    cursor: pointer;
    transition: all 0.2s;
  }

  .toolbar-btn:hover {
    background: rgba(255, 255, 255, 0.12);
    color: #f1f5f9;
  }

  .zoom-level {
    font-size: 12px;
    color: #64748b;
    min-width: 40px;
    text-align: center;
  }

  .graph-canvas {
    flex: 1;
    overflow: hidden;
    cursor: grab;
    position: relative;
    background: linear-gradient(135deg, rgba(30, 32, 40, 0.5) 0%, rgba(20, 22, 28, 0.8) 100%);
  }

  .graph-canvas:active {
    cursor: grabbing;
  }

  .graph-canvas svg {
    width: 100%;
    height: 100%;
  }

  .edge {
    pointer-events: none;
  }

  .task-node {
    cursor: pointer;
    transition: transform 0.2s;
  }

  .task-node:hover .task-rect {
    filter: brightness(1.2);
  }

  .task-node.selected .task-rect {
    stroke-width: 3;
    filter: drop-shadow(0 0 8px rgba(6, 182, 212, 0.5));
  }

  .task-label {
    font-size: 11px;
    font-weight: 500;
    pointer-events: none;
  }

  .running-pulse {
    animation: nodePulse 1.5s ease-in-out infinite;
  }

  @keyframes nodePulse {
    0%, 100% {
      opacity: 0.8;
      transform: scale(1);
    }
    50% {
      opacity: 0.3;
      transform: scale(1.05);
    }
  }

  .node-tooltip {
    position: absolute;
    background: rgba(30, 32, 40, 0.95);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 8px;
    padding: 10px 14px;
    transform: translate(-50%, -100%);
    z-index: 10;
    min-width: 150px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
  }

  .tooltip-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 6px;
  }

  .tooltip-header .task-name {
    font-weight: 600;
    color: #f1f5f9;
    font-size: 13px;
  }

  .tooltip-header .task-state {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
  }

  .tooltip-body {
    font-size: 11px;
    color: #94a3b8;
  }

  .minimap {
    position: relative;
    width: 150px;
    height: 60px;
    margin: 8px 16px;
    background: rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    overflow: hidden;
  }

  .minimap svg {
    width: 100%;
    height: 100%;
  }

  .minimap-viewport {
    position: absolute;
    border: 2px solid #06b6d4;
    background: rgba(6, 182, 212, 0.1);
    pointer-events: none;
  }

  /* Story 11.8 AC7: Error highlighting */
  .task-node.has-error .task-rect {
    filter: drop-shadow(0 0 8px rgba(239, 68, 68, 0.6));
  }

  .task-node.has-error .task-label {
    fill: #fca5a5;
  }

  .error-rect {
    animation: errorPulse 2s ease-in-out infinite;
  }

  @keyframes errorPulse {
    0%, 100% {
      stroke-opacity: 1;
    }
    50% {
      stroke-opacity: 0.6;
    }
  }

  .error-indicator {
    cursor: pointer;
    animation: errorBounce 1s ease-in-out infinite;
  }

  @keyframes errorBounce {
    0%, 100% {
      transform: translate(100px, -8px);
    }
    50% {
      transform: translate(100px, -12px);
    }
  }

  /* Story 11.8 AC7: Error detail panel */
  .error-panel {
    position: absolute;
    top: 60px;
    right: 20px;
    width: 320px;
    background: rgba(30, 32, 40, 0.95);
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 8px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4), 0 0 20px rgba(239, 68, 68, 0.2);
    z-index: 100;
    overflow: hidden;
  }

  .error-panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 14px;
    background: rgba(239, 68, 68, 0.15);
    border-bottom: 1px solid rgba(239, 68, 68, 0.3);
  }

  .error-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
    font-size: 14px;
    color: #fca5a5;
  }

  .close-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: rgba(255, 255, 255, 0.08);
    border: none;
    border-radius: 4px;
    color: #94a3b8;
    cursor: pointer;
    transition: all 0.2s;
  }

  .close-btn:hover {
    background: rgba(255, 255, 255, 0.12);
    color: #f1f5f9;
  }

  .error-panel-body {
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .error-field {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .error-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    color: #64748b;
    letter-spacing: 0.5px;
  }

  .error-value {
    font-size: 13px;
    color: #f1f5f9;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .error-type {
    color: #fca5a5;
    font-family: monospace;
    font-size: 12px;
  }

  .error-message {
    color: #f87171;
    word-break: break-word;
  }
</style>
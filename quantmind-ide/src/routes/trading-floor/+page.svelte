<script lang="ts">
  /**
   * Trading Floor Demo Page
   *
   * Showcases the Trading Floor visualization with sample agents.
   */
  import { onMount, onDestroy } from 'svelte';
  import TradingFloorCanvas from '$lib/components/trading-floor/TradingFloorCanvas.svelte';
  import {
    tradingFloorStore,
    updateAgentState,
    addSubAgent,
    sendMail,
    reset,
    type AgentState
  } from '$lib/stores/tradingFloorStore';
  import { connectTradingFloorWS, disconnectTradingFloorWS, tradingFloorWS } from '$lib/services/tradingFloorWebSocket';

  // Demo state
  let isRunning = $state(false);
  let demoInterval: number | null = null;

  // Sample departments
  const departments = ['development', 'research', 'risk', 'trading', 'portfolio'];

  // Sample tasks
  const sampleTasks = [
    { from: 'development', to: 'trading', type: 'dispatch', subject: 'EURUSD Signal: BUY' },
    { from: 'research', to: 'development', type: 'result', subject: 'Backtest Complete' },
    { from: 'risk', to: 'trading', type: 'question', subject: 'Position Size Query' },
    { from: 'development', to: 'portfolio', type: 'dispatch', subject: 'Market Update' },
    { from: 'trading', to: 'risk', type: 'status', subject: 'Order Filled' },
  ];

  onMount(() => {
    // Initialize demo
    initializeDemo();

    // Connect WebSocket
    connectTradingFloorWS();
  });

  onDestroy(() => {
    stopDemo();
    disconnectTradingFloorWS();
  });

  function initializeDemo() {
    // Reset store
    reset();

    // Add department heads
    const positions = {
      development: { x: 170, y: 130 },
      research: { x: 370, y: 130 },
      risk: { x: 570, y: 130 },
      trading: { x: 270, y: 300 },
      portfolio: { x: 470, y: 300 },
    };

    departments.forEach((dept) => {
      const agent: AgentState = {
        id: `${dept}-head`,
        name: `${dept.charAt(0).toUpperCase() + dept.slice(1)} Head`,
        department: dept,
        status: 'idle',
        position: positions[dept as keyof typeof positions],
        target: null,
        subAgents: [],
        isExpanded: false,
      };
      addSubAgent('floor-manager', agent);
    });
  }

  function startDemo() {
    if (isRunning) return;
    isRunning = true;

    demoInterval = window.setInterval(() => {
      runDemoStep();
    }, 2000);
  }

  function stopDemo() {
    isRunning = false;
    if (demoInterval) {
      clearInterval(demoInterval);
      demoInterval = null;
    }
  }

  let stepCount = 0;
  function runDemoStep() {
    stepCount++;

    // Cycle through agents and update their state
    const agentStates: AgentState['status'][] = ['thinking', 'typing', 'reading', 'idle'];
    const randomDept = departments[stepCount % departments.length];
    const randomState = agentStates[stepCount % agentStates.length];

    updateAgentState(`${randomDept}-head`, {
      status: randomState,
      speechBubble: randomState === 'thinking'
        ? { text: 'Analyzing...', type: 'thinking', duration: 2000 }
        : undefined,
    });

    // Occasionally send mail
    if (stepCount % 3 === 0) {
      const task = sampleTasks[stepCount % sampleTasks.length];
      sendMail({
        id: `mail-${Date.now()}`,
        fromDept: task.from,
        toDept: task.to,
        type: task.type as any,
        subject: task.subject,
        startX: 0,
        startY: 0,
        progress: 0,
        duration: 1500,
      });
    }

    // Occasionally spawn sub-agent
    if (stepCount % 5 === 0) {
      const dept = departments[stepCount % departments.length];
      const workerTypes = {
        development: 'market_analyst',
        research: 'backtester',
        risk: 'position_sizer',
        trading: 'order_router',
        portfolio: 'rebalancer',
      };

      const parentPos = getPositionForDept(dept);
      const subAgent: AgentState = {
        id: `${dept}-worker-${stepCount}`,
        name: workerTypes[dept as keyof typeof workerTypes] || 'worker',
        department: dept,
        status: 'thinking',
        position: {
          x: parentPos.x + (Math.random() * 60 - 30),
          y: parentPos.y + (Math.random() * 60 - 30),
        },
        target: null,
        subAgents: [],
        isExpanded: false,
      };

      addSubAgent(`${dept}-head`, subAgent);
    }
  }

  function getPositionForDept(dept: string): { x: number; y: number } {
    const positions: Record<string, { x: number; y: number }> = {
      development: { x: 170, y: 130 },
      research: { x: 370, y: 130 },
      risk: { x: 570, y: 130 },
      trading: { x: 270, y: 300 },
      portfolio: { x: 470, y: 300 },
    };
    return positions[dept] || { x: 300, y: 200 };
  }

  // Subscribe to store
  let floorStats = $derived($tradingFloorStore?.floorStats || { totalTasks: 0, activeTasks: 0 });
  let wsConnected = $derived(tradingFloorWS.connected || false);
</script>

<div class="trading-floor-demo">
  <header class="demo-header">
    <h1>Trading Floor Demo</h1>
    <p class="subtitle">Department-Based Agent Framework</p>
  </header>

  <div class="demo-controls">
    <button class="btn" onclick={startDemo} disabled={isRunning}>
      {isRunning ? 'Running...' : 'Start Demo'}
    </button>
    <button class="btn btn-secondary" onclick={stopDemo} disabled={!isRunning}>
      Stop
    </button>
    <button class="btn btn-outline" onclick={initializeDemo}>
      Reset
    </button>
  </div>

  <div class="demo-stats">
    <div class="stat">
      <span class="stat-label">Active</span>
      <span class="stat-value">{floorStats.activeTasks}</span>
    </div>
    <div class="stat">
      <span class="stat-label">Total</span>
      <span class="stat-value">{floorStats.totalTasks}</span>
    </div>
    <div class="stat">
      <span class="stat-label">WS</span>
      <span class="stat-value ws-indicator" class:connected={wsConnected}>
        {wsConnected ? '●' : '○'}
      </span>
    </div>
  </div>

  <div class="canvas-container">
    <TradingFloorCanvas />
  </div>

  <footer class="demo-footer">
    <p>Click on agents or departments to select. Demo shows 5 department heads with dynamic task routing.</p>
  </footer>
</div>

<style>
  .trading-floor-demo {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: #0f172a;
    color: #e2e8f0;
    padding: 1rem;
  }

  .demo-header {
    text-align: center;
    margin-bottom: 1rem;
  }

  .demo-header h1 {
    font-size: 1.5rem;
    font-weight: 600;
    margin: 0;
  }

  .subtitle {
    font-size: 0.875rem;
    color: #94a3b8;
    margin: 0.25rem 0 0;
  }

  .demo-controls {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }

  .btn {
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    border: none;
    background: #3b82f6;
    color: white;
  }

  .btn:hover:not(:disabled) {
    background: #2563eb;
  }

  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn-secondary {
    background: #475569;
  }

  .btn-secondary:hover:not(:disabled) {
    background: #64748b;
  }

  .btn-outline {
    background: transparent;
    border: 1px solid #475569;
    color: #94a3b8;
  }

  .btn-outline:hover:not(:disabled) {
    background: #1e293b;
  }

  .demo-stats {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-bottom: 1rem;
  }

  .stat {
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .stat-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
  }

  .stat-value {
    font-size: 1.25rem;
    font-weight: 600;
  }

  .ws-indicator {
    color: #ef4444;
  }

  .ws-indicator.connected {
    color: #22c55e;
  }

  .canvas-container {
    flex: 1;
    display: flex;
    justify-content: center;
    align-items: center;
    background: #1e293b;
    border-radius: 0.5rem;
    overflow: hidden;
  }

  .demo-footer {
    text-align: center;
    padding-top: 1rem;
    font-size: 0.75rem;
    color: #64748b;
  }
</style>

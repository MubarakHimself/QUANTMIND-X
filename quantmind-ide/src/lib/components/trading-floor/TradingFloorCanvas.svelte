<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import {
    agents,
    departments,
    mailMessages,
    animatingMail,
    selectedAgent,
    floorStats,
    selectAgent,
    type AgentState,
    type DepartmentPosition,
    type MailMessage
  } from '$lib/stores/tradingFloorStore';

  const CANVAS_WIDTH = 1000;
  const CANVAS_HEIGHT = 600;

  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D | null = null;
  let animationFrame: number = 0;

  const COLORS = {
    background: '#1a1a2a',
    desk: '#2d3740',
    deskHover: '#374151',
    floor: '#1e1a2a',
    analysis: '#3b82f6',
    research: '#8b5cf6',
    risk: '#ef4444',
    execution: '#f97316',
    portfolio: '#10b981',
    agent: {
      idle: '#6b7280',
      thinking: '#60a5fa',
      typing: '#fbbf24',
      reading: '#fde68a',
      walking: '#34d399',
      spawning: '#9333ea',
    },
    mail: {
      dispatch: '#3b82f6',
      result: '#10b981',
      question: '#f59e0b',
      status: '#6b7280',
    }
  };

  const AGENT_SIZE = 40;
  const AGENT_COLORS: Record<string, string> = {
    analysis: '#3b82f6',
    research: '#8b5cf6',
    risk: '#ef4444',
    execution: '#f97316',
    portfolio: '#10b981',
    coordination: '#6366f1',
  };

  onMount(() => {
    if (!canvas) return;
    ctx = canvas.getContext('2d');
    startAnimationLoop();
  });

  onDestroy(() => {
    cancelAnimationFrame(animationFrame);
  });

  function startAnimationLoop() {
    function loop() {
      render();
      animationFrame = requestAnimationFrame(loop);
    }
    animationFrame = requestAnimationFrame(loop);
  }

  function render() {
    if (!ctx) return;

    const $agents = get(agents);
    const $departments = get(departments);
    const $animatingMail = get(animatingMail);
    const $selectedAgent = get(selectedAgent);
    const $floorStats = get(floorStats);

    // Clear canvas
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    // Draw floor
    ctx.fillStyle = COLORS.floor;
    ctx.fillRect(50, 50, CANVAS_WIDTH - 100, 30);

    // Draw departments
    $departments.forEach((dept, key) => {
      drawDepartment(dept, key, $selectedAgent);
    });

    // Draw agents
    $agents.forEach((agent) => {
      drawAgent(agent, $selectedAgent);
    });

    // Draw mail animation
    if ($animatingMail) {
      drawMail($animatingMail, $departments);
    }

    // Draw stats
    renderStats($floorStats);
  }

  function drawDepartment(dept: DepartmentPosition, key: string, selectedId: string | null) {
    const isHovered = selectedId === key;
    const color = isHovered ? COLORS.deskHover : COLORS.desk;
    const name = key.charAt(0).toUpperCase() + key.slice(1);

    // Desk shadow
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.fillRect(dept.x + 2, dept.y + 2, dept.width, dept.height);

    // Desk surface
    ctx.fillStyle = color;
    ctx.fillRect(dept.x, dept.y, dept.width, dept.height);

    // Department label
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(name, dept.x + dept.width / 2, dept.y + 30);
  }

  function drawAgent(agent: AgentState, selectedId: string | null) {
    const { x, y } = agent.position;
    const isSelected = selectedId === agent.id;
    const color = AGENT_COLORS[agent.department] || AGENT_COLORS.coordination;
    const statusColor = COLORS.agent[agent.status] || COLORS.agent.idle;

    // Agent body (circle)
    ctx.beginPath();
    ctx.arc(x, y, AGENT_SIZE / 2, 0, 2 * Math.PI);
    ctx.fillStyle = isSelected ? '#fbbf24' : color;
    ctx.fill();

    // Status indicator
    ctx.beginPath();
    ctx.arc(x + AGENT_SIZE / 2 + 5, y - AGENT_SIZE / 2 + 5, 5, 0, 2 * Math.PI);
    ctx.fillStyle = statusColor;
    ctx.fill();

    // Agent name
    ctx.fillStyle = '#ffffff';
    ctx.font = '10px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(agent.name, x, y + AGENT_SIZE + 15);

    // Sub-agents indicator
    if (agent.subAgents && agent.subAgents.length > 0) {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
      ctx.beginPath();
      ctx.arc(x + AGENT_SIZE / 2 + 10, y + AGENT_SIZE / 2 + 10, 8, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.font = '8px Arial';
      ctx.fillText(`${agent.subAgents.length}`, x + AGENT_SIZE / 2 + 10, y + AGENT_SIZE / 2 + 13);
    }
  }

  function drawMail(mail: MailMessage, depts: Map<string, DepartmentPosition>) {
    const fromDept = depts.get(mail.fromDept);
    const toDept = depts.get(mail.toDept);

    if (!fromDept || !toDept) return;

    const progress = mail.progress / mail.duration;
    const currentX = fromDept.x + (toDept.x - fromDept.x) * progress;
    const currentY = fromDept.y + (toDept.y - fromDept.y) * progress;

    // Mail trail
    ctx.strokeStyle = '#fef3c7';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(fromDept.x + fromDept.width / 2, fromDept.y + fromDept.height / 2);
    ctx.lineTo(currentX, currentY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Mail envelope
    ctx.fillStyle = '#fef3c7';
    ctx.beginPath();
    const envelopeSize = 20;
    ctx.ellipse(currentX, currentY, envelopeSize / 2, envelopeSize * 0.3, 0, 0, 2 * Math.PI);
    ctx.fill();

    // Mail type icon
    const mailColor = mail.type === 'dispatch' ? '#3b82f6' :
                      mail.type === 'result' ? '#10b981' : '#f59e0b';
    ctx.fillStyle = mailColor;
    ctx.beginPath();
    ctx.arc(currentX, currentY, 5, 0, 2 * Math.PI);
    ctx.fill();
  }

  function renderStats(stats: { totalTasks: number; activeTasks: number; completedTasks: number; pendingMail: number }) {
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('Trading Floor', 20, 30);

    ctx.font = '12px Arial';
    ctx.fillText(`Active: ${stats.activeTasks} | Total: ${stats.totalTasks}`, 20, CANVAS_HEIGHT - 10);
  }

  function handleClick(event: MouseEvent) {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const $departments = get(departments);
    const $agents = get(agents);

    // Check if clicked on department
    let clickedDept: string | null = null;
    $departments.forEach((dept, key) => {
      if (
        x >= dept.x && x <= dept.x + dept.width &&
        y >= dept.y && y <= dept.y + dept.height
      ) {
        clickedDept = key;
      }
    });

    // Check if clicked on agent
    let clickedAgent: AgentState | null = null;
    $agents.forEach((agent) => {
      const dx = x - agent.position.x;
      const dy = y - agent.position.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      if (distance <= AGENT_SIZE / 2) {
        clickedAgent = agent;
      }
    });

    if (clickedAgent) {
      selectAgent(clickedAgent.id);
    } else if (clickedDept) {
      selectAgent(clickedDept);
    }
  }

  // Import get from svelte/store
  import { get } from 'svelte/store';
</script>

<div class="trading-floor-canvas">
  <canvas
    bind:this={canvas}
    width={CANVAS_WIDTH}
    height={CANVAS_HEIGHT}
    on:click={handleClick}
  ></canvas>
</div>

<style>
  .trading-floor-canvas {
    width: 100%;
    height: 600px;
    background: #1a1a2a;
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  canvas {
    max-width: 100%;
    max-height: 100%;
  }
</style>

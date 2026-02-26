<script lang="ts">
  import { onMount, onDestroy, tick } from 'svelte';
  import { tradingFloorStore } from '$lib/stores/tradingFloorStore';

  const CANVAS_WIDTH = 1000;
  const CANVAS_HEIGHT = 600;

  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D | null;
  let animationFrame: number = 0;

  $: {
    agents: Map<string, AgentState>;
    departments: Map<string, DepartmentPosition>;
    mailMessages: MailMessage[];
    animatingMail: MailMessage | null;
    selectedAgent: string | null;
    floorStats: { totalTasks: number; activeTasks: number; completedTasks: number; pendingMail: number };
  } = tradingFloorStore;

  const COLORS = {
    background: '#1a1a2a',
    desk: '#2d3740',
    deskHover: '#374151',
    floor: '#1e1a2a',
    analysis: '#3b82f6',
    research: '#8b5cf6',
    risk: '#ef4444',
    execution: '#f97316',
    portfolio: '#10b981a',
    agent: {
      idle: '#6b7280',
      thinking: '#60a5fa',
      typing: '#fbbf7',
      reading: '#fde68a',
      walking: '#34d399',
      spawning: '#9333ea',
    },
    mail: {
      dispatch: '#3b82f6',
      result: '#10b981a',
      question: '#f59e00',
      status: '#6b7280',
    }
  };

  const AGENT_SIZE = 40;
  const AGENT_COLORS: Record<string, string> = {
    analysis: '#3b82f6',
    research: '#8b5cf6',
    risk: '#ef4444',
    execution: '#f97316',
    portfolio: '#10b981a',
    coordination: '#6366f1',
  };

  onMount(() => {
    if (!canvas) return;
    ctx = canvas.getContext('2d');

    // Subscribe to store
    const unsubAgents = tradingFloorStore.agents.subscribe((value) => {
      agents = value;
      render();
    });

    const unsubDepts = tradingFloorStore.departments.subscribe((value) => {
      departments = value;
      render();
    });

    const unsubMail = tradingFloorStore.animatingMail.subscribe((value) => {
      animatingMail = value;
      render();
    });

    const unsubStats = tradingFloorStore.floorStats.subscribe((value) => {
      floorStats = value;
      renderStats();
    });

    // Start animation loop
    animationFrame = requestAnimationFrame(animate);
  });

  onDestroy(() => {
    cancelAnimationFrame(animationFrame);
  });

  function animate() {
    animationFrame = requestAnimationFrame(animate);
  }

  function render() {
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    // Draw floor
    ctx.fillStyle = COLORS.floor;
    ctx.fillRect(50, 50, CANVAS_WIDTH - 100, 30);

    // Draw departments
    departments.forEach((dept) => {
      drawDepartment(dept);
    });

    // Draw agents
    agents.forEach((agent) => {
      drawAgent(agent);
    });

    // Draw mail animation
    if (animatingMail) {
      drawMail(animatingMail);
    }

    // Draw stats
    renderStats();
  }

  function drawDepartment(dept: DepartmentPosition) {
    const isHovered = selectedAgent === dept.x.toString();
    const color = isHovered ? COLORS.deskHover : COLORS.desk;
    const deptName = dept.x.toString().split('-')[0];
    const name = deptName.charAt(0).toUpperCase() + deptName.slice(1);

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

  function drawAgent(agent: AgentState) {
    const { x, y } = agent.position;
    const isSelected = selectedAgent === agent.id;
    const color = AGENT_COLORS[agent.department] || AGENT_COLORS.coordination;
    const statusColor = COLORS.agent[agent.status];

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
      ctx.fillText(`${agent.subAgents.length}`, x + AGENT_SIZE / 2 + 10, y + AGENT_SIZE / 2 + 8);
    }
  }

  function drawMail(mail: MailMessage) {
    const fromDept = departments.get(mail.fromDept);
    const toDept = departments.get(mail.toDept);

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
    ctx.fillStyle = mail.type === 'dispatch' ? '#3b82f6' : mail.type === 'result' ? '#10b981a' : '#f59e0';
    ctx.beginPath();
    ctx.arc(currentX, currentY, 5, 0, 2 * Math.PI);
    ctx.fill();
  }

  function renderStats() {
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 16px Arial';
    ctx.fillText('Trading Floor', 20, 20);

    ctx.font = '12px Arial';
    ctx.fillText(`Active: ${floorStats.activeTasks} | Total: ${floorStats.totalTasks}`, 20, CANVAS_HEIGHT - 20);
  }

  function handleClick(event: MouseEvent) {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Check if clicked on department
    let clickedDept: string | null = null;
    departments.forEach((dept) => {
      if (
        x >= dept.x && x <= dept.x + dept.width &&
        y >= dept.y && y <= dept.y + dept.height
      ) {
        clickedDept = dept.x.toString();
      }
    });

    // Check if clicked on agent
    let clickedAgent: AgentState | null = null;
    agents.forEach((agent) => {
      const dx = x - agent.position.x;
      const dy = y - agent.position.y;
      if (dx >= 0 && dx <= AGENT_SIZE && dy >= 0 && dy <= AGENT_SIZE) {
        clickedAgent = agent;
      }
    });

    if (clickedDept) {
    tradingFloorStore.selectAgent(clickedDept);
    } else if (clickedAgent) {
    tradingFloorStore.selectAgent(clickedAgent.id);
    }
  }
</script>

<style>
  .trading-floor-canvas {
    width: 100%;
    height: 600px;
    background: #1a1a2a;
    border-radius: 8px;
    cursor: pointer;
  }
</style>

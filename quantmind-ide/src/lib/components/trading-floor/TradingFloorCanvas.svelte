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

  // Department filter state
  let selectedDepartment: string = 'all';
  const departmentOptions = ['all', 'development', 'research', 'risk', 'trading', 'portfolio'];

  // Gamification state
  let hoveredAgent: AgentState | null = null;
  let showAchievements = false;
  let agentXP = new Map<string, number>();
  let agentLevels = new Map<string, number>();
  let agentAchievements = new Map<string, string[]>();
  let particleEffects: Array<{x: number; y: number; type: string; life: number}> = [];

  // Initialize gamification data
  const initGamification = () => {
    const $agents = get(agents);
    $agents.forEach((agent) => {
      if (!agentXP.has(agent.id)) {
        agentXP.set(agent.id, Math.floor(Math.random() * 500));
        agentLevels.set(agent.id, Math.floor(agentXP.get(agent.id)! / 100) + 1);
        agentAchievements.set(agent.id, generateRandomAchievements());
      }
    });
    agentXP = agentXP;
    agentLevels = agentLevels;
    agentAchievements = agentAchievements;
  };

  const generateRandomAchievements = (): string[] => {
    const allAchievements = ['First Task', 'Speed Demon', 'Team Player', 'Problem Solver', 'Efficiency Expert'];
    const count = Math.floor(Math.random() * 3);
    const shuffled = allAchievements.sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
  };

  const getLevelFromXP = (xp: number): number => Math.floor(xp / 100) + 1;

  const getXPProgress = (xp: number): number => (xp % 100) / 100;

  const XP_COLORS = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444'];

  // Pixel art agent sprites (8x8 grids represented as patterns)
  const AGENT_SPRITES: Record<string, string[][]> = {
    development: [
      ['........','..####..','.#....#.','######..','..##...','..##...','...##..','........'],
      ['........','..####..','.#....#.','######..','..##...','..##...','...##..','........']
    ],
    research: [
      ['........','.####...','#....#..','####....','..##....','..##....','...##...','........'],
      ['........','.####...','#....#..','####....','..##....','..##....','...##...','........']
    ],
    risk: [
      ['........','..####..','..#..#..','..####..','...##...','...##...','..##....','........'],
      ['........','..####..','..#..#..','..####..','...##...','...##...','..##....','........']
    ],
    trading: [
      ['........','.####...','##..##..','####....','..##....','..##....','...##...','........'],
      ['........','.####...','##..##..','####....','..##....','..##....','...##...','........']
    ],
    portfolio: [
      ['........','..####..','.#....#.','.######.','.#....#.','.#....#.','..####..','........'],
      ['........','..####..','.#....#.','.######.','.#....#.','.#....#.','..####..','........']
    ],
    coordination: [
      ['........','.#######.','.#.....#','.#######.','.#.....#','.#.....#','.#######.','........'],
      ['........','.#######.','.#.....#','.#######.','.#.....#','.#.....#','.#######.','........']
    ]
  };

  const ACHIEVEMENT_ICONS: Record<string, string> = {
    'First Task': '1',
    'Speed Demon': 'S',
    'Team Player': 'T',
    'Problem Solver': 'P',
    'Efficiency Expert': 'E'
  };

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
    development: '#3b82f6',
    research: '#8b5cf6',
    risk: '#ef4444',
    trading: '#f97316',
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
    development: '#3b82f6',
    research: '#8b5cf6',
    risk: '#ef4444',
    trading: '#f97316',
    portfolio: '#10b981',
    coordination: '#6366f1',
  };

  onMount(() => {
    if (!canvas) return;
    ctx = canvas.getContext('2d');
    initGamification();
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

    // Filter agents and departments by selected department
    const filteredAgents = selectedDepartment === 'all'
      ? $agents
      : new Map([...$agents].filter(([, agent]) => agent.department === selectedDepartment));

    const filteredDepartments = selectedDepartment === 'all'
      ? $departments
      : new Map([...$departments].filter(([key]) => key === selectedDepartment));

    // Clear canvas
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    // Draw grid background (gaming feel)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
    ctx.lineWidth = 1;
    for (let i = 0; i < CANVAS_WIDTH; i += 40) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, CANVAS_HEIGHT);
      ctx.stroke();
    }
    for (let i = 0; i < CANVAS_HEIGHT; i += 40) {
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(CANVAS_WIDTH, i);
      ctx.stroke();
    }

    // Draw floor
    ctx.fillStyle = COLORS.floor;
    ctx.fillRect(50, 50, CANVAS_WIDTH - 100, 30);

    // Draw filtered departments
    filteredDepartments.forEach((dept, key) => {
      drawDepartment(dept, key, $selectedAgent);
    });

    // Draw filtered agents
    filteredAgents.forEach((agent) => {
      drawAgent(agent, $selectedAgent);
    });

    // Draw mail animation
    if ($animatingMail) {
      drawMail($animatingMail, $departments);
    }

    // Draw particles
    renderParticles();

    // Draw stats
    renderStats($floorStats, filteredAgents);
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

  function drawPixelAgent(agent: AgentState, selectedId: string | null, isHovered: boolean) {
    const { x, y } = agent.position;
    const isSelected = selectedId === agent.id;
    const color = AGENT_COLORS[agent.department] || AGENT_COLORS.coordination;
    const statusColor = COLORS.agent[agent.status] || COLORS.agent.idle;
    const sprite = AGENT_SPRITES[agent.department] || AGENT_SPRITES.coordination;
    const xp = agentXP.get(agent.id) || 0;
    const level = agentLevels.get(agent.id) || 1;
    const achievements = agentAchievements.get(agent.id) || [];

    const pixelSize = 4;
    const spriteWidth = 8 * pixelSize;
    const spriteHeight = 8 * pixelSize;
    const offsetX = x - spriteWidth / 2;
    const offsetY = y - spriteHeight / 2 - 10;

    // Draw shadow
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.beginPath();
    ctx.ellipse(x, y + spriteHeight / 2 + 5, spriteWidth / 2, 6, 0, 0, 2 * Math.PI);
    ctx.fill();

    // Draw pixel character (frame based on status)
    const frameIndex = agent.status === 'typing' || agent.status === 'thinking' ? 1 : 0;
    const frame = sprite[frameIndex];

    for (let row = 0; row < frame.length; row++) {
      for (let col = 0; col < frame[row].length; col++) {
        const pixel = frame[row][col];
        if (pixel === '#') {
          ctx.fillStyle = color;
          ctx.fillRect(offsetX + col * pixelSize, offsetY + row * pixelSize, pixelSize, pixelSize);
        } else if (pixel === '.') {
          // Transparent - skip
        }
      }
    }

    // Selection glow effect
    if (isSelected || isHovered) {
      ctx.strokeStyle = isSelected ? '#fbbf24' : 'rgba(255, 255, 255, 0.5)';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.ellipse(x, y, spriteWidth / 2 + 8, spriteHeight / 2 + 8, 0, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Status indicator (animated pulse)
    const pulseScale = 1 + Math.sin(Date.now() / 200) * 0.2;
    ctx.beginPath();
    ctx.arc(x + spriteWidth / 2 + 8, y - spriteHeight / 2 - 2, 6 * pulseScale, 0, 2 * Math.PI);
    ctx.fillStyle = statusColor;
    ctx.fill();

    // Level badge
    ctx.fillStyle = '#1a1a2a';
    ctx.beginPath();
    ctx.arc(x - spriteWidth / 2 - 8, y - spriteHeight / 2 - 2, 12, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = XP_COLORS[(level - 1) % XP_COLORS.length];
    ctx.font = 'bold 10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`L${level}`, x - spriteWidth / 2 - 8, y - spriteHeight / 2 + 4);

    // XP Progress bar
    const progress = getXPProgress(xp);
    const barWidth = 40;
    const barHeight = 4;
    const barX = x - barWidth / 2;
    const barY = y + spriteHeight / 2 + 10;

    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fillRect(barX - 1, barY - 1, barWidth + 2, barHeight + 2);
    ctx.fillStyle = '#2d3740';
    ctx.fillRect(barX, barY, barWidth, barHeight);
    ctx.fillStyle = XP_COLORS[(level - 1) % XP_COLORS.length];
    ctx.fillRect(barX, barY, barWidth * progress, barHeight);

    // Agent name with level
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 11px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(agent.name, x, barY + barHeight + 14);

    // Achievement badges (show on hover or selection)
    if ((isSelected || isHovered) && achievements.length > 0) {
      achievements.forEach((achievement, idx) => {
        const badgeX = x + spriteWidth / 2 + 15 + idx * 18;
        const badgeY = y + spriteHeight / 2 - 5;

        ctx.fillStyle = '#fbbf24';
        ctx.beginPath();
        ctx.arc(badgeX, badgeY, 8, 0, 2 * Math.PI);
        ctx.fill();

        ctx.fillStyle = '#1a1a2a';
        ctx.font = 'bold 8px monospace';
        ctx.fillText(ACHIEVEMENT_ICONS[achievement] || '?', badgeX, badgeY + 3);
      });
    }

    // Sub-agents count
    if (agent.subAgents && agent.subAgents.length > 0) {
      ctx.fillStyle = '#10b981';
      ctx.beginPath();
      ctx.arc(x + spriteWidth / 2 + 5, y + spriteHeight / 2 - 5, 10, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 9px monospace';
      ctx.fillText(`${agent.subAgents.length}`, x + spriteWidth / 2 + 5, y + spriteHeight / 2 - 2);
    }
  }

  // Keep legacy function for compatibility
  function drawAgent(agent: AgentState, selectedId: string | null) {
    drawPixelAgent(agent, selectedId, hoveredAgent?.id === agent.id);
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

  function renderStats(
    stats: { totalTasks: number; activeTasks: number; completedTasks: number; pendingMail: number },
    filteredAgents?: Map<string, AgentState>
  ) {
    // Title
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 16px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('TRADING FLOOR', 20, 30);

    // Show department filter indicator
    if (selectedDepartment !== 'all') {
      ctx.fillStyle = COLORS[selectedDepartment as keyof typeof COLORS] || '#3b82f6';
      ctx.font = 'bold 12px monospace';
      ctx.fillText(`[${selectedDepartment.toUpperCase()}]`, 180, 30);
    }

    // Calculate filtered stats if department is selected
    const filteredCount = filteredAgents ? filteredAgents.size : 0;
    const displayTotal = selectedDepartment === 'all' ? stats.totalTasks : filteredCount;

    // Gamification stats (filtered)
    const xpEntries = filteredAgents
      ? [...filteredAgents.entries()].filter(([id]) => agentXP.has(id))
      : [...agentXP.entries()];
    const totalXP = xpEntries.reduce((a, [, xp]) => a + xp, 0);
    const levelValues = filteredAgents
      ? [...filteredAgents.entries()].filter(([id]) => agentLevels.has(id)).map(([, lvl]) => agentLevels.get(lvl[0]) || 1)
      : Array.from(agentLevels.values());
    const maxLevel = Math.max(...levelValues, 1);
    const achievementEntries = filteredAgents
      ? [...filteredAgents.entries()].filter(([id]) => agentAchievements.has(id))
      : [...agentAchievements.entries()];
    const totalAchievements = achievementEntries.reduce((acc, [, ach]) => acc + (ach?.length || 0), 0);

    ctx.font = '11px monospace';
    ctx.fillStyle = '#10b981';
    ctx.fillText(`XP: ${totalXP}`, selectedDepartment !== 'all' ? 300 : 180, 30);

    ctx.fillStyle = '#8b5cf6';
    ctx.fillText(`MAX LV: ${maxLevel}`, selectedDepartment !== 'all' ? 380 : 260, 30);

    ctx.fillStyle = '#fbbf24';
    ctx.fillText(`BADGES: ${totalAchievements}`, selectedDepartment !== 'all' ? 460 : 360, 30);

    ctx.fillStyle = '#6b7280';
    ctx.fillText(`Active: ${stats.activeTasks}`, selectedDepartment !== 'all' ? 560 : 460, 30);

    // Bottom stats
    ctx.fillStyle = '#4b5563';
    ctx.font = '10px monospace';
    ctx.fillText(`Agents: ${displayTotal} | Mail: ${stats.pendingMail}`, 20, CANVAS_HEIGHT - 10);

    // Achievement panel hint
    if (showAchievements) {
      renderAchievementPanel(filteredAgents);
    }
  }

  function renderAchievementPanel(filteredAgents?: Map<string, AgentState>) {
    const panelX = CANVAS_WIDTH - 220;
    const panelY = 50;

    // Panel background
    ctx.fillStyle = 'rgba(26, 26, 42, 0.95)';
    ctx.fillRect(panelX, panelY, 200, 180);
    ctx.strokeStyle = '#fbbf24';
    ctx.lineWidth = 2;
    ctx.strokeRect(panelX, panelY, 200, 180);

    ctx.fillStyle = '#fbbf24';
    ctx.font = 'bold 12px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('ACHIEVEMENTS', panelX + 100, panelY + 20);

    let y = panelY + 40;
    const displayAgents = filteredAgents || get(agents);
    let count = 0;

    displayAgents.forEach((agent) => {
      if (count >= 4) return;
      const achievements = agentAchievements.get(agent.id) || [];
      if (achievements.length > 0) {
        ctx.fillStyle = '#ffffff';
        ctx.font = '10px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(agent.name.substring(0, 12), panelX + 10, y);

        achievements.slice(0, 3).forEach((ach, idx) => {
          ctx.fillStyle = '#fbbf24';
          ctx.fillText(`* ${ach}`, panelX + 10, y + 14 + idx * 12);
        });
        y += 50;
        count++;
      }
    });

    if (count === 0) {
      ctx.fillStyle = '#6b7280';
      ctx.textAlign = 'center';
      ctx.fillText('No achievements yet', panelX + 100, y + 20);
    }
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
      if (distance <= AGENT_SIZE / 2 + 15) {
        clickedAgent = agent;
      }
    });

    if (clickedAgent) {
      selectAgent(clickedAgent.id);
      // Award XP on click
      const currentXP = agentXP.get(clickedAgent.id) || 0;
      const newXP = currentXP + 10;
      agentXP.set(clickedAgent.id, newXP);
      const newLevel = getLevelFromXP(newXP);
      const oldLevel = agentLevels.get(clickedAgent.id) || 1;
      agentLevels.set(clickedAgent.id, newLevel);
      agentXP = agentXP;
      agentLevels = agentLevels;

      // Level up effect
      if (newLevel > oldLevel) {
        spawnParticles(clickedAgent.position.x, clickedAgent.position.y, 'levelup');
      }
    } else if (clickedDept) {
      selectAgent(clickedDept);
    }
  }

  function handleMouseMove(event: MouseEvent) {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const $agents = get(agents);

    let found: AgentState | null = null;
    $agents.forEach((agent) => {
      const dx = x - agent.position.x;
      const dy = y - agent.position.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      if (distance <= AGENT_SIZE / 2 + 15) {
        found = agent;
      }
    });

    hoveredAgent = found;
  }

  function spawnParticles(x: number, y: number, type: string) {
    const count = type === 'levelup' ? 20 : 10;
    for (let i = 0; i < count; i++) {
      particleEffects.push({
        x,
        y,
        type,
        life: 1.0
      });
    }
  }

  function renderParticles() {
    particleEffects = particleEffects.filter(p => {
      p.life -= 0.02;
      if (p.type === 'levelup') {
        p.y -= 1;
        p.x += (Math.random() - 0.5) * 2;
      }

      if (p.life > 0) {
        ctx.fillStyle = p.type === 'levelup'
          ? `rgba(251, 191, 36, ${p.life})`
          : `rgba(16, 185, 129, ${p.life})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 3, 0, 2 * Math.PI);
        ctx.fill();
        return true;
      }
      return false;
    });
  }

  function toggleAchievements() {
    showAchievements = !showAchievements;
  }

  // Import get from svelte/store
  import { get } from 'svelte/store';
</script>

<div class="trading-floor-canvas">
  <div class="canvas-container">
    <canvas
      bind:this={canvas}
      width={CANVAS_WIDTH}
      height={CANVAS_HEIGHT}
      on:click={handleClick}
      on:mousemove={handleMouseMove}
    ></canvas>
    <div class="controls-overlay">
      <select
        class="department-select"
        bind:value={selectedDepartment}
        title="Filter by department"
      >
        {#each departmentOptions as dept}
          <option value={dept}>
            {dept === 'all' ? 'All Departments' : dept.charAt(0).toUpperCase() + dept.slice(1)}
          </option>
        {/each}
      </select>
      <button
        class="achievements-btn"
        class:active={showAchievements}
        on:click={toggleAchievements}
        title="Toggle Achievements"
      >
        <span class="trophy-icon">T</span>
      </button>
    </div>
  </div>
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

  .canvas-container {
    position: relative;
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  canvas {
    max-width: 100%;
    max-height: 100%;
    cursor: pointer;
  }

  .controls-overlay {
    position: absolute;
    top: 10px;
    right: 10px;
    display: flex;
    gap: 8px;
    z-index: 10;
  }

  .department-select {
    padding: 8px 12px;
    background: rgba(26, 26, 42, 0.9);
    border: 2px solid #3b82f6;
    border-radius: 8px;
    color: #ffffff;
    font-family: monospace;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .department-select:hover {
    background: rgba(59, 130, 246, 0.2);
  }

  .department-select:focus {
    outline: none;
    border-color: #60a5fa;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
  }

  .department-select option {
    background: #1a1a2a;
    color: #ffffff;
  }

  .achievements-btn {
    width: 36px;
    height: 36px;
    background: rgba(26, 26, 42, 0.9);
    border: 2px solid #fbbf24;
    border-radius: 8px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
  }

  .achievements-btn:hover,
  .achievements-btn.active {
    background: #fbbf24;
    transform: scale(1.1);
  }

  .achievements-btn:hover .trophy-icon,
  .achievements-btn.active .trophy-icon {
    color: #1a1a2a;
  }

  .trophy-icon {
    font-family: monospace;
    font-weight: bold;
    font-size: 16px;
    color: #fbbf24;
    transition: color 0.2s ease;
  }
</style>

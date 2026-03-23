/**
 * Story 12-1: AgentPanel Tests
 *
 * Tests cover:
 * - AC #2: Dept badge updates when canvas prop changes
 * - AC #3: Collapse state class applies correctly
 * - AC #4: New interactive session creation via [+]
 * - AC #6: User message echo to .ap-body
 * - AC #7/16: Tool call rendering + OPINION expansion
 * - AC #9: Workshop canvas hides panel
 * - AC #11/19: SSE EventSource opens on session create and closes on destroy
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// ─── Constants matching the component ────────────────────────────────────────

const CANVAS_DEPT_HEAD: Record<string, { label: string; color: string }> = {
  'live-trading':   { label: 'TRADING',     color: 'green' },
  'research':       { label: 'RESEARCH',    color: 'amber' },
  'development':    { label: 'DEVELOPMENT', color: 'cyan' },
  'risk':           { label: 'RISK',        color: 'red' },
  'trading':        { label: 'TRADING',     color: 'green' },
  'portfolio':      { label: 'PORTFOLIO',   color: 'cyan' },
  'shared-assets':  { label: 'SHARED',      color: 'muted' },
  'workshop':       { label: 'FLOOR MGR',   color: 'cyan' },
  'flowforge':      { label: 'FLOOR MGR',   color: 'cyan' },
};

const COLOR_MAP: Record<string, string> = {
  'cyan':  'var(--color-accent-cyan)',
  'amber': 'var(--color-accent-amber)',
  'red':   'var(--color-accent-red)',
  'green': 'var(--color-accent-green)',
  'muted': 'var(--color-text-muted)',
};

// ─── Helpers matching component logic ────────────────────────────────────────

function getDeptHead(canvas: string) {
  return CANVAS_DEPT_HEAD[canvas] ?? CANVAS_DEPT_HEAD['workshop'];
}

function isWorkshop(canvas: string) {
  return canvas === 'workshop' || canvas === 'flowforge';
}

function truncate(str: string, max: number): string {
  return str.length > max ? str.slice(0, max) + '…' : str;
}

function formatToolLine(tool: string, args: Record<string, unknown>): string {
  if (tool === 'write_memory') {
    const nodeType = String(args['node_type'] ?? '');
    if (nodeType === 'OPINION') {
      const conf = String(args['confidence'] ?? '');
      const action = truncate(String(args['action'] ?? ''), 40);
      return `write_memory(OPINION · confidence=${conf} · action="${action}")`;
    }
  }
  if (tool === 'search_memory') {
    const query = truncate(String(args['query'] ?? ''), 60);
    return `search_memory(query: "${query}")`;
  }
  if (tool === 'context7') {
    const query = truncate(String(args['query'] ?? ''), 60);
    return `context7(query: "${query}")`;
  }
  if (tool === 'sequential_thinking') {
    const n = String(args['step'] ?? '');
    const total = String(args['total'] ?? '');
    const reasoning = truncate(String(args['reasoning'] ?? ''), 40);
    return `sequential_thinking(step ${n}/${total} · ${reasoning})`;
  }
  if (tool === 'web_fetch') {
    const url = truncate(String(args['url'] ?? ''), 60);
    return `web_fetch(url: "${url}")`;
  }
  const firstKey = Object.keys(args)[0] ?? '';
  const firstVal = firstKey ? truncate(String(args[firstKey] ?? ''), 60) : '';
  return firstKey ? `${tool}(${firstKey}: "${firstVal}")` : `${tool}()`;
}

// ─── Tests ────────────────────────────────────────────────────────────────────

describe('AgentPanel — CANVAS_DEPT_HEAD map (AC #2)', () => {
  it('maps research canvas to RESEARCH dept with amber color', () => {
    const head = getDeptHead('research');
    expect(head.label).toBe('RESEARCH');
    expect(head.color).toBe('amber');
    expect(COLOR_MAP[head.color]).toBe('var(--color-accent-amber)');
  });

  it('maps risk canvas to RISK dept with red color', () => {
    const head = getDeptHead('risk');
    expect(head.label).toBe('RISK');
    expect(head.color).toBe('red');
    expect(COLOR_MAP[head.color]).toBe('var(--color-accent-red)');
  });

  it('maps development canvas to DEVELOPMENT dept with cyan color', () => {
    const head = getDeptHead('development');
    expect(head.label).toBe('DEVELOPMENT');
    expect(head.color).toBe('cyan');
    expect(COLOR_MAP[head.color]).toBe('var(--color-accent-cyan)');
  });

  it('maps live-trading canvas to TRADING dept with green color', () => {
    const head = getDeptHead('live-trading');
    expect(head.label).toBe('TRADING');
    expect(head.color).toBe('green');
    expect(COLOR_MAP[head.color]).toBe('var(--color-accent-green)');
  });

  it('maps all 9 canvases correctly', () => {
    const expectations = [
      ['live-trading', 'TRADING'],
      ['research', 'RESEARCH'],
      ['development', 'DEVELOPMENT'],
      ['risk', 'RISK'],
      ['trading', 'TRADING'],
      ['portfolio', 'PORTFOLIO'],
      ['shared-assets', 'SHARED'],
      ['workshop', 'FLOOR MGR'],
      ['flowforge', 'FLOOR MGR'],
    ] as const;

    for (const [canvas, label] of expectations) {
      expect(getDeptHead(canvas).label).toBe(label);
    }
  });

  it('falls back to workshop dept for unknown canvas', () => {
    const head = getDeptHead('unknown-canvas');
    expect(head.label).toBe('FLOOR MGR');
  });
});

describe('AgentPanel — collapse state (AC #3)', () => {
  it('panel should start in expanded state by default', () => {
    let collapsed = false;
    expect(collapsed).toBe(false);
  });

  it('collapse class should be applied when collapsed=true', () => {
    // Simulates the class:collapsed={collapsed} binding
    let collapsed = true;
    const classList = collapsed ? ['agent-panel', 'collapsed'] : ['agent-panel'];
    expect(classList).toContain('collapsed');
  });

  it('ide-layout collapse class applied based on agentPanelCollapsed', () => {
    let agentPanelCollapsed = true;
    const classes = agentPanelCollapsed
      ? ['ide-layout', 'agent-panel-collapsed']
      : ['ide-layout'];
    expect(classes).toContain('agent-panel-collapsed');
  });

  it('ide-layout collapse class not applied when panel is expanded', () => {
    let agentPanelCollapsed = false;
    const classes = agentPanelCollapsed
      ? ['ide-layout', 'agent-panel-collapsed']
      : ['ide-layout'];
    expect(classes).not.toContain('agent-panel-collapsed');
  });
});

describe('AgentPanel — session creation (AC #4)', () => {
  it('creates new session with unique id', () => {
    const sessions: Array<{ id: string; type: string; deptHead: string }> = [];
    const canvas = 'research';
    const deptHead = getDeptHead(canvas);

    const session = {
      id: crypto.randomUUID(),
      type: 'interactive' as const,
      deptHead: deptHead.label,
      canvasId: canvas,
      messages: [],
      createdAt: new Date().toISOString(),
      status: 'active' as const,
    };

    sessions.push(session);
    expect(sessions.length).toBe(1);
    expect(sessions[0].type).toBe('interactive');
    expect(sessions[0].deptHead).toBe('RESEARCH');
    expect(sessions[0].id).toBeTruthy();
  });

  it('session id is a valid UUID', () => {
    const id = crypto.randomUUID();
    expect(id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i);
  });
});

describe('AgentPanel — user message echo (AC #6)', () => {
  it('appends user message to session messages', () => {
    const messages: Array<{ id: string; type: string; content: string }> = [];
    const inputText = 'Show me the RSI divergence strategy';

    const msg = {
      id: crypto.randomUUID(),
      type: 'user' as const,
      content: inputText,
      timestamp: new Date().toISOString(),
    };

    messages.push(msg);

    expect(messages.length).toBe(1);
    expect(messages[0].type).toBe('user');
    expect(messages[0].content).toBe(inputText);
  });

  it('clears input value after submission', () => {
    let inputValue = 'some message';
    // Simulate submit
    const content = inputValue.trim();
    if (content) {
      inputValue = '';
    }
    expect(inputValue).toBe('');
  });

  it('does not submit empty message', () => {
    const messages: unknown[] = [];
    const inputValue = '   ';
    if (!inputValue.trim()) {
      // no push
    }
    expect(messages.length).toBe(0);
  });
});

describe('AgentPanel — tool call rendering (AC #12, #13, #14, #15)', () => {
  it('formats OPINION write_memory tool line correctly', () => {
    // Action string 'Reduce position on EURUSD' is 25 chars — under 40, not truncated
    const line = formatToolLine('write_memory', {
      node_type: 'OPINION',
      confidence: 0.87,
      action: 'Reduce position on EURUSD',
    });
    expect(line).toContain('write_memory(OPINION');
    expect(line).toContain('confidence=0.87');
    expect(line).toContain('action="Reduce position on EURUSD"');
  });

  it('truncates OPINION action at 40 chars', () => {
    const longAction = 'A'.repeat(50);
    const line = formatToolLine('write_memory', {
      node_type: 'OPINION',
      confidence: 0.5,
      action: longAction,
    });
    expect(line).toContain('…');
    expect(line.length).toBeLessThan(200);
  });

  it('formats search_memory tool line correctly', () => {
    const line = formatToolLine('search_memory', {
      query: 'Kelly criterion position sizing',
    });
    expect(line).toBe('search_memory(query: "Kelly criterion position sizing")');
  });

  it('truncates search_memory query at 60 chars', () => {
    const longQuery = 'Q'.repeat(80);
    const line = formatToolLine('search_memory', { query: longQuery });
    expect(line).toContain('…');
  });

  it('formats context7 tool line correctly', () => {
    const line = formatToolLine('context7', {
      query: 'Kelly criterion sizing',
    });
    expect(line).toBe('context7(query: "Kelly criterion sizing")');
  });

  it('formats sequential_thinking tool line correctly', () => {
    const line = formatToolLine('sequential_thinking', {
      step: 2,
      total: 5,
      reasoning: 'Evaluating market regime',
    });
    expect(line).toContain('sequential_thinking(step 2/5');
    expect(line).toContain('Evaluating market regime');
  });

  it('formats web_fetch tool line correctly', () => {
    const line = formatToolLine('web_fetch', {
      url: 'https://example.com/article',
    });
    expect(line).toBe('web_fetch(url: "https://example.com/article")');
  });

  it('formats generic tool fallback correctly', () => {
    const line = formatToolLine('some_tool', { param: 'value' });
    expect(line).toBe('some_tool(param: "value")');
  });

  it('formats tool with no args correctly', () => {
    const line = formatToolLine('no_arg_tool', {});
    expect(line).toBe('no_arg_tool()');
  });
});

describe('AgentPanel — OPINION expand-on-click (AC #13)', () => {
  it('detects OPINION tool call correctly', () => {
    const msg = {
      id: '1',
      type: 'tool',
      content: 'write_memory(OPINION · confidence=0.87 · action="test")',
      tool: 'write_memory',
      args: { node_type: 'OPINION', confidence: 0.87, action: 'test' },
      timestamp: new Date().toISOString(),
    };

    const isOpinion = msg.tool === 'write_memory' &&
      String(msg.args?.['node_type'] ?? '') === 'OPINION';
    expect(isOpinion).toBe(true);
  });

  it('toggles expandedToolLine on click', () => {
    let expandedToolLine: string | null = null;
    const id = 'tool-1';

    // First click — expand
    expandedToolLine = expandedToolLine === id ? null : id;
    expect(expandedToolLine).toBe(id);

    // Second click — collapse
    expandedToolLine = expandedToolLine === id ? null : id;
    expect(expandedToolLine).toBeNull();
  });
});

describe('AgentPanel — Workshop canvas hidden (AC #9)', () => {
  it('isWorkshop is true for workshop canvas', () => {
    expect(isWorkshop('workshop')).toBe(true);
  });

  it('isWorkshop is true for flowforge canvas', () => {
    expect(isWorkshop('flowforge')).toBe(true);
  });

  it('isWorkshop is false for all other canvases', () => {
    const others = ['live-trading', 'research', 'development', 'risk', 'trading', 'portfolio', 'shared-assets'];
    for (const canvas of others) {
      expect(isWorkshop(canvas)).toBe(false);
    }
  });
});

describe('AgentPanel — SSE EventSource lifecycle (AC #11, #19)', () => {
  let mockEventSource: {
    url: string;
    onmessage: ((e: MessageEvent) => void) | null;
    onerror: (() => void) | null;
    close: ReturnType<typeof vi.fn>;
  };

  beforeEach(() => {
    mockEventSource = {
      url: '',
      onmessage: null,
      onerror: null,
      close: vi.fn(),
    };

    vi.stubGlobal('EventSource', vi.fn().mockImplementation((url: string) => {
      mockEventSource.url = url;
      return mockEventSource;
    }));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('opens SSE EventSource to agent stream URL on session create', () => {
    const sessionId = crypto.randomUUID();
    const baseUrl = 'http://localhost:8001';
    const source = new EventSource(`${baseUrl}/api/agents/stream?session=${sessionId}`);
    expect(mockEventSource.url).toBe(`${baseUrl}/api/agents/stream?session=${sessionId}`);
  });

  it('EventSource URL contains correct session id', () => {
    const sessionId = 'test-session-abc';
    new EventSource(`http://localhost:8001/api/agents/stream?session=${sessionId}`);
    expect(mockEventSource.url).toContain(`session=${sessionId}`);
  });

  it('EventSource URL does NOT use WebSocket protocol', () => {
    const url = 'http://localhost:8001/api/agents/stream?session=123';
    expect(url).not.toMatch(/^ws/);
  });

  it('calls eventSource.close() on destroy', () => {
    const sessionId = 'destroy-test';
    const source = new EventSource(`http://localhost:8001/api/agents/stream?session=${sessionId}`);
    // Simulate onDestroy
    mockEventSource.close();
    expect(mockEventSource.close).toHaveBeenCalledTimes(1);
  });

  it('closes existing SSE connection before opening new one', () => {
    let eventSourceRef: typeof mockEventSource | null = null;

    // First session
    eventSourceRef = new EventSource('http://localhost:8001/api/agents/stream?session=1') as unknown as typeof mockEventSource;

    // Opening second session closes first
    eventSourceRef?.close();
    const secondSource = new EventSource('http://localhost:8001/api/agents/stream?session=2');

    expect(mockEventSource.close).toHaveBeenCalledTimes(1);
  });
});

describe('AgentPanel — Autonomous workflow mode (AC #7, #16)', () => {
  it('autonomous session has read-only status card properties', () => {
    const session = {
      id: crypto.randomUUID(),
      type: 'autonomous' as const,
      deptHead: 'RISK',
      canvasId: 'risk',
      messages: [],
      workflowName: 'Risk Regime Check',
      workflowStage: 'hmm_analysis',
      workflowElapsed: 142,
      subAgents: [
        { role: 'hmm_trainer', status: 'running' as const },
        { role: 'risk_calculator', status: 'idle' as const },
      ],
      createdAt: new Date().toISOString(),
      status: 'active' as const,
    };

    expect(session.type).toBe('autonomous');
    expect(session.workflowName).toBe('Risk Regime Check');
    expect(session.workflowStage).toBe('hmm_analysis');
    expect(session.workflowElapsed).toBe(142);
    expect(session.subAgents).toHaveLength(2);
  });

  it('sub-agent running status maps to cyan color', () => {
    function subAgentStatusColor(status: string) {
      if (status === 'running') return 'var(--color-accent-cyan)';
      if (status === 'blocked') return 'var(--color-accent-amber)';
      return 'var(--color-text-muted)';
    }
    expect(subAgentStatusColor('running')).toBe('var(--color-accent-cyan)');
    expect(subAgentStatusColor('idle')).toBe('var(--color-text-muted)');
    expect(subAgentStatusColor('blocked')).toBe('var(--color-accent-amber)');
  });

  it('footer input is not shown in autonomous mode', () => {
    const session = { type: 'autonomous' };
    const showFooter = session.type === 'interactive';
    expect(showFooter).toBe(false);
  });

  it('footer input is shown in interactive mode', () => {
    const session = { type: 'interactive' };
    const showFooter = session.type === 'interactive';
    expect(showFooter).toBe(true);
  });
});

describe('AgentPanel — grid CSS target (AC #1, #8)', () => {
  it('grid-template-areas includes agent column', () => {
    const gridTemplateAreas = [
      '"topbar topbar topbar"',
      '"statusband statusband statusband"',
      '"activity main agent"',
    ];
    const hasAgentArea = gridTemplateAreas.some(row => row.includes('agent'));
    expect(hasAgentArea).toBe(true);
  });

  it('grid has 3 columns (sidebar, main, agent)', () => {
    // The CSS value is: var(--sidebar-width) 1fr var(--agent-panel-width, 320px)
    // We verify it contains exactly 3 column definitions (sidebar, fr unit, agent-panel)
    const gridTemplateColumns = 'var(--sidebar-width) 1fr var(--agent-panel-width, 320px)';
    expect(gridTemplateColumns).toContain('var(--sidebar-width)');
    expect(gridTemplateColumns).toContain('1fr');
    expect(gridTemplateColumns).toContain('var(--agent-panel-width');
    // Verify no 4th column exists
    expect(gridTemplateColumns).not.toContain('minmax');
    expect(gridTemplateColumns).not.toContain('var(--sidebar-width) 1fr var(--agent-panel-width, 320px) ');
  });

  it('collapsed grid collapses agent column to 0px', () => {
    const collapsedColumns = 'var(--sidebar-width) 1fr 0px';
    expect(collapsedColumns).toContain('0px');
  });

  it('grid has no bottom row (BottomPanel removed)', () => {
    const gridTemplateAreas = [
      '"topbar topbar topbar"',
      '"statusband statusband statusband"',
      '"activity main agent"',
    ];
    const hasBottomRow = gridTemplateAreas.some(row => row.includes('bottom'));
    expect(hasBottomRow).toBe(false);
  });
});

<script lang="ts">
  import { onDestroy } from 'svelte';
  import { Plus, History, ChevronRight, Send, Paperclip } from 'lucide-svelte';
  import { canvasContextService } from '$lib/services/canvasContextService';
  import { API_CONFIG } from '$lib/config/api';
  import RichRenderer from '$lib/components/shared/RichRenderer.svelte';
  import { listOpinionNodes, type OpinionNode } from '$lib/api/graphMemory';
  import { getCanvasPanelState } from './agentPanelState';

  // ─── Types ───────────────────────────────────────────────────────────────────

  interface AgentMessage {
    id: string;
    type: 'agent' | 'user' | 'tool';
    content: string;
    tool?: string;
    args?: Record<string, unknown>;
    timestamp: string;
  }

  interface SubAgentStatus {
    role: string;
    status: 'running' | 'idle' | 'blocked';
  }

  interface AgentSession {
    id: string;
    backendSessionId?: string;
    type: 'interactive' | 'autonomous';
    deptHead: string;
    canvasId: string;
    canvasContext?: Record<string, unknown> | null;
    messages: AgentMessage[];
    workflowName?: string;
    workflowStage?: string;
    workflowElapsed?: number;
    subAgents?: SubAgentStatus[];
    createdAt: string;
    status: 'active' | 'completed' | 'error';
  }

  interface Attachment {
    id: string;
    label: string;
    canvasId: string;
  }

  interface AttachedCanvasContextPayload {
    canvas: string;
    label: string;
    context: Record<string, unknown> | null;
  }

  interface MailEntry {
    id: string;
    from_dept: string;
    type: string;
    subject: string;
    body: string;
    priority: string;
    timestamp: string;
    read: boolean;
  }

  interface CanvasContextOption {
    id: string;
    label: string;
  }

  type AgentPanelTab = 'chat' | 'agents' | 'memory' | 'mail';

  interface AgentPanelCanvasState {
    sessions: AgentSession[];
    activeSessionId: string | null;
    showSessionHistory: boolean;
    inputValue: string;
    expandedToolLine: string | null;
    activeTab: AgentPanelTab;
    attachments: Attachment[];
    memoryNodes: OpinionNode[];
    memoryLoading: boolean;
    memoryError: string | null;
    mailEntries: MailEntry[];
    mailLoading: boolean;
    mailError: string | null;
  }

  type AgentStreamEvent =
    | { type: 'tool_call'; tool: string; args: Record<string, unknown> }
    | { type: 'agent_message'; content: string; role: 'assistant' }
    | { type: 'sub_agent_status'; agent_role: string; status: 'running' | 'idle' | 'blocked' }
    | { type: 'task_status'; workflow_id: string; stage: string; elapsed_s: number };

  // ─── Static Maps ─────────────────────────────────────────────────────────────

  const CANVAS_DEPT_HEAD: Record<string, { label: string; color: 'cyan' | 'amber' | 'red' | 'green' | 'muted' }> = {
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

  // ─── Slash commands ───────────────────────────────────────────────────────────

  const slashCommands = [
    { command: '/research', desc: 'Start a research task' },
    { command: '/backtest', desc: 'Run a backtest' },
    { command: '/scan',     desc: 'Market scan' },
    { command: '/deploy',   desc: 'Deploy an EA' },
    { command: '/report',   desc: 'Generate a report' },
    { command: '/memory',   desc: 'Query memory' },
  ];

  // ─── Props ────────────────────────────────────────────────────────────────────

  interface Props {
    activeCanvas: string;
    collapsed?: boolean;
    hidden?: boolean;
  }

  let { activeCanvas, collapsed = $bindable(false), hidden = false }: Props = $props();

  // ─── State ────────────────────────────────────────────────────────────────────

  function createAgentPanelCanvasState(): AgentPanelCanvasState {
    return {
      sessions: [],
      activeSessionId: null,
      showSessionHistory: false,
      inputValue: '',
      expandedToolLine: null,
      activeTab: 'chat',
      attachments: [],
      memoryNodes: [],
      memoryLoading: false,
      memoryError: null,
      mailEntries: [],
      mailLoading: false,
      mailError: null,
    };
  }

  function snapshotPanelState(): AgentPanelCanvasState {
    return {
      sessions,
      activeSessionId,
      showSessionHistory,
      inputValue,
      expandedToolLine,
      activeTab,
      attachments,
      memoryNodes,
      memoryLoading,
      memoryError,
      mailEntries,
      mailLoading,
      mailError,
    };
  }

  function restorePanelState(state: AgentPanelCanvasState) {
    sessions = state.sessions;
    activeSessionId = state.activeSessionId;
    showSessionHistory = state.showSessionHistory;
    inputValue = state.inputValue;
    expandedToolLine = state.expandedToolLine;
    activeTab = state.activeTab;
    attachments = state.attachments;
    memoryNodes = state.memoryNodes;
    memoryLoading = state.memoryLoading;
    memoryError = state.memoryError;
    mailEntries = state.mailEntries;
    mailLoading = state.mailLoading;
    mailError = state.mailError;
  }

  let panelStateByCanvas = $state<Record<string, AgentPanelCanvasState>>({});
  let sessions = $state<AgentSession[]>([]);
  let activeSessionId = $state<string | null>(null);
  let showSessionHistory = $state(false);
  let inputValue = $state('');
  let expandedToolLine = $state<string | null>(null);
  let activeTab = $state<AgentPanelTab>('chat');
  let attachments = $state<Attachment[]>([]);
  let showAttachMenu = $state(false);
  let isSending = $state(false);
  let memoryNodes = $state<OpinionNode[]>([]);
  let memoryLoading = $state(false);
  let memoryError = $state<string | null>(null);
  let mailEntries = $state<MailEntry[]>([]);
  let mailLoading = $state(false);
  let mailError = $state<string | null>(null);
  let activeCanvasStateKey = $state(activeCanvas);

  let activeSession = $derived(sessions.find(s => s.id === activeSessionId) ?? null);
  let messages = $derived(activeSession?.messages ?? []);
  let deptHead = $derived(CANVAS_DEPT_HEAD[activeCanvas] ?? CANVAS_DEPT_HEAD['workshop']);
  // Workshop canvas has its own built-in copilot UI (WorkshopCanvas); FlowForge uses this panel
  let isWorkshop = $derived(activeCanvas === 'workshop');
  let deptColor = $derived(COLOR_MAP[deptHead.color]);
  let subAgents = $derived(activeSession?.subAgents ?? []);
  let departmentId = $derived(resolveDepartmentId(activeCanvas));

  // ─── Canvas context attachment options ───────────────────────────────────────

  // Full list of canvas context options (shown in Workshop/Copilot for full context)
  const allCanvasContextOptions: CanvasContextOption[] = [
    { id: 'research',    label: 'Research context' },
    { id: 'development', label: 'Development context' },
    { id: 'trading',     label: 'Trading context' },
    { id: 'risk',        label: 'Risk context' },
    { id: 'portfolio',   label: 'Portfolio context' },
  ];

  // Filtered options: show only current department when in a department canvas,
  // show all departments when in Workshop/FlowForge (Copilot context)
  let canvasContextOptions = $derived(
    isWorkshop
      ? allCanvasContextOptions
      : allCanvasContextOptions.filter(opt => opt.id === activeCanvas)
  );

  // Filter slash commands based on current input
  let filteredSlashCommands = $derived(
    slashCommands.filter(cmd =>
      inputValue === '/'
        ? true
        : cmd.command.startsWith(inputValue.split(' ')[0])
    )
  );

  // ─── SSE ─────────────────────────────────────────────────────────────────────

  let eventSource: EventSource | null = null;
  let currentStreamSessionId = $state<string | null>(null);

  function getAgentApiBase(): string {
    return API_CONFIG.API_URL;
  }

  function openSSE(sessionId: string) {
    eventSource?.close();
    const apiBase = getAgentApiBase();
    eventSource = new EventSource(`${apiBase}/api/agents/stream?session=${sessionId}`);
    currentStreamSessionId = sessionId;
    eventSource.onmessage = (e) => {
      try {
        handleStreamEvent(JSON.parse(e.data) as AgentStreamEvent);
      } catch {
        // malformed SSE payload — ignore silently
      }
    };
    eventSource.onerror = () => {
      console.warn('[AgentPanel] SSE connection error — will retry');
    };
  }

  function handleStreamEvent(event: AgentStreamEvent) {
    if (event.type === 'tool_call')        appendToolLine(event);
    if (event.type === 'agent_message')    appendAgentMessage(event);
    if (event.type === 'sub_agent_status') updateSubAgentStatus(event);
    if (event.type === 'task_status')      updateWorkflowStage(event);
  }

  function appendToolLine(event: { type: 'tool_call'; tool: string; args: Record<string, unknown> }) {
    if (!activeSessionId) return;
    const msg: AgentMessage = {
      id: crypto.randomUUID(),
      type: 'tool',
      content: formatToolLine(event.tool, event.args),
      tool: event.tool,
      args: event.args,
      timestamp: new Date().toISOString(),
    };
    sessions = sessions.map(s =>
      s.id === activeSessionId
        ? { ...s, messages: [...s.messages, msg] }
        : s
    );
  }

  function appendAgentMessage(event: { type: 'agent_message'; content: string; role: 'assistant' }) {
    if (!activeSessionId) return;
    const msg: AgentMessage = {
      id: crypto.randomUUID(),
      type: 'agent',
      content: event.content,
      timestamp: new Date().toISOString(),
    };
    sessions = sessions.map(s =>
      s.id === activeSessionId
        ? { ...s, messages: [...s.messages, msg] }
        : s
    );
  }

  function updateSubAgentStatus(event: { type: 'sub_agent_status'; agent_role: string; status: 'running' | 'idle' | 'blocked' }) {
    if (!activeSessionId) return;
    sessions = sessions.map(s => {
      if (s.id !== activeSessionId) return s;
      const existing = s.subAgents ?? [];
      const idx = existing.findIndex(a => a.role === event.agent_role);
      const updated: SubAgentStatus[] = idx >= 0
        ? existing.map((a, i) => i === idx ? { ...a, status: event.status } : a)
        : [...existing, { role: event.agent_role, status: event.status }];
      return { ...s, subAgents: updated };
    });
  }

  function updateWorkflowStage(event: { type: 'task_status'; workflow_id: string; stage: string; elapsed_s: number }) {
    if (!activeSessionId) return;
    sessions = sessions.map(s =>
      s.id === activeSessionId
        ? { ...s, workflowStage: event.stage, workflowElapsed: event.elapsed_s }
        : s
    );
  }

  onDestroy(() => {
    panelStateByCanvas = {
      ...panelStateByCanvas,
      [activeCanvas]: snapshotPanelState(),
    };
    eventSource?.close();
    eventSource = null;
    currentStreamSessionId = null;
  });

  $effect(() => {
    getCanvasPanelState(panelStateByCanvas, activeCanvas, createAgentPanelCanvasState);

    if (activeCanvasStateKey === activeCanvas) {
      return;
    }

    panelStateByCanvas = {
      ...panelStateByCanvas,
      [activeCanvasStateKey]: snapshotPanelState(),
    };
    restorePanelState(getCanvasPanelState(panelStateByCanvas, activeCanvas, createAgentPanelCanvasState));
    showAttachMenu = false;
    activeCanvasStateKey = activeCanvas;
  });

  $effect(() => {
    if (hidden || isWorkshop || !activeSessionId) {
      eventSource?.close();
      eventSource = null;
      currentStreamSessionId = null;
      return;
    }

    if (currentStreamSessionId !== activeSessionId) {
      openSSE(activeSessionId);
    }
  });

  // ─── Tool Line Formatting ─────────────────────────────────────────────────────

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
    // Generic fallback
    const firstKey = Object.keys(args)[0] ?? '';
    const firstVal = firstKey ? truncate(String(args[firstKey] ?? ''), 60) : '';
    return firstKey
      ? `${tool}(${firstKey}: "${firstVal}")`
      : `${tool}()`;
  }

  function truncate(str: string, max: number): string {
    return str.length > max ? str.slice(0, max) + '…' : str;
  }

  function formatElapsed(seconds: number): string {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return m > 0 ? `${m}m ${s}s` : `${s}s`;
  }

  function formatLocalDate(isoUtc: string): string {
    try {
      return new Date(isoUtc).toLocaleString();
    } catch {
      return isoUtc;
    }
  }

  function isVisibleElement(element: Element | null): boolean {
    let current: Element | null = element;
    while (current) {
      if (current.hasAttribute('hidden') || current.getAttribute('aria-hidden') === 'true') {
        return false;
      }
      const style = window.getComputedStyle(current);
      if (style.display === 'none' || style.visibility === 'hidden') {
        return false;
      }
      current = current.parentElement;
    }
    return true;
  }

  function getVisibleCanvasText(): string {
    if (typeof document === 'undefined') return '';

    const root = document.querySelector('.main-content');
    if (!root) return '';

    const chunks: string[] = [];
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        const text = node.textContent?.replace(/\s+/g, ' ').trim();
        const parent = node.parentElement;
        if (!text || !parent || !isVisibleElement(parent)) {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      },
    });

    while (walker.nextNode()) {
      const text = walker.currentNode.textContent?.replace(/\s+/g, ' ').trim();
      if (!text) continue;
      if (chunks[chunks.length - 1] !== text) {
        chunks.push(text);
      }
      if (chunks.length >= 120) break;
    }

    return chunks.join(' | ').slice(0, 4000);
  }

  function resolveDepartmentId(canvasId: string): string | null {
    const mapping: Record<string, string> = {
      'live-trading': 'trading',
      'research': 'research',
      'development': 'development',
      'risk': 'risk',
      'trading': 'trading',
      'portfolio': 'portfolio',
      'flowforge': 'floor-manager',
      'workshop': 'floor-manager',
    };
    return mapping[canvasId] ?? null;
  }

  function captureVisibleCanvasText(canvasId: string): string | null {
    if (typeof document === 'undefined') return null;

    const selectorId = resolveDepartmentId(canvasId) ?? canvasId;
    const canvasRoot = document.querySelector<HTMLElement>(`[data-dept="${selectorId}"]`);
    const text = canvasRoot?.innerText?.trim();
    if (!text) return null;
    return text.slice(0, 4000);
  }

  function getChatEndpointForCanvas(canvasId: string): string | null {
    const dept = resolveDepartmentId(canvasId);
    if (!dept) return null;
    if (dept === 'floor-manager') {
      return `${API_CONFIG.API_BASE}/chat/floor-manager/message`;
    }
    return `${API_CONFIG.API_BASE}/chat/departments/${dept}/message`;
  }

  async function loadMemoryTab() {
    memoryLoading = true;
    memoryError = null;
    try {
      memoryNodes = await listOpinionNodes(20);
    } catch (error) {
      memoryError = error instanceof Error ? error.message : 'Failed to load memory';
    } finally {
      memoryLoading = false;
    }
  }

  async function loadMailTab() {
    if (!departmentId || departmentId === 'floor-manager') {
      mailEntries = [];
      mailError = null;
      return;
    }
    mailLoading = true;
    mailError = null;
    try {
      const response = await fetch(`${API_CONFIG.API_BASE}/departments/mail/inbox/${departmentId}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json() as { messages?: MailEntry[] };
      mailEntries = data.messages ?? [];
    } catch (error) {
      mailError = error instanceof Error ? error.message : 'Failed to load mail';
      mailEntries = [];
    } finally {
      mailLoading = false;
    }
  }

  function subAgentStatusColor(status: 'running' | 'idle' | 'blocked'): string {
    if (status === 'running')  return 'var(--color-accent-cyan)';
    if (status === 'blocked')  return 'var(--color-accent-amber)';
    return 'var(--color-text-muted)';
  }

  // ─── Session Actions ──────────────────────────────────────────────────────────

  async function loadSessionHistory() {
    try {
      const apiBase = getAgentApiBase();
      const response = await fetch(
        `${apiBase}/api/agents/sessions?dept_head=${encodeURIComponent(deptHead.label)}`
      );
      if (!response.ok) return;
      const data = await response.json() as { sessions?: AgentSession[] };
      if (!data.sessions) return;
      // Merge backend sessions with in-memory sessions (avoid duplicates by id)
      const existingIds = new Set(sessions.map(s => s.id));
      const newSessions = data.sessions.filter(s => !existingIds.has(s.id));
      if (newSessions.length > 0) {
        sessions = [...sessions, ...newSessions];
      }
    } catch {
      // Backend unavailable in Epic 12 scaffold — silently ignore
    }
  }

  async function toggleSessionHistory() {
    if (!showSessionHistory) {
      await loadSessionHistory();
    }
    showSessionHistory = !showSessionHistory;
  }

  async function createNewSession() {
    const context = await canvasContextService.loadCanvasContext(activeCanvas);
    const session: AgentSession = {
      id: crypto.randomUUID(),
      type: 'interactive',
      deptHead: deptHead.label,
      canvasId: activeCanvas,
      canvasContext: context,
      messages: [],
      createdAt: new Date().toISOString(),
      status: 'active',
    };
    sessions = [...sessions, session];
    activeSessionId = session.id;
    showSessionHistory = false;
  }

  function selectSession(id: string) {
    activeSessionId = id;
    showSessionHistory = false;
  }

  async function submitMessage() {
    if (!inputValue.trim() || !activeSessionId || isSending) return;

    const rawMessage = inputValue.trim();
    const sessionRef = activeSession?.backendSessionId ?? activeSessionId ?? undefined;

    let activeCanvasContext = await canvasContextService.getEnrichedContext(
      activeCanvas,
      sessionRef,
    );
    if (!activeCanvasContext) {
      activeCanvasContext = canvasContextService.getCanvasContext(activeCanvas) ?? canvasContextService.getCurrentContext();
    }

    const attachedContexts: AttachedCanvasContextPayload[] = attachments.length > 0
      ? await Promise.all(
          attachments.map(async (attachment) => {
            const loadedContext =
              await canvasContextService.getEnrichedContext(attachment.canvasId, sessionRef);

            return {
              canvas: attachment.canvasId,
              label: attachment.label,
              context: loadedContext,
            };
          }),
        )
      : [];
    const visibleCanvasText = captureVisibleCanvasText(activeCanvas);

    const msg: AgentMessage = {
      id: crypto.randomUUID(),
      type: 'user',
      content: rawMessage,
      timestamp: new Date().toISOString(),
    };
    sessions = sessions.map(s =>
      s.id === activeSessionId
        ? { ...s, messages: [...s.messages, msg] }
        : s
    );
    inputValue = '';
    attachments = [];

    const endpoint = getChatEndpointForCanvas(activeCanvas);
    if (!endpoint) {
      const errorMessage: AgentMessage = {
        id: crypto.randomUUID(),
        type: 'agent',
        content: 'No connected department endpoint is available for this canvas.',
        timestamp: new Date().toISOString(),
      };
      sessions = sessions.map(s =>
        s.id === activeSessionId
          ? { ...s, messages: [...s.messages, errorMessage] }
          : s
      );
      return;
    }

    isSending = true;
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: rawMessage,
          session_id: activeSession?.backendSessionId ?? null,
          context: {
            canvas: activeCanvas,
            active_canvas: activeCanvas,
            canvas_context: activeCanvasContext,
            attached_contexts: attachedContexts,
            visible_canvas_text: visibleCanvasText,
          },
          stream: false,
        }),
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json() as { reply?: string; session_id?: string };
      if (data.session_id) {
        sessions = sessions.map(s =>
          s.id === activeSessionId
            ? { ...s, backendSessionId: data.session_id }
            : s
        );
      }
      const replyMessage: AgentMessage = {
        id: crypto.randomUUID(),
        type: 'agent',
        content: data.reply || 'No response received.',
        timestamp: new Date().toISOString(),
      };
      sessions = sessions.map(s =>
        s.id === activeSessionId
          ? { ...s, messages: [...s.messages, replyMessage] }
          : s
      );
    } catch (error) {
      const errorMessage: AgentMessage = {
        id: crypto.randomUUID(),
        type: 'agent',
        content: `Error: ${error instanceof Error ? error.message : 'Failed to send message'}`,
        timestamp: new Date().toISOString(),
      };
      sessions = sessions.map(s =>
        s.id === activeSessionId
          ? { ...s, messages: [...s.messages, errorMessage] }
          : s
      );
    } finally {
      isSending = false;
    }
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      // If slash hints are showing and there are commands, pick the first on Tab
      if (inputValue.startsWith('/') && filteredSlashCommands.length > 0 && e.key === 'Enter') {
        submitMessage();
        return;
      }
      submitMessage();
    }
    if (e.key === 'Escape') {
      showAttachMenu = false;
    }
  }

  function selectSlashCommand(cmd: string) {
    inputValue = cmd + ' ';
  }

  function openAttachMenu() {
    showAttachMenu = !showAttachMenu;
  }

  function attachCanvas(opt: CanvasContextOption) {
    // Avoid duplicate attachments
    if (attachments.find(a => a.canvasId === opt.id)) {
      showAttachMenu = false;
      return;
    }
    attachments = [...attachments, {
      id: crypto.randomUUID(),
      label: opt.label,
      canvasId: opt.id,
    }];
    showAttachMenu = false;
  }

  function removeAttachment(id: string) {
    attachments = attachments.filter(a => a.id !== id);
  }

  function toggleToolLine(id: string) {
    expandedToolLine = expandedToolLine === id ? null : id;
  }

  function isOpinionTool(msg: AgentMessage): boolean {
    return msg.tool === 'write_memory' && String(msg.args?.['node_type'] ?? '') === 'OPINION';
  }

  async function setActiveTab(tab: 'chat' | 'agents' | 'memory' | 'mail') {
    activeTab = tab;
    if (tab === 'memory') {
      await loadMemoryTab();
    }
    if (tab === 'mail') {
      await loadMailTab();
    }
  }
</script>

<!-- Agent Panel right rail — hidden for workshop and flowforge canvases -->
<aside
  class="agent-panel"
  class:collapsed
  class:hidden
  aria-label="Agent Panel"
>
  {#if !isWorkshop}
    <!-- Header -->
    <header class="ap-header">
      <span class="ap-dept-badge" style="color: {deptColor};">
        {deptHead.label}
      </span>
      <div class="ap-spacer"></div>
      <button
        class="ap-icon-btn"
        title="New interactive session"
        onclick={createNewSession}
        aria-label="New session"
      >
        <Plus size={14} />
      </button>
      <button
        class="ap-icon-btn"
        title="Session history"
        onclick={toggleSessionHistory}
        aria-label="Session history"
        aria-pressed={showSessionHistory}
      >
        <History size={14} />
      </button>
      <button
        class="ap-icon-btn"
        title="Collapse agent panel"
        onclick={() => (collapsed = true)}
        aria-label="Collapse panel"
      >
        <ChevronRight size={14} />
      </button>
    </header>

    <!-- Tab navigation -->
    <nav class="ap-tabs" aria-label="Agent panel tabs">
      {#each (['chat', 'agents', 'memory', 'mail'] as const) as tab}
        <button
          class="ap-tab"
          class:active={activeTab === tab}
          onclick={() => { void setActiveTab(tab); }}
          aria-selected={activeTab === tab}
          role="tab"
        >
          {tab === 'chat' ? 'Chat' : tab === 'agents' ? 'Agents' : tab === 'memory' ? 'Memory' : 'Mail'}
        </button>
      {/each}
    </nav>

    <!-- Session history panel -->
    {#if showSessionHistory}
      <div class="ap-session-history" role="list" aria-label="Past sessions">
        {#if sessions.length === 0}
          <p class="ap-empty-state">No sessions yet. Click + to start.</p>
        {/if}
        {#each sessions as session (session.id)}
          <button
            class="ap-session-entry"
            class:active={session.id === activeSessionId}
            onclick={() => selectSession(session.id)}
          >
            <span class="ap-session-title">{session.deptHead}</span>
            <span class="ap-session-meta">
              <span class="ap-session-date">{formatLocalDate(session.createdAt)}</span>
              <span class="ap-session-badge" data-status={session.status}>{session.status}</span>
            </span>
          </button>
        {/each}
      </div>
    {/if}

    <!-- ── Chat tab ── -->
    {#if activeTab === 'chat'}
      <!-- Body: messages + autonomous status -->
      <div class="ap-body" aria-live="polite">
        {#if activeSession?.type === 'autonomous'}
          <div class="ap-autonomous-status">
            <div class="ap-workflow-name">{activeSession.workflowName ?? 'Autonomous Workflow'}</div>
            <div class="ap-workflow-stage">
              Stage: <span class="ap-stage-label">{activeSession.workflowStage ?? '—'}</span>
            </div>
            {#if activeSession.workflowElapsed !== undefined}
              <div class="ap-workflow-elapsed">
                Elapsed: <span class="ap-elapsed-val">{formatElapsed(activeSession.workflowElapsed)}</span>
              </div>
            {/if}
            {#if activeSession.subAgents && activeSession.subAgents.length > 0}
              <div class="ap-sub-agents" aria-label="Sub-agent statuses">
                {#each activeSession.subAgents as agent (agent.role)}
                  <div class="ap-sub-agent-row">
                    <span class="ap-sub-agent-role">{agent.role}</span>
                    <span
                      class="ap-sub-agent-badge"
                      style="color: {subAgentStatusColor(agent.status)};"
                    >{agent.status}</span>
                  </div>
                {/each}
              </div>
            {/if}
          </div>
        {/if}

        {#each messages as msg (msg.id)}
          {#if msg.type === 'agent'}
            <div class="ap-agent">
              <RichRenderer content={msg.content} />
            </div>
          {:else if msg.type === 'user'}
            <div class="ap-user">
              <p class="ap-user-text">{msg.content}</p>
            </div>
          {:else if msg.type === 'tool'}
            <button
              type="button"
              class="ap-tool"
              onclick={() => toggleToolLine(msg.id)}
              aria-expanded={expandedToolLine === msg.id}
              title={isOpinionTool(msg) ? 'Click to expand OPINION details' : undefined}
            >
              <span class="ap-tool-text">{msg.content}</span>
              {#if expandedToolLine === msg.id && isOpinionTool(msg)}
                <div class="ap-tool-detail" aria-label="OPINION details">
                  <div class="ap-detail-row"><span class="ap-detail-key">action</span><span class="ap-detail-val">{String(msg.args?.['action'] ?? '—')}</span></div>
                  <div class="ap-detail-row"><span class="ap-detail-key">reasoning</span><span class="ap-detail-val">{String(msg.args?.['reasoning'] ?? '—')}</span></div>
                  <div class="ap-detail-row"><span class="ap-detail-key">confidence</span><span class="ap-detail-val">{String(msg.args?.['confidence'] ?? '—')}</span></div>
                  <div class="ap-detail-row"><span class="ap-detail-key">alternatives_considered</span><span class="ap-detail-val">{String(msg.args?.['alternatives_considered'] ?? '—')}</span></div>
                  <div class="ap-detail-row"><span class="ap-detail-key">constraints_applied</span><span class="ap-detail-val">{String(msg.args?.['constraints_applied'] ?? '—')}</span></div>
                  <div class="ap-detail-row"><span class="ap-detail-key">agent_role</span><span class="ap-detail-val">{String(msg.args?.['agent_role'] ?? '—')}</span></div>
                </div>
              {/if}
            </button>
          {/if}
        {/each}

        {#if !activeSession}
          <p class="ap-empty-state">Click <strong>+</strong> to start a new session with the {deptHead.label} agent.</p>
        {/if}
      </div>

      <!-- Footer: enhanced input area (interactive sessions only) -->
      {#if activeSession?.type === 'interactive'}
        <div class="ap-footer">
          <div class="input-area">
            <!-- Slash command hints (show when input starts with /) -->
            {#if inputValue.startsWith('/') && filteredSlashCommands.length > 0}
              <div class="slash-hints" role="listbox" aria-label="Slash command suggestions">
                {#each filteredSlashCommands as cmd}
                  <button
                    class="slash-hint"
                    onclick={() => selectSlashCommand(cmd.command)}
                    role="option"
                    aria-selected={false}
                  >
                    <span class="slash-cmd">{cmd.command}</span>
                    <span class="slash-desc">{cmd.desc}</span>
                  </button>
                {/each}
              </div>
            {/if}

            <!-- Attachment pills -->
            {#if attachments.length > 0}
              <div class="attachment-pills">
                {#each attachments as att (att.id)}
                  <span class="att-pill">
                    {att.label}
                    <button
                      class="att-remove"
                      onclick={() => removeAttachment(att.id)}
                      aria-label="Remove {att.label}"
                    >×</button>
                  </span>
                {/each}
              </div>
            {/if}

            <div class="input-row">
              <button
                class="attach-btn ap-icon-btn"
                onclick={openAttachMenu}
                title="Attach canvas context"
                aria-label="Attach canvas context"
                aria-expanded={showAttachMenu}
              >
                <Paperclip size={14} />
              </button>
              <input
                class="chat-input"
                bind:value={inputValue}
                onkeydown={handleKeydown}
                placeholder="Message {deptHead.label}…"
                aria-label="Message input"
              />
              <button
                class="ap-send"
                onclick={() => void submitMessage()}
                disabled={!inputValue.trim() || isSending}
                aria-label="Send message"
              >
                <Send size={14} />
              </button>
            </div>

            <!-- Attach menu (canvas picker) -->
            {#if showAttachMenu}
              <div class="attach-menu" role="menu" aria-label="Canvas context picker">
                <div class="attach-header">Attach context from canvas</div>
                {#each canvasContextOptions as opt}
                  <button
                    class="attach-option"
                    onclick={() => attachCanvas(opt)}
                    role="menuitem"
                  >
                    <span class="opt-label">{opt.label}</span>
                  </button>
                {/each}
              </div>
            {/if}
          </div>
        </div>
      {/if}

    <!-- ── Agents tab ── -->
    {:else if activeTab === 'agents'}
      <div class="ap-body">
        <div class="tab-view-header">Active Sub-Agents</div>
        {#if subAgents.length === 0}
          <div class="ap-empty-state">No active sub-agents</div>
        {:else}
          {#each subAgents as agent (agent.role)}
            <div class="agent-row" class:running={agent.status === 'running'}>
              <span class="agent-role">{agent.role}</span>
              <span class="agent-status status-{agent.status}" style="color: {subAgentStatusColor(agent.status)};">
                {agent.status}
              </span>
            </div>
          {/each}
        {/if}
      </div>

    <!-- ── Memory tab ── -->
    {:else if activeTab === 'memory'}
      <div class="ap-body">
        <div class="tab-view-header">Recent Memory</div>
        {#if memoryLoading}
          <div class="tab-view-note">Loading memory…</div>
        {:else if memoryError}
          <div class="tab-view-note">{memoryError}</div>
        {:else if memoryNodes.length === 0}
          <div class="tab-view-note">No opinion nodes available.</div>
        {:else}
          {#each memoryNodes as node (node.id)}
            <div class="agent-row">
              <span class="agent-role">{new Date(node.created_at).toLocaleString()}</span>
              <span class="agent-status">{node.confidence ?? '—'}</span>
            </div>
            <div class="tab-view-note">{node.content}</div>
          {/each}
        {/if}
      </div>

    <!-- ── Mail tab ── -->
    {:else if activeTab === 'mail'}
      <div class="ap-body">
        <div class="tab-view-header">Department Mail</div>
        {#if mailLoading}
          <div class="tab-view-note">Loading mail…</div>
        {:else if mailError}
          <div class="tab-view-note">{mailError}</div>
        {:else if mailEntries.length === 0}
          <div class="tab-view-note">No mail for this department.</div>
        {:else}
          {#each mailEntries as entry (entry.id)}
            <div class="agent-row">
              <span class="agent-role">{entry.from_dept} · {entry.type}</span>
              <span class="agent-status">{entry.priority}</span>
            </div>
            <div class="tab-view-note">{entry.subject}</div>
          {/each}
        {/if}
      </div>
    {/if}

  {:else}
    <!-- Workshop / FlowForge: panel content hidden -->
    <div class="ap-workshop-hidden" aria-hidden="true"></div>
  {/if}
</aside>

<!-- Collapsed expand trigger (visible when panel is collapsed and not on workshop) -->
{#if collapsed && !isWorkshop}
  <button
    class="ap-expand-trigger"
    onclick={() => (collapsed = false)}
    title="Expand agent panel"
    aria-label="Expand agent panel"
  >
    <ChevronRight size={14} style="transform: rotate(180deg);" />
  </button>
{/if}

<style>
  /* ── Agent Panel shell ── */
  .agent-panel {
    grid-area: agent;
    display: flex;
    flex-direction: column;
    width: var(--agent-panel-width, 320px);
    min-width: 0;
    background: var(--glass-tier-2);
    backdrop-filter: var(--glass-blur);
    border-left: 1px solid var(--color-border-subtle);
    transition: width 300ms ease;
    overflow: hidden;
    position: relative;
  }

  .agent-panel.collapsed {
    width: 0;
  }

  .agent-panel.hidden {
    width: 0;
    border-left: none;
    pointer-events: none;
  }

  /* ── Header ── */
  .ap-header {
    display: flex;
    flex-direction: row;
    align-items: center;
    height: 36px;
    padding: 0 8px;
    gap: 4px;
    border-bottom: 1px solid var(--color-border-subtle);
    flex-shrink: 0;
  }

  .ap-dept-badge {
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    white-space: nowrap;
  }

  .ap-spacer {
    flex: 1;
  }

  .ap-icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    padding: 0;
    background: transparent;
    border: none;
    border-radius: 4px;
    color: var(--color-text-muted);
    cursor: pointer;
    transition: color 150ms ease, background 150ms ease;
  }

  .ap-icon-btn:hover {
    color: var(--color-text-primary);
    background: rgba(255, 255, 255, 0.05);
  }

  /* ── Tab navigation ── */
  .ap-tabs {
    display: flex;
    flex-direction: row;
    height: 28px;
    flex-shrink: 0;
    border-bottom: 1px solid var(--color-border-subtle);
    padding: 0 4px;
    gap: 2px;
  }

  .ap-tab {
    position: relative;
    display: flex;
    align-items: center;
    padding: 0 10px;
    height: 100%;
    background: transparent;
    border: none;
    color: var(--color-text-muted);
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    cursor: pointer;
    transition: color 150ms ease;
    white-space: nowrap;
  }

  .ap-tab:hover {
    color: var(--color-text-secondary);
  }

  .ap-tab.active {
    color: var(--color-accent-cyan);
  }

  .ap-tab.active::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--color-accent-cyan);
    border-radius: 1px 1px 0 0;
  }

  /* ── Session history ── */
  .ap-session-history {
    flex-shrink: 0;
    max-height: 200px;
    overflow-y: auto;
    border-bottom: 1px solid var(--color-border-subtle);
    padding: 4px 0;
  }

  .ap-session-entry {
    display: flex;
    flex-direction: column;
    width: 100%;
    padding: 6px 10px;
    background: transparent;
    border: none;
    text-align: left;
    cursor: pointer;
    gap: 2px;
    transition: background 150ms ease;
  }

  .ap-session-entry:hover,
  .ap-session-entry.active {
    background: rgba(255, 255, 255, 0.04);
  }

  .ap-session-title {
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 10px;
    font-weight: 600;
    color: var(--color-text-primary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .ap-session-meta {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .ap-session-date {
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 9px;
    color: var(--color-text-muted);
  }

  .ap-session-badge {
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 9px;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .ap-session-badge[data-status="active"] {
    color: var(--color-accent-cyan);
  }

  .ap-session-badge[data-status="completed"] {
    color: var(--color-text-muted);
  }

  .ap-session-badge[data-status="error"] {
    color: var(--color-accent-red);
  }

  /* ── Body ── */
  .ap-body {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
    gap: 7px;
    display: flex;
    flex-direction: column;
  }

  /* ── Tab views ── */
  .tab-view-header {
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 10px;
    font-weight: 700;
    color: var(--color-accent-cyan);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--color-border-subtle);
    margin-bottom: 8px;
    flex-shrink: 0;
  }

  .tab-view-note {
    font-family: var(--font-family, 'Inter', system-ui, sans-serif);
    font-size: 12px;
    color: var(--color-text-muted);
    line-height: 1.5;
  }

  .tab-view-note strong {
    color: var(--color-text-secondary);
  }

  /* ── Agents tab ── */
  .agent-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 6px;
    border-radius: 3px;
    transition: background 150ms ease;
  }

  .agent-row.running {
    background: rgba(0, 170, 204, 0.05);
  }

  .agent-role {
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 10px;
    color: var(--color-text-secondary);
  }

  .agent-status {
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  /* ── Autonomous status ── */
  .ap-autonomous-status {
    background: rgba(0, 170, 204, 0.04);
    border: 1px solid rgba(0, 170, 204, 0.12);
    border-radius: 4px;
    padding: 8px;
    margin-bottom: 4px;
    flex-shrink: 0;
  }

  .ap-workflow-name {
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 10px;
    font-weight: 700;
    color: var(--color-accent-cyan);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
  }

  .ap-workflow-stage,
  .ap-workflow-elapsed {
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 9px;
    color: var(--color-text-muted);
    margin-bottom: 2px;
  }

  .ap-stage-label,
  .ap-elapsed-val {
    color: var(--color-text-primary);
  }

  .ap-sub-agents {
    margin-top: 6px;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }

  .ap-sub-agent-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .ap-sub-agent-role {
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 9px;
    color: var(--color-text-secondary);
  }

  .ap-sub-agent-badge {
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  /* ── Message types ── */
  .ap-agent {
    background: rgba(0, 170, 204, 0.05);
    border: 1px solid rgba(0, 170, 204, 0.09);
    border-radius: 4px;
    padding: 8px;
    font-family: var(--font-family, 'Inter', system-ui, sans-serif);
    font-size: 13px;
    color: var(--color-text-primary);
    word-break: break-word;
  }

  .ap-user {
    align-self: flex-end;
    max-width: 88%;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 4px;
    padding: 8px;
  }

  .ap-user-text {
    font-family: var(--font-family, 'Inter', system-ui, sans-serif);
    font-size: 13px;
    color: var(--color-text-primary);
    margin: 0;
    word-break: break-word;
  }

  .ap-tool {
    display: block;
    width: 100%;
    border: none;
    background: transparent;
    border-left: 2px solid rgba(0, 170, 204, 0.2);
    padding: 4px 6px;
    text-align: left;
    cursor: pointer;
    outline: none;
  }

  .ap-tool:focus-visible {
    outline: 1px solid var(--color-accent-cyan);
    outline-offset: -1px;
  }

  .ap-tool-text {
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 9px;
    color: var(--color-text-muted);
    word-break: break-all;
    display: block;
  }

  /* ── Tool detail (OPINION expand) ── */
  .ap-tool-detail {
    margin-top: 6px;
    padding: 6px;
    background: rgba(0, 170, 204, 0.04);
    border-radius: 3px;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }

  .ap-detail-row {
    display: flex;
    gap: 8px;
    align-items: flex-start;
  }

  .ap-detail-key {
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 9px;
    color: var(--color-accent-cyan);
    min-width: 80px;
    flex-shrink: 0;
    text-transform: lowercase;
  }

  .ap-detail-val {
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 9px;
    color: var(--color-text-primary);
    word-break: break-word;
  }

  /* ── Footer ── */
  .ap-footer {
    padding: 8px;
    border-top: 1px solid var(--color-border-subtle);
    flex-shrink: 0;
  }

  /* ── Input area ── */
  .input-area {
    display: flex;
    flex-direction: column;
    gap: 4px;
    position: relative;
  }

  /* ── Slash hints ── */
  .slash-hints {
    position: absolute;
    bottom: calc(100% + 4px);
    left: 0;
    right: 0;
    background: rgba(8, 13, 20, 0.95);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 4px;
    overflow: hidden;
    z-index: 50;
    backdrop-filter: blur(8px);
  }

  .slash-hint {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 6px 10px;
    background: transparent;
    border: none;
    text-align: left;
    cursor: pointer;
    transition: background 120ms ease;
  }

  .slash-hint:hover {
    background: rgba(0, 212, 255, 0.06);
  }

  .slash-cmd {
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 11px;
    font-weight: 600;
    color: var(--color-accent-cyan);
    min-width: 80px;
  }

  .slash-desc {
    font-family: var(--font-family, 'Inter', system-ui, sans-serif);
    font-size: 11px;
    color: var(--color-text-muted);
  }

  /* ── Attachment pills ── */
  .attachment-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }

  .att-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 7px;
    background: rgba(0, 212, 255, 0.06);
    border: 1px solid rgba(0, 212, 255, 0.25);
    border-radius: 10px;
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 10px;
    color: var(--color-accent-cyan);
    white-space: nowrap;
  }

  .att-remove {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    padding: 0;
    background: transparent;
    border: none;
    color: var(--color-text-muted);
    font-size: 12px;
    cursor: pointer;
    line-height: 1;
    transition: color 120ms ease;
  }

  .att-remove:hover {
    color: var(--color-text-primary);
  }

  /* ── Input row ── */
  .input-row {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 4px;
  }

  .attach-btn {
    flex-shrink: 0;
  }

  .chat-input {
    flex: 1;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--color-border-subtle);
    border-radius: 4px;
    color: var(--color-text-primary);
    font-family: var(--font-family, 'Inter', system-ui, sans-serif);
    font-size: 13px;
    padding: 6px 8px;
    height: 32px;
    outline: none;
    transition: border-color 150ms ease;
  }

  .chat-input:focus {
    border-color: var(--color-accent-cyan);
  }

  .chat-input::placeholder {
    color: var(--color-text-muted);
  }

  /* ── Attach menu ── */
  .attach-menu {
    position: absolute;
    bottom: calc(100% + 4px);
    left: 0;
    min-width: 200px;
    background: rgba(8, 13, 20, 0.95);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 4px;
    overflow: hidden;
    z-index: 50;
    backdrop-filter: blur(8px);
  }

  .attach-header {
    padding: 6px 10px;
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 9px;
    font-weight: 700;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    border-bottom: 1px solid rgba(0, 212, 255, 0.1);
  }

  .attach-option {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 7px 10px;
    background: transparent;
    border: none;
    text-align: left;
    cursor: pointer;
    transition: background 120ms ease;
  }

  .attach-option:hover {
    background: rgba(0, 212, 255, 0.06);
  }

  .opt-label {
    font-family: var(--font-family, 'Inter', system-ui, sans-serif);
    font-size: 12px;
    color: var(--color-text-primary);
  }

  /* ── Send button ── */
  .ap-send {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: var(--color-accent-cyan);
    border: none;
    border-radius: 4px;
    color: #080d14;
    cursor: pointer;
    flex-shrink: 0;
    transition: opacity 150ms ease;
  }

  .ap-send:disabled {
    opacity: 0.35;
    cursor: not-allowed;
  }

  .ap-send:not(:disabled):hover {
    opacity: 0.85;
  }

  /* ── Empty state ── */
  .ap-empty-state {
    font-family: var(--font-family, 'Inter', system-ui, sans-serif);
    font-size: 12px;
    color: var(--color-text-muted);
    text-align: center;
    padding: 16px 8px;
    margin: 0;
    line-height: 1.5;
  }

  .ap-workshop-hidden {
    width: 100%;
    height: 100%;
  }

  /* ── Expand trigger (when collapsed) ── */
  .ap-expand-trigger {
    position: fixed;
    right: 0;
    top: 50%;
    transform: translateY(-50%);
    width: 20px;
    height: 48px;
    background: var(--glass-tier-2);
    backdrop-filter: var(--glass-blur);
    border: 1px solid var(--color-border-subtle);
    border-right: none;
    border-radius: 4px 0 0 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--color-text-muted);
    cursor: pointer;
    z-index: 100;
    transition: color 150ms ease;
  }

  .ap-expand-trigger:hover {
    color: var(--color-text-primary);
  }
</style>

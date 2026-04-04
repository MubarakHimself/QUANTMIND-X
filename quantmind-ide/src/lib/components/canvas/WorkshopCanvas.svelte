<script lang="ts">
  /**
   * Workshop Canvas — redesigned
   *
   * Main Copilot / FloorManager interface — the conversation IS the canvas.
   * Claude.ai-inspired layout: persistent left nav sidebar + main content area.
   * Left nav items navigate between chat, projects, memory, skills, workflows.
   * Each section fills the main content area (no separate right panel).
   */
  import { onMount, tick } from 'svelte';
  import {
    Plus,
    MessageSquare,
    FolderOpen,
    Brain,
    Zap,
    GitBranch,
    Send,
    Paperclip,
    ChevronRight,
    Bot,
    Loader,
    User,
    Trash2,
    Wrench,
    Pencil,
    Check,
    X
  } from 'lucide-svelte';
  import { chatApi, type ChatSession, type StoredChatMessage } from '$lib/api/chatApi';
  import { listSkills, type Skill } from '$lib/api/skillsApi';
  import { listAllAssets } from '$lib/api/sharedAssetsApi';
  import { getHotNodes, getWarmNodes, type GraphMemoryNode, listOpinionNodes, createOpinionNode, type OpinionNode } from '$lib/api/graphMemory';
  import { canvasContextService, type CanvasAttachableResource } from '$lib/services/canvasContextService';
  import { navigationStore } from '$lib/stores/navigationStore';
  import { copilotKillSwitchService } from '$lib/services/copilotKillSwitchService';
  import { getBaseUrl } from '$lib/config/api';
  import { apiFetch } from '$lib/api';
  import { marked } from 'marked';
  import DOMPurify from 'dompurify';

  function renderMarkdown(text: string): string {
    if (!text) return '';
    const html = marked.parse(text, { breaks: true, gfm: true }) as string;
    return DOMPurify.sanitize(html);
  }

  // ── Types ────────────────────────────────────────────────────────────────────

  interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    isStreaming?: boolean;
    messageType?: 'text' | 'audit_timeline' | 'audit_reasoning' | 'morning_digest';
    metadata?: Record<string, unknown>;
    toolCalls?: Array<{ name: string; input: Record<string, unknown>; result?: string }>;
  }

  interface WorkflowItem {
    id: string;
    name: string;
    status: string;
    last_run?: string;
  }

interface ProjectItem {
  id: string;
  name: string;
  description?: string;
  status?: string;
}

  interface WorkshopAttachment {
    id: string;
    label: string;
    canvasId: string;
    kind: 'canvas' | 'resource';
    resource?: CanvasAttachableResource;
  }

  interface WorkshopCanvasContextOption {
    id: string;
    label: string;
  }

  // ── State ────────────────────────────────────────────────────────────────────

  let activeSection = $state<'chat' | 'projects' | 'memory' | 'skills' | 'workflows' | 'subagents'>('chat');

  // Chat
  let messages = $state<Message[]>([]);
  let inputMessage = $state('');
  let isLoading = $state(false);
  let currentSessionId = $state<string | null>(null);
  let messagesEnd = $state<HTMLElement | null>(null);
  let attachments = $state<WorkshopAttachment[]>([]);
  let showAttachMenu = $state(false);
  let attachableResources = $state<CanvasAttachableResource[]>([]);

  // Sessions (recent history in sidebar)
  let sessions = $state<ChatSession[]>([]);
  let sessionsLoading = $state(false);
  let sessionActionLoading = $state(false);
  let renamingSessionId = $state<string | null>(null);
  let renameSessionTitle = $state('');

  // Skills
  let skills = $state<Skill[]>([]);
  let skillsLoading = $state(false);
  let selectedSkill = $state<Skill | null>(null);
  let queuedSkillCommand = $state('');

  // Memory
  let memoryNodes = $state<GraphMemoryNode[]>([]);
  let memoryLoading = $state(false);
  let memoryFilter = $state<'all' | 'hot' | 'warm'>('all');
  let expandedNodeId = $state<string | null>(null);

  // Memory sub-tab
  let memoryTab = $state<'nodes' | 'opinions'>('nodes');

  // Opinion nodes
  let opinionNodes = $state<OpinionNode[]>([]);
  let opinionsLoading = $state(false);
  let opinionContent = $state('');
  let opinionConfidence = $state(0.7);
  let opinionSaving = $state(false);

  // Auto-compaction threshold
  const TOKEN_COMPACT_THRESHOLD = 160000;

  // Workflows
  let workflows = $state<WorkflowItem[]>([]);
  let workflowsLoading = $state(false);

  // Sub-agents
  interface SubAgent { id: string; type: string; task: string; department: string | null; status: string; created_at: string }
  const SUBAGENT_TYPES = [
    { id: 'strategy_researcher', label: 'Strategy Researcher' },
    { id: 'market_analyst',       label: 'Market Analyst' },
    { id: 'backtester',           label: 'Backtester' },
    { id: 'mql5_dev',             label: 'MQL5 Dev' },
  ];
  let runningAgents = $state<SubAgent[]>([]);
  let agentsLoading = $state(false);
  let spawnAgentType = $state('strategy_researcher');
  let spawnTask = $state('');
  let spawnModel = $state('');
  let spawnProvider = $state('');
  let spawnDropdownOpen = $state<string | null>(null); // agent type id with open dropdown
  let spawning = $state(false);

  // Projects
  let projects = $state<ProjectItem[]>([]);
  let projectsLoading = $state(false);

  // Token counter
  let totalTokens = $state(0);

  // Slash command menu
  interface SlashCommand { name: string; description: string; action: () => void }
  let slashMenuOpen = $state(false);
  let slashMenuIndex = $state(0);
  let slashMenuFilter = $state('');

  // Tool call tile expand state — keyed by `${msgId}:${toolIndex}`
  let expandedToolCalls = $state<Set<string>>(new Set());

  // Model selector
  interface ProviderOption {
    id: string;
    display_name: string;
    models: Array<{ id: string; name: string }>;
  }
  let availableProviders = $state<ProviderOption[]>([]);
  let selectedProvider = $state('');
  let selectedModel = $state('');
  let modelDropdownOpen = $state(false);

  const selectedModelLabel = $derived(() => {
    for (const p of availableProviders) {
      const m = p.models.find(m => m.id === selectedModel);
      if (m) return m.name;
    }
    return selectedModel || 'Select model';
  });

  // ── Derived ──────────────────────────────────────────────────────────────────

  const greeting = $derived(() => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning';
    if (hour < 17) return 'Good afternoon';
    return 'Good evening';
  });

  const currentTime = $derived(() => {
    return new Date().toLocaleTimeString('en-GB', {
      hour: '2-digit',
      minute: '2-digit'
    });
  });

  // ── Constants ────────────────────────────────────────────────────────────────

  const API_BASE = getBaseUrl('');
  const ATTACH_CANVAS_OPTIONS: WorkshopCanvasContextOption[] = [
    { id: 'research', label: 'Research context' },
    { id: 'development', label: 'Development context' },
    { id: 'risk', label: 'Risk context' },
    { id: 'trading', label: 'Trading context' },
    { id: 'portfolio', label: 'Portfolio context' },
    { id: 'shared-assets', label: 'Shared Assets context' },
    { id: 'flowforge', label: 'FlowForge context' },
  ];

  const SUGGESTION_CHIPS = [
    'What happened overnight?',
    'Show pending approvals',
    'Research a new strategy',
    'Check active workflows',
    'Review department status'
  ];

  // ── Lifecycle ────────────────────────────────────────────────────────────────

  onMount(async () => {
    await Promise.all([loadSessions(), loadSkills(), loadProviders()]);
    loadCanvasContext();
  });

  // ── Data loaders ─────────────────────────────────────────────────────────────

  async function loadCanvasContext() {
    try {
      await canvasContextService.loadCanvasContext('workshop');
    } catch (e) {
      console.error('Failed to load canvas context:', e);
    }
  }

  async function openAttachMenu() {
    const nextOpen = !showAttachMenu;
    showAttachMenu = nextOpen;
    if (!nextOpen) {
      return;
    }

    const resources: CanvasAttachableResource[] = [];
    for (const option of ATTACH_CANVAS_OPTIONS) {
      await canvasContextService.getEnrichedContext(option.id);
      resources.push(
        ...canvasContextService.getAttachableResources(option.id).map((resource) => ({
          ...resource,
          label: `${option.label}: ${resource.label}`,
        })),
      );
    }
    attachableResources = resources.slice(0, 150);
  }

  function attachCanvasContext(option: WorkshopCanvasContextOption) {
    if (attachments.find((attachment) => attachment.kind === 'canvas' && attachment.canvasId === option.id)) {
      showAttachMenu = false;
      return;
    }
    attachments = [...attachments, {
      id: crypto.randomUUID(),
      label: option.label,
      canvasId: option.id,
      kind: 'canvas',
    }];
    showAttachMenu = false;
  }

  function attachResource(resource: CanvasAttachableResource) {
    if (attachments.find((attachment) => attachment.kind === 'resource' && attachment.resource?.id === resource.id)) {
      showAttachMenu = false;
      return;
    }
    attachments = [...attachments, {
      id: crypto.randomUUID(),
      label: resource.label,
      canvasId: resource.canvas,
      kind: 'resource',
      resource,
    }];
    showAttachMenu = false;
  }

  function removeAttachment(id: string) {
    attachments = attachments.filter((attachment) => attachment.id !== id);
  }

  async function loadProviders() {
    try {
      const currentConfig = await apiFetch<{ provider?: string; model?: string }>('/agent-config/floor_manager/model')
        .catch(() => ({}));
      const data = await apiFetch<{
        providers?: Array<{
          id: string;
          name?: string;
          provider_type?: string;
          display_name: string;
          has_api_key?: boolean;
          is_active?: boolean;
          available?: boolean;
          models?: Array<{ id: string; name: string }>;
        }>;
      }>('/providers/available');

      const providers: ProviderOption[] = (data.providers || [])
        .filter((p) => Boolean(p.available))
        .map((p) => {
          const providerId = p.provider_type || p.name || p.id;
          return {
            id: providerId,
            display_name: p.display_name,
            models: Array.isArray(p.models) ? p.models : [],
          };
        })
        .filter((p) => Boolean(p.id) && p.models.length > 0);

      availableProviders = providers;

      if (providers.length > 0) {
        const configuredProvider = currentConfig.provider
          ? providers.find((p) => p.id === currentConfig.provider)
          : undefined;
        const providerByModel = currentConfig.model
          ? providers.find((p) => p.models.some((model) => model.id === currentConfig.model))
          : undefined;
        const chosen = configuredProvider || providerByModel || providers[0];
        selectedProvider = chosen.id;
        selectedModel = currentConfig.model && chosen.models.some((model) => model.id === currentConfig.model)
          ? currentConfig.model
          : (chosen.models[0]?.id || '');
      } else {
        selectedProvider = '';
        selectedModel = '';
      }
    } catch (e) {
      console.error('Failed to load providers for model selector:', e);
      availableProviders = [];
      selectedProvider = '';
      selectedModel = '';
    }
  }

  async function loadSessions() {
    sessionsLoading = true;
    try {
      sessions = (await listWorkshopSessions()).slice(0, 20);
    } catch (e) {
      console.error('Failed to load sessions:', e);
    } finally {
      sessionsLoading = false;
    }
  }

  async function listWorkshopSessions() {
    const floorManagerSessions = await chatApi.listSessions(undefined, 'floor-manager');
    return floorManagerSessions.sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime());
  }

  async function loadSkills() {
    skillsLoading = true;
    try {
      skills = await listSkills();
    } catch (e) {
      console.error('Failed to load skills:', e);
    } finally {
      skillsLoading = false;
    }
  }

  async function loadMemoryNodes() {
    memoryLoading = true;
    try {
      if (memoryFilter === 'hot') {
        memoryNodes = await getHotNodes(50);
      } else if (memoryFilter === 'warm') {
        memoryNodes = await getWarmNodes(50);
      } else {
        const [hot, warm] = await Promise.all([getHotNodes(25), getWarmNodes(25)]);
        memoryNodes = [...hot, ...warm];
      }
    } catch (e) {
      console.error('Failed to load memory nodes:', e);
    } finally {
      memoryLoading = false;
    }
  }

  async function loadOpinionNodes() {
    opinionsLoading = true;
    try {
      opinionNodes = await listOpinionNodes(50);
    } catch (e) {
      console.error('Failed to load opinion nodes:', e);
    } finally {
      opinionsLoading = false;
    }
  }

  async function saveOpinion() {
    if (!opinionContent.trim() || opinionSaving) return;
    opinionSaving = true;
    try {
      await createOpinionNode(opinionContent.trim(), opinionConfidence);
      opinionContent = '';
      opinionConfidence = 0.7;
      await loadOpinionNodes();
    } catch (e) {
      console.error('Failed to save opinion:', e);
    } finally {
      opinionSaving = false;
    }
  }

  async function compactSession() {
    if (!currentSessionId) return;
    try {
      const res = await fetch(`${API_BASE}/chat/sessions/${currentSessionId}/compact`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const data = await res.json();
      if (data.compacted) {
        messages = [...messages, {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: `[Context compacted \u2014 summary saved to memory]`,
          timestamp: new Date(),
          messageType: 'text'
        }];
        totalTokens = 0;
        await tick();
        messagesEnd?.scrollIntoView({ behavior: 'smooth' });
      }
    } catch (e) {
      console.error('Compact session failed:', e);
    }
  }

  async function loadWorkflows() {
    workflowsLoading = true;
    try {
      const res = await fetch(`${API_BASE}/prefect/workflows`);
      if (res.ok) {
        const data = await res.json();
        // Endpoint returns { workflows: [...], by_state: {...}, total: N }
        // Normalize to WorkflowItem[] — map state → status
        const raw: Array<{ id: string; name: string; state?: string; status?: string; started_at?: string | null }> =
          Array.isArray(data) ? data : (data.workflows ?? []);
        workflows = raw.map(wf => ({
          id: wf.id,
          name: wf.name,
          status: (wf.status ?? wf.state ?? 'unknown').toLowerCase(),
          last_run: wf.started_at ?? undefined
        }));
      }
    } catch (e) {
      console.error('Failed to load workflows:', e);
    } finally {
      workflowsLoading = false;
    }
  }

  async function loadProjects() {
    projectsLoading = true;
    try {
      const grouped = await listAllAssets();
      const items: ProjectItem[] = [];
      for (const [assetType, records] of Object.entries(grouped)) {
        for (const record of records) {
          items.push({
            id: record.id,
            name: record.name,
            description: record.source_path || '',
            status: assetType,
          });
        }
      }
      projects = items.sort((a, b) => a.name.localeCompare(b.name));
    } catch (e) {
      console.error('Failed to load projects:', e);
    } finally {
      projectsLoading = false;
    }
  }

  async function loadRunningAgents() {
    agentsLoading = true;
    try {
      const res = await fetch(`${API_BASE}/floor-manager/subagents/list`);
      if (res.ok) {
        const data = await res.json();
        runningAgents = data.agents ?? [];
      }
    } catch (e) {
      console.error('Failed to load sub-agents:', e);
    } finally {
      agentsLoading = false;
    }
  }

  async function spawnAgent(agentType: string, taskText: string, modelOverride: string, providerOverride: string) {
    if (!taskText.trim()) return;
    spawning = true;
    try {
      const body: Record<string, string> = { agent_type: agentType, task: taskText };
      if (modelOverride) body.model = modelOverride;
      if (providerOverride) body.provider = providerOverride;
      const res = await fetch(`${API_BASE}/floor-manager/subagents/spawn`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      if (res.ok) {
        spawnTask = '';
        await loadRunningAgents();
      }
    } catch (e) {
      console.error('Failed to spawn sub-agent:', e);
    } finally {
      spawning = false;
    }
  }

  // ── Navigation ───────────────────────────────────────────────────────────────

  function navigateTo(section: typeof activeSection) {
    activeSection = section;
    if (section === 'memory') {
      if (memoryNodes.length === 0) loadMemoryNodes();
      if (opinionNodes.length === 0) loadOpinionNodes();
    } else if (section === 'workflows' && workflows.length === 0) {
      loadWorkflows();
    } else if (section === 'projects' && projects.length === 0) {
      loadProjects();
    } else if (section === 'subagents') {
      loadRunningAgents();
    }
  }

  // ── Chat actions ─────────────────────────────────────────────────────────────

  async function startNewChat() {
    messages = [];
    activeSection = 'chat';
    inputMessage = '';
    attachments = [];
    showAttachMenu = false;

    sessionActionLoading = true;
    try {
      const newSession = await chatApi.createSession({
        agentType: 'floor-manager',
        agentId: 'floor_manager',
        userId: 'anonymous',
        context: {
          canvas: 'workshop',
          active_canvas: 'workshop',
          session_type: 'interactive_session',
        },
      });
      currentSessionId = newSession.id;
      sessions = [newSession, ...sessions.filter((session) => session.id !== newSession.id)]
        .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
        .slice(0, 20);
    } catch (e) {
      console.error('Failed to create workshop session:', e);
      currentSessionId = null;
    } finally {
      sessionActionLoading = false;
    }
  }

  async function selectSession(session: ChatSession) {
    currentSessionId = session.id;
    activeSection = 'chat';
    messages = [];
    try {
      const stored: StoredChatMessage[] = await chatApi.getSessionMessages(session.id);
      messages = stored.map(m => ({
        id: m.id,
        role: m.role,
        content: m.content,
        timestamp: new Date(m.created_at),
        metadata: m.metadata
      }));
      await tick();
      messagesEnd?.scrollIntoView({ behavior: 'smooth' });
    } catch (e) {
      console.error('Failed to load session messages:', e);
    }
  }

  async function deleteSession(sessionId: string, e: MouseEvent) {
    e.stopPropagation();
    if (renamingSessionId === sessionId) {
      renamingSessionId = null;
      renameSessionTitle = '';
    }
    sessions = sessions.filter(s => s.id !== sessionId);
    if (currentSessionId === sessionId) {
      currentSessionId = null;
      messages = [];
    }
    try {
      await chatApi.deleteSession(sessionId);
    } catch (err) {
      console.error('Failed to delete session:', err);
      // Re-load to restore state if delete failed
      await loadSessions();
    }
  }

  function startRenameSession(session: ChatSession, e: MouseEvent) {
    e.stopPropagation();
    renamingSessionId = session.id;
    renameSessionTitle = (session.title || '').trim();
  }

  function cancelRenameSession(e?: Event) {
    e?.stopPropagation();
    renamingSessionId = null;
    renameSessionTitle = '';
  }

  async function commitRenameSession(sessionId: string, e?: Event) {
    e?.stopPropagation();
    const nextTitle = renameSessionTitle.trim();
    if (!nextTitle) {
      cancelRenameSession();
      return;
    }

    const previous = sessions;
    sessions = sessions.map((session) =>
      session.id === sessionId ? { ...session, title: nextTitle } : session
    );
    renamingSessionId = null;
    renameSessionTitle = '';

    try {
      await chatApi.updateSessionTitle(sessionId, { title: nextTitle });
    } catch (err) {
      console.error('Failed to rename session:', err);
      sessions = previous;
      await loadSessions();
    }
  }

  async function deleteAllSessions() {
    if (sessions.length === 0 || sessionActionLoading) return;

    const allSessions = await listWorkshopSessions();
    const sessionIds = allSessions.map((session) => session.id);
    const previousSessions = sessions;
    const activeSessionWasDeleted = currentSessionId ? sessionIds.includes(currentSessionId) : false;

    sessions = [];
    if (activeSessionWasDeleted) {
      currentSessionId = null;
      messages = [];
    }

    sessionActionLoading = true;
    try {
      await Promise.all(sessionIds.map((sessionId) => chatApi.deleteSession(sessionId)));
    } catch (e) {
      console.error('Failed to delete all workshop sessions:', e);
      sessions = previousSessions;
      await loadSessions();
    } finally {
      sessionActionLoading = false;
    }
  }

  async function sendMessage(content?: string) {
    const text = content ?? inputMessage;
    if (!text.trim() || isLoading) return;

    // Kill switch check — default to inactive if endpoint unavailable
    let ksActive = false;
    try {
      const ks = await copilotKillSwitchService.getStatus();
      if (ks.active) {
        messages = [...messages, {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: `Agent system is currently disabled. Please try again later.`,
          timestamp: new Date()
        }];
        return;
      }
    } catch {
      // Kill switch endpoint unavailable — proceed normally
      ksActive = false;
    }
    void ksActive;

    activeSection = 'chat';

    const attachedContexts = attachments.length > 0
      ? await Promise.all(
          attachments.map(async (attachment) => {
            if (attachment.kind === 'resource' && attachment.resource) {
              return {
                canvas: attachment.canvasId,
                label: attachment.label,
                context: {
                  attachment_type: 'resource',
                  resource: canvasContextService.buildResourceAttachmentContext(attachment.resource),
                },
              };
            }

            const loadedContext = await canvasContextService.getChatContext(
              attachment.canvasId,
              currentSessionId ?? undefined,
            );
            return {
              canvas: attachment.canvasId,
              label: attachment.label,
              context: loadedContext,
            };
          }),
        )
      : [];

    messages = [...messages, {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      timestamp: new Date()
    }];
    inputMessage = '';
    attachments = [];
    showAttachMenu = false;
    isLoading = true;

    const assistantId = crypto.randomUUID();
    messages = [...messages, {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true
    }];

    await tick();
    messagesEnd?.scrollIntoView({ behavior: 'smooth' });

    try {
      const workshopContext = await canvasContextService.getChatContext(
        'workshop',
        currentSessionId ?? undefined
      );
      // inject model/provider selection for backend routing
      workshopContext.model = selectedModel;
      if (selectedProvider) workshopContext.provider = selectedProvider;
      const result = await chatApi.sendMessage(
        'floor-manager',
        text,
        currentSessionId ?? undefined,
        false,
        undefined,
        {
          canvas: 'workshop',
          active_canvas: 'workshop',
          canvas_context: workshopContext,
          attached_contexts: attachedContexts,
          session_type: 'interactive_session',
          workspace_contract: {
            version: 'manifest-v1',
            strategy: 'manifest-first',
            natural_resource_search: true,
          },
        }
      );
      if (result.session_id && result.session_id !== currentSessionId) {
        currentSessionId = result.session_id;
      }
      await loadSessions();

      // Update token counter from usage if present
      if (result.usage) {
        const inp = result.usage.input_tokens ?? 0;
        const out = result.usage.output_tokens ?? 0;
        totalTokens = totalTokens + inp + out;
      } else {
        // Rough fallback estimate: ~4 chars per token
        const charCount = messages.reduce((s, m) => s + (m.content?.length ?? 0), 0);
        totalTokens = Math.floor(charCount / 4);
      }

      // Auto-compact when context exceeds threshold
      if (totalTokens > TOKEN_COMPACT_THRESHOLD && currentSessionId) {
        await compactSession();
      }

      messages = messages.map(m =>
        m.id === assistantId
          ? {
              ...m,
              content: result.reply ?? '',
              isStreaming: false,
              messageType: 'text',
              metadata: result,
              toolCalls: Array.isArray(result.tool_calls) ? result.tool_calls : undefined
            }
          : m
      );
    } catch (e) {
      console.error('Chat error:', e);
      messages = messages.map(m =>
        m.id === assistantId
          ? { ...m, content: 'Sorry, I encountered an error. Please try again.', isStreaming: false }
          : m
      );
    } finally {
      isLoading = false;
      await tick();
      messagesEnd?.scrollIntoView({ behavior: 'smooth' });
    }
  }

  // ── Slash commands ───────────────────────────────────────────────────────

  const SLASH_COMMANDS: SlashCommand[] = [
    {
      name: '/clear',
      description: 'Clear all messages and reset context',
      action: () => {
        messages = [];
        totalTokens = 0;
        currentSessionId = null;
        inputMessage = '';
        slashMenuOpen = false;
      }
    },
    {
      name: '/compact',
      description: 'Compact context — trim to last 10 messages',
      action: async () => {
        inputMessage = '';
        slashMenuOpen = false;
        if (currentSessionId) {
          try {
            await fetch(`${API_BASE}/chat/sessions/${currentSessionId}/compact`, { method: 'POST' });
          } catch { /* endpoint may not exist */ }
        }
        if (messages.length > 10) {
          const summary: Message = {
            id: crypto.randomUUID(),
            role: 'assistant',
            content: `[Context compacted — showing last ${Math.min(messages.length, 10)} messages]`,
            timestamp: new Date()
          };
          messages = [summary, ...messages.slice(-10)];
        }
        totalTokens = 0;
      }
    },
    {
      name: '/model',
      description: 'Open model selector',
      action: () => {
        inputMessage = '';
        slashMenuOpen = false;
        modelDropdownOpen = true;
      }
    },
    {
      name: '/help',
      description: 'Show available slash commands',
      action: () => {
        inputMessage = '';
        slashMenuOpen = false;
        const helpLines = SLASH_COMMANDS.map(c => `${c.name} — ${c.description}`).join('\n');
        messages = [...messages, {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: `Available commands:\n${helpLines}`,
          timestamp: new Date()
        }];
        activeSection = 'chat';
      }
    }
  ];

  const filteredSlashCommands = $derived(() => {
    if (!slashMenuFilter) return SLASH_COMMANDS;
    const q = slashMenuFilter.toLowerCase();
    return SLASH_COMMANDS.filter(c => c.name.toLowerCase().includes(q));
  });

  function handleInput() {
    const val = inputMessage;
    if (val.startsWith('/')) {
      const fragment = val.slice(1);
      // Only show menu if user is still typing a command (no space yet — means mid-command)
      if (!val.includes(' ')) {
        slashMenuFilter = fragment;
        slashMenuOpen = true;
        slashMenuIndex = 0;
      } else {
        slashMenuOpen = false;
      }
    } else {
      slashMenuOpen = false;
    }
  }

  function executeSlashCommand(cmd: SlashCommand) {
    cmd.action();
    slashMenuOpen = false;
  }

  function toggleToolCall(msgId: string, toolIndex: number) {
    const key = `${msgId}:${toolIndex}`;
    const next = new Set(expandedToolCalls);
    if (next.has(key)) {
      next.delete(key);
    } else {
      next.add(key);
    }
    expandedToolCalls = next;
  }

  function handleKeyDown(e: KeyboardEvent) {
    // Slash menu navigation
    if (slashMenuOpen) {
      const cmds = filteredSlashCommands();
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        slashMenuIndex = (slashMenuIndex + 1) % Math.max(cmds.length, 1);
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        slashMenuIndex = (slashMenuIndex - 1 + Math.max(cmds.length, 1)) % Math.max(cmds.length, 1);
        return;
      }
      if (e.key === 'Enter') {
        e.preventDefault();
        if (cmds[slashMenuIndex]) {
          executeSlashCommand(cmds[slashMenuIndex]);
        }
        return;
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        slashMenuOpen = false;
        return;
      }
    }

    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  function selectModelOption(providerId: string, modelId: string) {
    selectedProvider = providerId;
    selectedModel = modelId;
    modelDropdownOpen = false;
  }

  function closeModelDropdown(e: MouseEvent) {
    const target = e.target as Element;
    if (!target.closest('.model-selector-wrap')) {
      modelDropdownOpen = false;
      spawnDropdownOpen = null;
    }
  }

  function invokeSkill(skill: Skill) {
    selectedSkill = skill;
    queuedSkillCommand = (skill.slash_command || `/${skill.name}`) + ' ';
    inputMessage = queuedSkillCommand;
  }

  function toggleNodeExpansion(nodeId: string) {
    expandedNodeId = expandedNodeId === nodeId ? null : nodeId;
  }
</script>

<svelte:window onclick={closeModelDropdown} />

<div class="workshop-canvas" data-dept="workshop">
  <!-- ── Left Sidebar ─────────────────────────────────────────────────────── -->
  <aside class="workshop-sidebar">
    <div class="sidebar-top">
      <button class="new-chat-btn" onclick={startNewChat}>
        <Plus size={15} />
        <span>New Chat</span>
      </button>
    </div>

    <div class="sidebar-divider"></div>

    <nav class="sidebar-nav">
      <!-- RECENT -->
      <div class="nav-section-label-row">
        <div class="nav-section-label">Recent</div>
        {#if sessions.length > 0}
          <button
            class="clear-history-btn"
            onclick={deleteAllSessions}
            disabled={sessionActionLoading}
            title="Delete all recent workshop sessions"
          >
            Clear history
          </button>
        {/if}
      </div>
      {#if sessionsLoading}
        <div class="nav-loading"><Loader size={12} class="spin" /></div>
      {:else if sessions.length === 0}
        <div class="nav-empty">
          No recent chats
        </div>
      {:else}
        {#each sessions.slice(0, 3) as session}
          <div
            class="nav-item session-row"
            class:active={currentSessionId === session.id}
            onclick={() => selectSession(session)}
            onkeypress={(e) => e.key === 'Enter' && selectSession(session)}
            role="button"
            tabindex="0"
          >
            <MessageSquare size={15} />
            {#if renamingSessionId === session.id}
              <input
                class="session-title-input"
                bind:value={renameSessionTitle}
                onclick={(e) => e.stopPropagation()}
                onkeydown={(e) => {
                  if (e.key === 'Enter') {
                    void commitRenameSession(session.id, e);
                  } else if (e.key === 'Escape') {
                    cancelRenameSession(e);
                  }
                }}
              />
              <button
                class="session-action-btn"
                onclick={(e) => commitRenameSession(session.id, e)}
                title="Save title"
              >
                <Check size={11} />
              </button>
              <button
                class="session-action-btn"
                onclick={(e) => cancelRenameSession(e)}
                title="Cancel rename"
              >
                <X size={11} />
              </button>
            {:else}
              <span class="nav-label">{session.title || 'Untitled'}</span>
              <button
                class="session-action-btn"
                onclick={(e) => startRenameSession(session, e)}
                title="Rename"
              >
                <Pencil size={11} />
              </button>
              <button
                class="session-action-btn"
                onclick={(e) => deleteSession(session.id, e)}
                title="Delete"
              >
                <Trash2 size={11} />
              </button>
            {/if}
          </div>
        {/each}
      {/if}

      <div class="nav-spacer"></div>

      <!-- SHARED ASSETS -->
      <div class="nav-section-label">Assets</div>
      <button
        class="nav-item"
        class:active={activeSection === 'projects'}
        onclick={() => navigateTo('projects')}
      >
        <FolderOpen size={15} />
        <span class="nav-label">Shared Assets</span>
        <ChevronRight size={12} class="nav-arrow" />
      </button>

      <!-- MEMORY -->
      <div class="nav-section-label">Memory</div>
      <button
        class="nav-item"
        class:active={activeSection === 'memory'}
        onclick={() => navigateTo('memory')}
      >
        <Brain size={15} />
        <span class="nav-label">Graph Memory</span>
        <ChevronRight size={12} class="nav-arrow" />
      </button>

      <!-- SKILLS -->
      <div class="nav-section-label">Skills</div>
      <button
        class="nav-item"
        class:active={activeSection === 'skills'}
        onclick={() => navigateTo('skills')}
      >
        <Zap size={15} />
        <span class="nav-label">Skills Catalogue</span>
        <ChevronRight size={12} class="nav-arrow" />
      </button>

      <!-- WORKFLOWS -->
      <div class="nav-section-label">Workflows</div>
      <button
        class="nav-item"
        class:active={activeSection === 'workflows'}
        onclick={() => navigateTo('workflows')}
      >
        <GitBranch size={15} />
        <span class="nav-label">My Workflows</span>
        <ChevronRight size={12} class="nav-arrow" />
      </button>

      <!-- SUB-AGENTS -->
      <div class="nav-section-label">Agents</div>
      <button
        class="nav-item"
        class:active={activeSection === 'subagents'}
        onclick={() => navigateTo('subagents')}
      >
        <Bot size={15} />
        <span class="nav-label">Sub-agents</span>
        <ChevronRight size={12} class="nav-arrow" />
      </button>

          </nav>

  </aside>

  <!-- ── Main Content Area ───────────────────────────────────────────────── -->
  <main class="workshop-main">

    <!-- CHAT VIEW -->
    {#if activeSection === 'chat'}
      <div class="chat-view">
        <div class="messages-area">
          {#if messages.length === 0}
            <div class="welcome-state">
              <h1 class="greeting-text">
                {greeting()},&nbsp;<span class="greeting-name">Mubarak</span>
              </h1>
              <p class="greeting-sub">{currentTime()} — System nominal</p>
              <div class="suggestion-chips">
                {#each SUGGESTION_CHIPS as chip}
                  <button class="chip" onclick={() => sendMessage(chip)}>
                    {chip}
                  </button>
                {/each}
              </div>
            </div>
          {:else}
            <div class="messages-list">
              {#each messages as msg (msg.id)}
                <div class="message" class:user={msg.role === 'user'}>
                  <div class="msg-avatar">
                    {#if msg.role === 'user'}
                      <User size={16} />
                    {:else}
                      <Bot size={16} />
                    {/if}
                  </div>
                  <div class="msg-body" class:streaming={msg.isStreaming}>
                    {#if msg.messageType === 'audit_timeline' || msg.messageType === 'audit_reasoning'}
                      <div class="audit-block">
                        {@html renderMarkdown(msg.content)}
                      </div>
                    {:else if msg.role === 'user'}
                      {msg.content}
                    {:else}
                      {@html renderMarkdown(msg.content)}
                    {/if}

                    {#if msg.toolCalls && msg.toolCalls.length > 0}
                      <div class="tool-calls">
                        {#each msg.toolCalls as tc, i}
                          {@const tileKey = `${msg.id}:${i}`}
                          {@const isExpanded = expandedToolCalls.has(tileKey)}
                          <button
                            class="tool-call-tile"
                            onclick={() => toggleToolCall(msg.id, i)}
                          >
                            <div class="tool-call-header">
                              <Wrench size={12} class="tool-icon" />
                              <span class="tool-call-name">{tc.name}</span>
                              <span class="tool-call-chevron" class:rotated={isExpanded}>▾</span>
                            </div>
                            {#if isExpanded}
                              <div class="tool-call-body">
                                <div class="tool-call-section-label">Input</div>
                                <pre class="tool-call-json">{JSON.stringify(tc.input, null, 2)}</pre>
                                {#if tc.result}
                                  <div class="tool-call-section-label">Result</div>
                                  <pre class="tool-call-json tool-call-result">{tc.result}</pre>
                                {/if}
                              </div>
                            {/if}
                          </button>
                        {/each}
                      </div>
                    {/if}

                    {#if msg.isStreaming}
                      <span class="cursor">▊</span>
                    {/if}
                  </div>
                </div>
              {/each}
              <div bind:this={messagesEnd}></div>
            </div>
          {/if}
        </div>

        <!-- Input bar — centered, max 600px -->
        <div class="input-area">

          <!-- Slash command menu -->
          {#if slashMenuOpen && filteredSlashCommands().length > 0}
            <div class="slash-menu">
              {#each filteredSlashCommands() as cmd, i}
                <button
                  class="slash-menu-item"
                  class:active={slashMenuIndex === i}
                  onclick={() => executeSlashCommand(cmd)}
                >
                  <span class="slash-cmd-name">{cmd.name}</span>
                  <span class="slash-cmd-desc">{cmd.description}</span>
                </button>
              {/each}
            </div>
          {/if}

          {#if showAttachMenu}
            <div class="attach-menu">
              <div class="attach-header">Attach context from canvas</div>
              {#each ATTACH_CANVAS_OPTIONS as option}
                <button class="attach-option" onclick={() => attachCanvasContext(option)}>
                  <span class="attach-label">{option.label}</span>
                </button>
              {/each}
              {#if attachableResources.length > 0}
                <div class="attach-header attach-subheader">Attach a visible file or tile</div>
                {#each attachableResources as resource}
                  <button class="attach-option" onclick={() => attachResource(resource)}>
                    <span class="attach-label">{resource.label}</span>
                    <span class="attach-meta">{resource.resource_type}</span>
                  </button>
                {/each}
              {/if}
            </div>
          {/if}

          {#if attachments.length > 0}
            <div class="attachment-pills">
              {#each attachments as attachment (attachment.id)}
                <span class="attachment-pill">
                  {attachment.label}
                  <button class="attachment-remove" onclick={() => removeAttachment(attachment.id)}>×</button>
                </span>
              {/each}
            </div>
          {/if}

          <div class="input-bar">
            <textarea
              bind:value={inputMessage}
              onkeydown={handleKeyDown}
              oninput={handleInput}
              placeholder="Ask anything… type / for commands"
              rows="1"
              disabled={isLoading}
            ></textarea>
            <div class="input-actions">
              <button class="action-btn" title="Attach canvas context" disabled={isLoading} onclick={() => openAttachMenu()}>
                <Paperclip size={16} />
              </button>
              {#if availableProviders.length > 0}
                <div class="model-selector-wrap">
                  <button
                    class="model-selector-btn"
                    onclick={(e) => { e.stopPropagation(); modelDropdownOpen = !modelDropdownOpen; }}
                    title="Select model"
                  >
                    {selectedModelLabel()}
                    <ChevronRight size={10} style="transform: rotate(90deg); opacity: 0.6;" />
                  </button>
                  {#if modelDropdownOpen}
                    <div class="model-dropdown">
                      {#each availableProviders as provider}
                        {#if provider.models.length > 0}
                          <div class="model-dropdown-group">
                            <span class="model-dropdown-provider">{provider.display_name}</span>
                            {#each provider.models as model}
                              <button
                                class="model-dropdown-item"
                                class:active={selectedModel === model.id}
                                onclick={() => selectModelOption(provider.id, model.id)}
                              >
                                {model.name}
                              </button>
                            {/each}
                          </div>
                        {/if}
                      {/each}
                    </div>
                  {/if}
                </div>
              {/if}
              <button
                class="send-btn"
                onclick={() => sendMessage()}
                disabled={!inputMessage.trim() || isLoading}
                title="Send"
              >
                {#if isLoading}
                  <Loader size={16} class="spin" />
                {:else}
                  <Send size={16} />
                {/if}
              </button>
            </div>
          </div>
          <div class="input-footer">
            <span
              class="token-counter"
              class:token-warn={totalTokens > 150000 && totalTokens <= 190000}
              class:token-danger={totalTokens > 190000}
            >
              {#if totalTokens > 190000}
                Near limit — {totalTokens > 999 ? (totalTokens / 1000).toFixed(1) + 'k' : totalTokens} tokens
              {:else if totalTokens > 150000}
                ~{(totalTokens / 1000).toFixed(1)}k tokens — consider /compact
              {:else if totalTokens > 0}
                ~{totalTokens > 999 ? (totalTokens / 1000).toFixed(1) + 'k' : totalTokens} tokens
              {/if}
            </span>
            <p class="input-hint">Enter to send · Shift+Enter for new line · / for commands</p>
          </div>
        </div>
      </div>

    <!-- SHARED ASSETS VIEW -->
    {:else if activeSection === 'projects'}
      <div class="section-view">
        <div class="section-header">
          <FolderOpen size={20} />
          <h2>Shared Assets</h2>
        </div>
        {#if projectsLoading}
          <div class="loading-state"><Loader size={24} class="spin" /></div>
        {:else if projects.length === 0}
          <div class="empty-state">
            <FolderOpen size={40} />
            <p>No shared assets found.</p>
          </div>
        {:else}
          <div class="tile-grid">
            {#each projects as proj}
              <div class="tile">
                <div class="tile-title">{proj.name}</div>
                {#if proj.description}
                  <div class="tile-sub">{proj.description}</div>
                {/if}
                {#if proj.status}
                  <span class="tile-badge">{proj.status}</span>
                {/if}
              </div>
            {/each}
          </div>
        {/if}
      </div>

    <!-- MEMORY VIEW -->
    {:else if activeSection === 'memory'}
      <div class="section-view">
        <div class="section-header">
          <Brain size={20} />
          <h2>Graph Memory</h2>
        </div>

        <!-- Sub-tabs -->
        <div class="memory-tabs">
          <button
            class="memory-tab"
            class:active={memoryTab === 'nodes'}
            onclick={() => { memoryTab = 'nodes'; }}
          >Hot Nodes</button>
          <button
            class="memory-tab"
            class:active={memoryTab === 'opinions'}
            onclick={() => { memoryTab = 'opinions'; if (opinionNodes.length === 0) loadOpinionNodes(); }}
          >Opinion Nodes</button>
        </div>

        {#if memoryTab === 'nodes'}
          <!-- Hot / Warm nodes with filter pills -->
          <div class="filter-pills" style="margin-bottom:12px;">
            {#each (['all', 'hot', 'warm'] as const) as f}
              <button
                class="pill"
                class:active={memoryFilter === f}
                onclick={() => { memoryFilter = f; loadMemoryNodes(); }}
              >{f}</button>
            {/each}
          </div>
          {#if memoryLoading}
            <div class="loading-state"><Loader size={24} class="spin" /></div>
          {:else if memoryNodes.length === 0}
            <div class="empty-state">
              <Brain size={40} />
              <p>No memory nodes found.</p>
            </div>
          {:else}
            <div class="memory-list">
              {#each memoryNodes as node}
                <button class="memory-card" onclick={() => toggleNodeExpansion(node.id)}>
                  <div class="memory-card-header">
                    <span class="type-badge">{node.node_type}</span>
                    <span class="memory-preview">{node.content.substring(0, 60)}&hellip;</span>
                    <ChevronRight size={14} />
                  </div>
                  {#if expandedNodeId === node.id}
                    <div class="memory-detail">{node.content}</div>
                  {/if}
                </button>
              {/each}
            </div>
          {/if}

        {:else}
          <!-- Opinion nodes list -->
          {#if opinionsLoading}
            <div class="loading-state"><Loader size={24} class="spin" /></div>
          {:else if opinionNodes.length === 0}
            <div class="empty-state">
              <Brain size={36} />
              <p>No opinion nodes yet. Write one below.</p>
            </div>
          {:else}
            <div class="memory-list" style="margin-bottom:16px;">
              {#each opinionNodes as op}
                <div class="opinion-card">
                  <p class="opinion-content">{op.content}</p>
                  <div class="opinion-meta">
                    {#if op.confidence !== null && op.confidence !== undefined}
                      <span class="confidence-badge">
                        {Math.round(op.confidence * 100)}% confidence
                      </span>
                    {/if}
                    <span class="opinion-date">{new Date(op.created_at).toLocaleDateString()}</span>
                  </div>
                </div>
              {/each}
            </div>
          {/if}

          <!-- Write opinion form -->
          <div class="opinion-form">
            <div class="opinion-form-label">Write Opinion</div>
            <textarea
              class="opinion-textarea"
              placeholder="Record your market view, strategy opinion, or observation&hellip;"
              bind:value={opinionContent}
              rows={4}
            ></textarea>
            <div class="opinion-slider-row">
              <label class="slider-label">Confidence: {Math.round(opinionConfidence * 100)}%</label>
              <input
                type="range"
                min="0" max="1" step="0.05"
                bind:value={opinionConfidence}
                class="opinion-slider"
              />
            </div>
            <button
              class="save-opinion-btn"
              onclick={saveOpinion}
              disabled={opinionSaving || !opinionContent.trim()}
            >
              {#if opinionSaving}
                <Loader size={14} class="spin" />
                Saving&hellip;
              {:else}
                Save Opinion
              {/if}
            </button>
          </div>
        {/if}
      </div>

    <!-- SKILLS VIEW -->
    {:else if activeSection === 'skills'}
      <div class="section-view">
        <div class="section-header">
          <Zap size={20} />
          <h2>Skills Catalogue</h2>
        </div>
        {#if selectedSkill}
          <div class="skill-preview">
            <div class="skill-preview-header">
              <div>
                <div class="skill-preview-command">{queuedSkillCommand.trim()}</div>
                <div class="skill-preview-title">{selectedSkill.name}</div>
              </div>
              <button class="skill-preview-action" onclick={() => { activeSection = 'chat'; }}>
                Open Chat
              </button>
            </div>
            <p class="skill-preview-description">{selectedSkill.description}</p>
            <div class="skill-preview-meta">
              <span>Queued in chat draft</span>
              {#if selectedSkill.usage_count}
                <span>{selectedSkill.usage_count} uses</span>
              {/if}
            </div>
          </div>
        {/if}
        {#if skillsLoading}
          <div class="loading-state"><Loader size={24} class="spin" /></div>
        {:else if skills.length === 0}
          <div class="empty-state">
            <Zap size={40} />
            <p>No skills registered yet.</p>
          </div>
        {:else}
          <div class="skills-grid">
            {#each skills as skill}
              <button class="skill-card" onclick={() => invokeSkill(skill)}>
                <div class="skill-command">{skill.slash_command || `/${skill.name}`}</div>
                <div class="skill-name">{skill.name}</div>
                <div class="skill-desc">{skill.description}</div>
                {#if skill.usage_count}
                  <span class="skill-uses">{skill.usage_count} uses</span>
                {/if}
              </button>
            {/each}
          </div>
        {/if}
      </div>

    <!-- WORKFLOWS VIEW -->
    {:else if activeSection === 'workflows'}
      <div class="section-view">
        <div class="section-header">
          <GitBranch size={20} />
          <h2>My Workflows</h2>
        </div>
        {#if workflowsLoading}
          <div class="loading-state"><Loader size={24} class="spin" /></div>
        {:else if workflows.length === 0}
          <div class="empty-state">
            <GitBranch size={40} />
            <p>No workflows found.</p>
          </div>
        {:else}
          <div class="workflow-list">
            {#each workflows as wf}
              <div class="workflow-row">
                <GitBranch size={14} />
                <span class="wf-name">{wf.name}</span>
                <span class="wf-status" class:running={wf.status === 'running'}>{wf.status}</span>
                {#if wf.last_run}
                  <span class="wf-time">{wf.last_run}</span>
                {/if}
              </div>
            {/each}
          </div>
        {/if}
      </div>

    <!-- SUB-AGENTS VIEW -->
    {:else if activeSection === 'subagents'}
      <div class="section-view">
        <div class="section-header">
          <Bot size={20} />
          <h2>Sub-agents</h2>
          <button class="refresh-btn" onclick={loadRunningAgents} title="Refresh" disabled={agentsLoading}>
            <Loader size={14} class={agentsLoading ? 'spin' : ''} />
          </button>
        </div>

        <!-- Spawn panel -->
        <div class="subagent-spawn-panel">
          <div class="spawn-row">
            <select class="spawn-select" bind:value={spawnAgentType}>
              {#each SUBAGENT_TYPES as t}
                <option value={t.id}>{t.label}</option>
              {/each}
            </select>
            {#if availableProviders.length > 0}
              <div class="model-selector-wrap">
                <button
                  class="model-selector-btn"
                  onclick={(e) => { e.stopPropagation(); spawnDropdownOpen = spawnDropdownOpen ? null : 'spawn'; }}
                  title="Select model for sub-agent"
                >
                  {spawnModel ? (availableProviders.flatMap(p => p.models).find(m => m.id === spawnModel)?.name ?? spawnModel) : 'Default model'}
                  <ChevronRight size={10} style="transform: rotate(90deg); opacity: 0.6;" />
                </button>
                {#if spawnDropdownOpen === 'spawn'}
                  <div class="model-dropdown">
                    <button
                      class="model-dropdown-item"
                      class:active={!spawnModel}
                      onclick={() => { spawnModel = ''; spawnProvider = ''; spawnDropdownOpen = null; }}
                    >Default model</button>
                    {#each availableProviders as provider}
                      {#if provider.models.length > 0}
                        <div class="model-dropdown-group">
                          <span class="model-dropdown-provider">{provider.display_name}</span>
                          {#each provider.models as model}
                            <button
                              class="model-dropdown-item"
                              class:active={spawnModel === model.id}
                              onclick={() => { spawnModel = model.id; spawnProvider = provider.id; spawnDropdownOpen = null; }}
                            >{model.name}</button>
                          {/each}
                        </div>
                      {/if}
                    {/each}
                  </div>
                {/if}
              </div>
            {/if}
          </div>
          <div class="spawn-task-row">
            <input
              class="spawn-task-input"
              type="text"
              bind:value={spawnTask}
              placeholder="Task description…"
              onkeydown={(e) => e.key === 'Enter' && spawnAgent(spawnAgentType, spawnTask, spawnModel, spawnProvider)}
            />
            <button
              class="spawn-btn"
              onclick={() => spawnAgent(spawnAgentType, spawnTask, spawnModel, spawnProvider)}
              disabled={!spawnTask.trim() || spawning}
            >
              {spawning ? 'Spawning…' : 'Spawn'}
            </button>
          </div>
        </div>

        <!-- Running agents list -->
        {#if agentsLoading}
          <div class="loading-state"><Loader size={24} class="spin" /></div>
        {:else if runningAgents.length === 0}
          <div class="empty-state">
            <Bot size={40} />
            <p>No sub-agents running.</p>
          </div>
        {:else}
          <div class="agent-list">
            {#each runningAgents as agent}
              <div class="agent-row" class:completed={agent.status === 'completed'} class:failed={agent.status === 'failed'}>
                <Bot size={13} />
                <span class="agent-type">{agent.type}</span>
                <span class="agent-task">{agent.task.slice(0, 60)}{agent.task.length > 60 ? '…' : ''}</span>
                <span class="agent-status" class:running={agent.status === 'running'}>{agent.status}</span>
              </div>
            {/each}
          </div>
        {/if}
      </div>

    {/if}

  </main>
</div>

<style>
  /* ── Shell layout ───────────────────────────────────────────────────────── */
  .workshop-canvas {
    display: flex;
    height: 100%;
    width: 100%;
    background: transparent;
    overflow: hidden;
  }

  /* ── Left Sidebar ───────────────────────────────────────────────────────── */
  .workshop-sidebar {
    width: 200px;
    min-width: 200px;
    height: 100%;
    display: flex;
    flex-direction: column;
    background: rgba(8, 13, 20, 0.6);
    backdrop-filter: blur(24px);
    border-right: 1px solid rgba(0, 212, 255, 0.08);
    overflow: hidden;
  }

  .sidebar-top {
    padding: 14px 12px 12px;
  }

  .new-chat-btn {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 10px 14px;
    background: rgba(0, 212, 255, 0.12);
    border: 1px solid rgba(0, 212, 255, 0.25);
    border-radius: 8px;
    color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
  }

  .new-chat-btn:hover {
    background: rgba(0, 212, 255, 0.2);
    border-color: rgba(0, 212, 255, 0.4);
  }

  .sidebar-divider {
    height: 1px;
    margin: 0 12px;
    background: rgba(0, 212, 255, 0.08);
  }

  .sidebar-nav {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 10px 8px 6px;
    overflow-y: auto;
    gap: 1px;
  }

  .sidebar-nav::-webkit-scrollbar { width: 4px; }
  .sidebar-nav::-webkit-scrollbar-track { background: transparent; }
  .sidebar-nav::-webkit-scrollbar-thumb {
    background: rgba(0, 212, 255, 0.15);
    border-radius: 2px;
  }

  .nav-section-label {
    padding: 10px 8px 3px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: rgba(100, 116, 139, 0.7);
    user-select: none;
  }

  .nav-section-label-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }

  .clear-history-btn {
    padding: 4px 8px;
    background: transparent;
    border: 1px solid rgba(239, 68, 68, 0.18);
    border-radius: 999px;
    color: rgba(248, 113, 113, 0.9);
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    cursor: pointer;
    transition: background 0.12s ease, border-color 0.12s ease, color 0.12s ease;
    margin-right: 8px;
  }

  .clear-history-btn:hover:not(:disabled) {
    background: rgba(239, 68, 68, 0.08);
    border-color: rgba(248, 113, 113, 0.32);
    color: #fca5a5;
  }

  .clear-history-btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .nav-spacer { height: 6px; }

  .nav-item {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 7px 10px;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    cursor: pointer;
    transition: background 0.12s, color 0.12s;
    text-align: left;
    width: 100%;
  }

  .nav-item:hover {
    background: rgba(0, 212, 255, 0.07);
    color: #94a3b8;
  }

  .nav-item.active {
    background: rgba(0, 212, 255, 0.13);
    color: #00d4ff;
  }

  .nav-label {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  :global(.nav-arrow) {
    opacity: 0.35;
    flex-shrink: 0;
  }

  .session-row { cursor: pointer; }

  .session-action-btn {
    padding: 2px;
    background: transparent;
    border: none;
    color: #64748b;
    cursor: pointer;
    transition: color 0.12s;
    flex-shrink: 0;
  }

  .session-action-btn:hover { color: #cbd5e1; }
  .session-row:hover .session-action-btn:last-child { color: #f87171; }

  .session-title-input {
    flex: 1;
    min-width: 0;
    border: 1px solid rgba(148, 163, 184, 0.28);
    background: rgba(15, 23, 42, 0.8);
    color: #e2e8f0;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    padding: 2px 6px;
    height: 22px;
  }

  .nav-loading {
    padding: 6px 12px;
    color: #475569;
    font-size: 11px;
  }

  .nav-empty {
    padding: 6px 12px;
    color: #475569;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
  }


  /* ── Main content ───────────────────────────────────────────────────────── */
  .workshop-main {
    flex: 1;
    display: flex;
    flex-direction: column;
    height: 100%;
    min-width: 0;
    background: transparent;
  }

  /* ── Chat view ──────────────────────────────────────────────────────────── */
  .chat-view {
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  .messages-area {
    flex: 1;
    overflow-y: auto;
    padding: 0 24px 16px;
    display: flex;
    flex-direction: column;
  }

  .messages-area::-webkit-scrollbar { width: 5px; }
  .messages-area::-webkit-scrollbar-track { background: transparent; }
  .messages-area::-webkit-scrollbar-thumb {
    background: rgba(0, 212, 255, 0.12);
    border-radius: 3px;
  }

  .welcome-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 64px 32px 32px;
  }

  .greeting-text {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(28px, 4vw, 40px);
    color: #e2e8f0;
    margin: 0 0 10px;
    line-height: 1.15;
  }

  .greeting-name { color: #00d4ff; }

  .greeting-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #475569;
    margin: 0 0 40px;
  }

  .suggestion-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: center;
    max-width: 600px;
  }

  .chip {
    padding: 9px 16px;
    background: rgba(0, 212, 255, 0.07);
    border: 1px solid rgba(0, 212, 255, 0.14);
    border-radius: 20px;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s, color 0.15s;
    white-space: nowrap;
  }

  .chip:hover {
    background: rgba(0, 212, 255, 0.14);
    border-color: rgba(0, 212, 255, 0.3);
    color: #00d4ff;
  }

  .messages-list {
    display: flex;
    flex-direction: column;
    gap: 20px;
    padding-top: 28px;
    max-width: 760px;
    width: 100%;
    margin: 0 auto;
  }

  .message {
    display: flex;
    gap: 14px;
    align-items: flex-start;
  }

  .message.user {
    flex-direction: row-reverse;
    margin-left: auto;
  }

  .msg-avatar {
    width: 32px;
    height: 32px;
    min-width: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.18);
    border-radius: 50%;
    color: #00d4ff;
    flex-shrink: 0;
  }

  .message.user .msg-avatar {
    background: rgba(0, 200, 150, 0.1);
    border-color: rgba(0, 200, 150, 0.18);
    color: #00c896;
  }

  .msg-body {
    padding: 10px 15px;
    background: rgba(17, 24, 39, 0.55);
    border: 1px solid rgba(0, 212, 255, 0.1);
    border-radius: 12px;
    color: #e2e8f0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    line-height: 1.65;
    max-width: 640px;
  }

  .message.user .msg-body {
    background: rgba(0, 212, 255, 0.07);
    border-color: rgba(0, 212, 255, 0.14);
  }

  .msg-body.streaming { border-color: rgba(0, 212, 255, 0.28); }

  .cursor {
    color: #00d4ff;
    animation: blink 1s step-end infinite;
  }

  @keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
  }

  .audit-block {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    line-height: 1.6;
    color: #94a3b8;
    padding: 8px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 6px;
    border-left: 2px solid #00d4ff;
  }

  .input-area {
    position: relative;
    padding: 12px 24px 20px;
    background: linear-gradient(to top, rgba(8, 13, 20, 0.8) 60%, transparent);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
  }

  .input-bar {
    display: flex;
    align-items: flex-end;
    gap: 10px;
    width: 100%;
    max-width: 600px;
    padding: 10px 12px;
    background: rgba(17, 24, 39, 0.75);
    border: 1px solid rgba(0, 212, 255, 0.14);
    border-radius: 14px;
    transition: border-color 0.15s, box-shadow 0.15s;
  }

  .input-bar:focus-within {
    border-color: rgba(0, 212, 255, 0.32);
    box-shadow: 0 0 22px rgba(0, 212, 255, 0.08);
  }

  .input-bar textarea {
    flex: 1;
    background: transparent;
    border: none;
    outline: none;
    color: #e2e8f0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    resize: none;
    min-height: 22px;
    max-height: 140px;
    line-height: 1.5;
  }

  .input-bar textarea::placeholder { color: #475569; }
  .input-bar textarea:disabled { opacity: 0.45; }

  .input-actions {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
  }

  .action-btn {
    width: 30px;
    height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: #475569;
    cursor: pointer;
    transition: color 0.12s, background 0.12s;
  }

  .action-btn:hover:not(:disabled) {
    color: #94a3b8;
    background: rgba(0, 212, 255, 0.07);
  }

  .action-btn:disabled { opacity: 0.3; cursor: not-allowed; }

  .attach-menu {
    width: 100%;
    max-width: 600px;
    max-height: 280px;
    overflow: auto;
    background: rgba(12, 18, 28, 0.96);
    border: 1px solid rgba(0, 212, 255, 0.16);
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
    backdrop-filter: blur(16px);
  }

  .attach-header {
    padding: 8px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-bottom: 1px solid rgba(0, 212, 255, 0.08);
  }

  .attach-subheader {
    border-top: 1px solid rgba(0, 212, 255, 0.08);
  }

  .attach-option {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 10px 12px;
    background: transparent;
    border: none;
    color: #e2e8f0;
    text-align: left;
    cursor: pointer;
    transition: background 0.12s ease;
  }

  .attach-option:hover {
    background: rgba(0, 212, 255, 0.06);
  }

  .attach-label {
    font-size: 12px;
    color: #cbd5e1;
  }

  .attach-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    color: #64748b;
  }

  .attachment-pills {
    width: 100%;
    max-width: 600px;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }

  .attachment-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    background: rgba(0, 212, 255, 0.08);
    border: 1px solid rgba(0, 212, 255, 0.16);
    border-radius: 999px;
    font-size: 11px;
    color: #cbd5e1;
  }

  .attachment-remove {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    border: none;
    border-radius: 50%;
    background: transparent;
    color: #94a3b8;
    cursor: pointer;
  }

  .attachment-remove:hover {
    background: rgba(255, 255, 255, 0.08);
    color: #e2e8f0;
  }

  .send-btn {
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 212, 255, 0.14);
    border: 1px solid rgba(0, 212, 255, 0.22);
    border-radius: 8px;
    color: #00d4ff;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
  }

  .send-btn:hover:not(:disabled) {
    background: rgba(0, 212, 255, 0.24);
    border-color: rgba(0, 212, 255, 0.42);
  }

  .send-btn:disabled { opacity: 0.35; cursor: not-allowed; }

  /* ── Model selector ─────────────────────────────────────────────────────── */
  .model-selector-wrap {
    position: relative;
  }

  .model-selector-btn {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    background: transparent;
    border: 1px solid rgba(0, 212, 255, 0.12);
    border-radius: 6px;
    color: #475569;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    cursor: pointer;
    white-space: nowrap;
    transition: color 0.12s, border-color 0.12s, background 0.12s;
    max-width: 120px;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .model-selector-btn:hover {
    color: #94a3b8;
    border-color: rgba(0, 212, 255, 0.25);
    background: rgba(0, 212, 255, 0.05);
  }

  .model-dropdown {
    position: absolute;
    bottom: calc(100% + 6px);
    right: 0;
    min-width: 180px;
    background: rgba(12, 18, 28, 0.97);
    border: 1px solid rgba(0, 212, 255, 0.18);
    border-radius: 8px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    z-index: 100;
    overflow: hidden;
    backdrop-filter: blur(16px);
  }

  .model-dropdown-group {
    padding: 4px 0;
    border-bottom: 1px solid rgba(0, 212, 255, 0.06);
  }

  .model-dropdown-group:last-child {
    border-bottom: none;
  }

  .model-dropdown-provider {
    display: block;
    padding: 6px 12px 2px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(0, 212, 255, 0.45);
    user-select: none;
  }

  .model-dropdown-item {
    display: block;
    width: 100%;
    text-align: left;
    padding: 6px 12px;
    background: transparent;
    border: none;
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: background 0.1s, color 0.1s;
  }

  .model-dropdown-item:hover {
    background: rgba(0, 212, 255, 0.07);
    color: #94a3b8;
  }

  .model-dropdown-item.active {
    color: #00d4ff;
    background: rgba(0, 212, 255, 0.1);
  }

  .input-hint {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #334155;
    margin: 0;
    flex: 1;
    text-align: center;
  }

  /* ── Section views ──────────────────────────────────────────────────────── */
  .section-view {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 28px 32px;
    overflow-y: auto;
    gap: 20px;
  }

  .section-view::-webkit-scrollbar { width: 5px; }
  .section-view::-webkit-scrollbar-track { background: transparent; }
  .section-view::-webkit-scrollbar-thumb {
    background: rgba(0, 212, 255, 0.12);
    border-radius: 3px;
  }

  .section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    color: #94a3b8;
  }

  .section-header h2 {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 20px;
    color: #e2e8f0;
    margin: 0;
    flex: 1;
  }

  .filter-pills {
    display: flex;
    gap: 6px;
    margin-left: auto;
  }

  .pill {
    padding: 4px 12px;
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(0, 212, 255, 0.1);
    border-radius: 12px;
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: background 0.12s, color 0.12s;
    text-transform: capitalize;
  }

  .pill:hover { color: #94a3b8; }

  .pill.active {
    background: rgba(0, 212, 255, 0.14);
    border-color: rgba(0, 212, 255, 0.28);
    color: #00d4ff;
  }

  .loading-state {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #475569;
  }

  .empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 14px;
    color: #334155;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    text-align: center;
  }

  /* Projects tile grid */
  .tile-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 14px;
  }

  .tile {
    padding: 16px;
    background: rgba(17, 24, 39, 0.5);
    border: 1px solid rgba(0, 212, 255, 0.1);
    border-radius: 10px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    transition: border-color 0.15s;
    cursor: default;
  }

  .tile:hover { border-color: rgba(0, 212, 255, 0.22); }

  .tile-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    color: #e2e8f0;
  }

  .tile-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #64748b;
  }

  .tile-badge {
    align-self: flex-start;
    padding: 2px 8px;
    background: rgba(0, 212, 255, 0.1);
    border-radius: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #00d4ff;
    text-transform: capitalize;
  }

  /* Memory */
  .memory-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .memory-card {
    padding: 12px;
    background: rgba(17, 24, 39, 0.45);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 8px;
    cursor: pointer;
    text-align: left;
    width: 100%;
    transition: border-color 0.12s;
  }

  .memory-card:hover { border-color: rgba(0, 212, 255, 0.2); }

  .memory-card-header {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .type-badge {
    padding: 2px 7px;
    background: rgba(0, 212, 255, 0.1);
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #00d4ff;
    text-transform: uppercase;
    flex-shrink: 0;
  }

  .memory-preview {
    flex: 1;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #64748b;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .memory-detail {
    margin-top: 10px;
    padding: 10px;
    background: rgba(0, 0, 0, 0.25);
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #94a3b8;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 180px;
    overflow-y: auto;
  }

  /* Memory sub-tabs */
  .memory-tabs {
    display: flex;
    gap: 4px;
    margin-bottom: 14px;
  }

  .memory-tab {
    padding: 5px 14px;
    background: rgba(17, 24, 39, 0.4);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 6px;
    font-size: 12px;
    color: #64748b;
    cursor: pointer;
    transition: border-color 0.12s, color 0.12s;
  }

  .memory-tab.active {
    border-color: rgba(0, 212, 255, 0.35);
    color: #00d4ff;
    background: rgba(0, 212, 255, 0.07);
  }

  /* Opinion cards */
  .opinion-card {
    padding: 12px;
    background: rgba(17, 24, 39, 0.45);
    border: 1px solid rgba(139, 92, 246, 0.15);
    border-radius: 8px;
  }

  .opinion-content {
    font-size: 13px;
    color: #cbd5e1;
    line-height: 1.5;
    margin: 0 0 8px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .opinion-meta {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .confidence-badge {
    padding: 2px 7px;
    background: rgba(139, 92, 246, 0.15);
    border-radius: 4px;
    font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
    color: #a78bfa;
  }

  .opinion-date {
    font-size: 11px;
    color: #475569;
    font-family: 'JetBrains Mono', monospace;
  }

  /* Opinion form */
  .opinion-form {
    padding: 14px;
    background: rgba(17, 24, 39, 0.5);
    border: 1px solid rgba(139, 92, 246, 0.12);
    border-radius: 10px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .opinion-form-label {
    font-size: 11px;
    font-weight: 600;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .opinion-textarea {
    width: 100%;
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(139, 92, 246, 0.15);
    border-radius: 6px;
    color: #e2e8f0;
    font-size: 13px;
    padding: 10px;
    resize: vertical;
    font-family: inherit;
    line-height: 1.5;
    box-sizing: border-box;
  }

  .opinion-textarea:focus {
    outline: none;
    border-color: rgba(139, 92, 246, 0.4);
  }

  .opinion-slider-row {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .slider-label {
    font-size: 12px;
    color: #94a3b8;
    white-space: nowrap;
    min-width: 140px;
    font-family: 'JetBrains Mono', monospace;
  }

  .opinion-slider {
    flex: 1;
    accent-color: #a78bfa;
  }

  .save-opinion-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 8px 18px;
    background: rgba(139, 92, 246, 0.2);
    border: 1px solid rgba(139, 92, 246, 0.35);
    border-radius: 6px;
    color: #a78bfa;
    font-size: 13px;
    cursor: pointer;
    transition: background 0.12s;
    align-self: flex-end;
  }

  .save-opinion-btn:hover:not(:disabled) {
    background: rgba(139, 92, 246, 0.3);
  }

  .save-opinion-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  /* Skills grid */
  .skills-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 12px;
  }

  .skill-preview {
    padding: 16px 18px;
    background: rgba(17, 24, 39, 0.55);
    border: 1px solid rgba(0, 212, 255, 0.14);
    border-radius: 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .skill-preview-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
  }

  .skill-preview-command {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #00d4ff;
  }

  .skill-preview-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 15px;
    color: #e2e8f0;
    margin-top: 4px;
  }

  .skill-preview-description {
    margin: 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    line-height: 1.55;
    color: #94a3b8;
  }

  .skill-preview-meta {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .skill-preview-action {
    padding: 7px 12px;
    background: rgba(0, 212, 255, 0.12);
    border: 1px solid rgba(0, 212, 255, 0.22);
    border-radius: 8px;
    color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    white-space: nowrap;
    transition: background 0.12s ease, border-color 0.12s ease;
  }

  .skill-preview-action:hover {
    background: rgba(0, 212, 255, 0.18);
    border-color: rgba(0, 212, 255, 0.34);
  }

  .skill-card {
    padding: 14px;
    background: rgba(17, 24, 39, 0.45);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 10px;
    cursor: pointer;
    text-align: left;
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 4px;
    transition: border-color 0.12s, background 0.12s;
  }

  .skill-card:hover {
    border-color: rgba(0, 212, 255, 0.22);
    background: rgba(0, 212, 255, 0.05);
  }

  .skill-command {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    color: #00d4ff;
  }

  .skill-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #e2e8f0;
  }

  .skill-desc {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #64748b;
    line-height: 1.4;
  }

  .skill-uses {
    align-self: flex-start;
    margin-top: 4px;
    padding: 2px 7px;
    background: rgba(0, 0, 0, 0.25);
    border-radius: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #475569;
  }

  /* Workflows */
  .workflow-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .workflow-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: rgba(17, 24, 39, 0.45);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 8px;
    color: #94a3b8;
  }

  .wf-name {
    flex: 1;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #e2e8f0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .wf-status {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #475569;
    text-transform: capitalize;
  }

  .wf-status.running { color: #00c896; }

  .wf-time {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #334155;
  }

  /* Spin animation */
  :global(.spin) {
    animation: spin 0.9s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  /* ── Sub-agents ─────────────────────────────────────────────────────────── */
  .subagent-spawn-panel {
    margin-bottom: 18px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .spawn-row {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }

  .spawn-select {
    background: rgba(17, 24, 39, 0.7);
    border: 1px solid rgba(0, 212, 255, 0.18);
    border-radius: 6px;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    padding: 6px 10px;
    cursor: pointer;
  }

  .spawn-task-row {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .spawn-task-input {
    flex: 1;
    background: rgba(17, 24, 39, 0.6);
    border: 1px solid rgba(0, 212, 255, 0.12);
    border-radius: 6px;
    color: #e2e8f0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    padding: 7px 12px;
    outline: none;
  }

  .spawn-task-input:focus {
    border-color: rgba(0, 212, 255, 0.35);
  }

  .spawn-btn {
    padding: 7px 16px;
    background: rgba(0, 212, 255, 0.12);
    border: 1px solid rgba(0, 212, 255, 0.25);
    border-radius: 6px;
    color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    cursor: pointer;
    white-space: nowrap;
  }

  .spawn-btn:hover:not(:disabled) {
    background: rgba(0, 212, 255, 0.2);
  }

  .spawn-btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .refresh-btn {
    margin-left: auto;
    background: transparent;
    border: none;
    color: #475569;
    cursor: pointer;
    padding: 4px;
    display: flex;
    align-items: center;
  }

  .refresh-btn:hover { color: #00d4ff; }

  .agent-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .agent-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: rgba(17, 24, 39, 0.45);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 8px;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }

  .agent-row.failed { border-color: rgba(239, 68, 68, 0.2); }

  .agent-type {
    color: #00d4ff;
    font-size: 11px;
    white-space: nowrap;
  }

  .agent-task {
    flex: 1;
    color: #e2e8f0;
    font-size: 12px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .agent-status {
    font-size: 11px;
    color: #475569;
    text-transform: capitalize;
  }

  .agent-status.running { color: #00c896; }

  /* ── Input footer (token counter + hint row) ────────────────────────────── */
  .input-footer {
    display: flex;
    align-items: center;
    width: 100%;
    max-width: 600px;
    gap: 10px;
  }

  .token-counter {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #334155;
    white-space: nowrap;
    min-width: 0;
  }

  .token-counter.token-warn {
    color: #f59e0b;
  }

  .token-counter.token-danger {
    color: #f87171;
  }

  /* ── Slash command menu ─────────────────────────────────────────────────── */
  .slash-menu {
    width: 100%;
    max-width: 600px;
    background: rgba(10, 16, 26, 0.97);
    border: 1px solid rgba(0, 212, 255, 0.18);
    border-radius: 10px;
    overflow: hidden;
    backdrop-filter: blur(20px);
    box-shadow: 0 -6px 24px rgba(0, 0, 0, 0.5);
    display: flex;
    flex-direction: column;
  }

  .slash-menu-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 9px 14px;
    background: transparent;
    border: none;
    cursor: pointer;
    text-align: left;
    transition: background 0.1s;
    width: 100%;
    border-bottom: 1px solid rgba(0, 212, 255, 0.05);
  }

  .slash-menu-item:last-child {
    border-bottom: none;
  }

  .slash-menu-item:hover,
  .slash-menu-item.active {
    background: rgba(0, 212, 255, 0.08);
  }

  .slash-cmd-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    color: #00d4ff;
    min-width: 80px;
    flex-shrink: 0;
  }

  .slash-cmd-desc {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #475569;
  }

  /* ── Tool call tiles ────────────────────────────────────────────────────── */
  .tool-calls {
    display: flex;
    flex-direction: column;
    gap: 5px;
    margin-top: 10px;
  }

  .tool-call-tile {
    display: flex;
    flex-direction: column;
    background: rgba(5, 10, 18, 0.65);
    border: none;
    border-radius: 7px;
    cursor: pointer;
    text-align: left;
    width: 100%;
    padding: 0;
    transition: background 0.12s;
    overflow: hidden;
  }

  .tool-call-tile:hover {
    background: rgba(5, 10, 18, 0.85);
  }

  .tool-call-header {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 6px 10px;
  }

  :global(.tool-icon) {
    color: #475569;
    flex-shrink: 0;
  }

  .tool-call-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    color: #64748b;
    flex: 1;
  }

  .tool-call-chevron {
    font-size: 13px;
    color: #334155;
    transition: transform 0.15s;
    line-height: 1;
  }

  .tool-call-chevron.rotated {
    transform: rotate(180deg);
  }

  .tool-call-body {
    padding: 0 10px 10px;
    display: flex;
    flex-direction: column;
    gap: 5px;
  }

  .tool-call-section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #334155;
    margin-top: 4px;
  }

  .tool-call-json {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #94a3b8;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 5px;
    padding: 8px 10px;
    margin: 0;
    white-space: pre-wrap;
    word-break: break-all;
    max-height: 200px;
    overflow-y: auto;
  }

  .tool-call-result {
    color: #6ee7b7;
  }
</style>

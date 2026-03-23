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
    Settings,
    Send,
    Paperclip,
    ChevronRight,
    Bot,
    Loader,
    User,
    Trash2
  } from 'lucide-svelte';
  import { chatApi, type ChatSession } from '$lib/api/chatApi';
  import { listSkills, type Skill } from '$lib/api/skillsApi';
  import { getHotNodes, getWarmNodes, type GraphMemoryNode } from '$lib/api/graphMemory';
  import { canvasContextService } from '$lib/services/canvasContextService';
  import { copilotKillSwitchService } from '$lib/services/copilotKillSwitchService';
  import { API_CONFIG } from '$lib/config/api';
  import AgentThoughtsPanel from '$lib/components/AgentThoughtsPanel.svelte';

  // ── Types ────────────────────────────────────────────────────────────────────

  interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    isStreaming?: boolean;
    messageType?: 'text' | 'audit_timeline' | 'audit_reasoning' | 'morning_digest';
    metadata?: Record<string, unknown>;
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

  // ── State ────────────────────────────────────────────────────────────────────

  let activeSection = $state<'chat' | 'projects' | 'memory' | 'skills' | 'workflows' | 'agent-thoughts'>('chat');

  // Chat
  let messages = $state<Message[]>([]);
  let inputMessage = $state('');
  let isLoading = $state(false);
  let currentSessionId = $state<string | null>(null);
  let messagesEnd = $state<HTMLElement | null>(null);

  // Sessions (recent history in sidebar)
  let sessions = $state<ChatSession[]>([]);
  let sessionsLoading = $state(false);

  // Skills
  let skills = $state<Skill[]>([]);
  let skillsLoading = $state(false);

  // Memory
  let memoryNodes = $state<GraphMemoryNode[]>([]);
  let memoryLoading = $state(false);
  let memoryFilter = $state<'all' | 'hot' | 'warm'>('all');
  let expandedNodeId = $state<string | null>(null);

  // Workflows
  let workflows = $state<WorkflowItem[]>([]);
  let workflowsLoading = $state(false);

  // Projects
  let projects = $state<ProjectItem[]>([]);
  let projectsLoading = $state(false);

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

  const API_BASE = API_CONFIG.API_BASE;

  const SUGGESTION_CHIPS = [
    'What happened overnight?',
    'Show pending approvals',
    'Research a new strategy',
    'Check active workflows',
    'Review department status'
  ];

  // ── Lifecycle ────────────────────────────────────────────────────────────────

  onMount(async () => {
    await Promise.all([loadSessions(), loadSkills()]);
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

  async function loadSessions() {
    sessionsLoading = true;
    try {
      const result = await chatApi.listSessions();
      sessions = result.slice(0, 20);
    } catch (e) {
      console.error('Failed to load sessions:', e);
    } finally {
      sessionsLoading = false;
    }
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

  async function loadWorkflows() {
    workflowsLoading = true;
    try {
      const res = await fetch(`${API_BASE}/prefect/workflows`);
      if (res.ok) {
        workflows = await res.json();
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
      const res = await fetch(`${API_BASE}/projects`);
      if (res.ok) {
        projects = await res.json();
      }
    } catch (e) {
      console.error('Failed to load projects:', e);
    } finally {
      projectsLoading = false;
    }
  }

  // ── Navigation ───────────────────────────────────────────────────────────────

  function navigateTo(section: typeof activeSection) {
    activeSection = section;
    if (section === 'memory' && memoryNodes.length === 0) {
      loadMemoryNodes();
    } else if (section === 'workflows' && workflows.length === 0) {
      loadWorkflows();
    } else if (section === 'projects' && projects.length === 0) {
      loadProjects();
    }
  }

  // ── Chat actions ─────────────────────────────────────────────────────────────

  function startNewChat() {
    messages = [];
    currentSessionId = null;
    activeSection = 'chat';
    inputMessage = '';
  }

  function selectSession(session: ChatSession) {
    currentSessionId = session.id;
    activeSection = 'chat';
    messages = [];
  }

  function deleteSession(sessionId: string, e: MouseEvent) {
    e.stopPropagation();
    sessions = sessions.filter(s => s.id !== sessionId);
  }

  async function sendMessage(content?: string) {
    const text = content ?? inputMessage;
    if (!text.trim() || isLoading) return;

    // Kill switch check
    const ks = await copilotKillSwitchService.getStatus();
    if (ks.active) {
      messages = [...messages, {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: `Copilot is currently disabled. ${ks.reason || 'Please try again later.'}`,
        timestamp: new Date()
      }];
      return;
    }

    activeSection = 'chat';

    messages = [...messages, {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      timestamp: new Date()
    }];
    inputMessage = '';
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
      const res = await fetch(`${API_BASE}/floor-manager/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          canvas: 'workshop',
          canvas_context: 'workshop',
          session_id: currentSessionId,
          stream: false
        })
      });

      const result = await res.json();

      messages = messages.map(m =>
        m.id === assistantId
          ? {
              ...m,
              content: result.content ?? result.reply ?? '',
              isStreaming: false,
              messageType: result.type ?? 'text',
              metadata: result
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

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  function invokeSkill(skill: Skill) {
    inputMessage = (skill.slash_command || `/${skill.name}`) + ' ';
    activeSection = 'chat';
  }

  function toggleNodeExpansion(nodeId: string) {
    expandedNodeId = expandedNodeId === nodeId ? null : nodeId;
  }
</script>

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
      <div class="nav-section-label">Recent</div>
      {#if sessionsLoading}
        <div class="nav-loading"><Loader size={12} class="spin" /></div>
      {:else if sessions.length === 0}
        <button
          class="nav-item"
          class:active={activeSection === 'chat'}
          onclick={() => navigateTo('chat')}
        >
          <MessageSquare size={15} />
          <span class="nav-label">Today's session</span>
        </button>
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
            <span class="nav-label">{session.title || 'Untitled'}</span>
            <button
              class="delete-btn"
              onclick={(e) => deleteSession(session.id, e)}
              title="Delete"
            >
              <Trash2 size={11} />
            </button>
          </div>
        {/each}
      {/if}

      <div class="nav-spacer"></div>

      <!-- PROJECTS -->
      <div class="nav-section-label">Projects</div>
      <button
        class="nav-item"
        class:active={activeSection === 'projects'}
        onclick={() => navigateTo('projects')}
      >
        <FolderOpen size={15} />
        <span class="nav-label">My Projects</span>
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

      <!-- AGENT THOUGHTS -->
      <div class="nav-section-label">Agents</div>
      <button
        class="nav-item"
        class:active={activeSection === 'agent-thoughts'}
        onclick={() => navigateTo('agent-thoughts')}
      >
        <Brain size={15} />
        <span class="nav-label">Agent Thoughts</span>
        <ChevronRight size={12} class="nav-arrow" />
      </button>
    </nav>

    <!-- Settings at bottom -->
    <div class="sidebar-bottom">
      <div class="sidebar-divider"></div>
      <button class="nav-item settings-item" onclick={() => {}}>
        <Settings size={15} />
        <span class="nav-label">Settings</span>
      </button>
    </div>
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
                        {@html msg.content.replace(/\n/g, '<br>')}
                      </div>
                    {:else}
                      {@html msg.content.replace(/\n/g, '<br>')}
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
          <div class="input-bar">
            <textarea
              bind:value={inputMessage}
              onkeydown={handleKeyDown}
              placeholder="Ask anything… type / for commands"
              rows="1"
              disabled={isLoading}
            ></textarea>
            <div class="input-actions">
              <button class="action-btn" title="Attach file" disabled={isLoading}>
                <Paperclip size={16} />
              </button>
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
          <p class="input-hint">Enter to send · Shift+Enter for new line · / for slash commands</p>
        </div>
      </div>

    <!-- PROJECTS VIEW -->
    {:else if activeSection === 'projects'}
      <div class="section-view">
        <div class="section-header">
          <FolderOpen size={20} />
          <h2>My Projects</h2>
        </div>
        {#if projectsLoading}
          <div class="loading-state"><Loader size={24} class="spin" /></div>
        {:else if projects.length === 0}
          <div class="empty-state">
            <FolderOpen size={40} />
            <p>No projects yet. Start a new chat to create a project.</p>
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
          <div class="filter-pills">
            {#each (['all', 'hot', 'warm'] as const) as f}
              <button
                class="pill"
                class:active={memoryFilter === f}
                onclick={() => { memoryFilter = f; loadMemoryNodes(); }}
              >{f}</button>
            {/each}
          </div>
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
                  <span class="memory-preview">{node.content.substring(0, 60)}…</span>
                  <ChevronRight size={14} />
                </div>
                {#if expandedNodeId === node.id}
                  <div class="memory-detail">{node.content}</div>
                {/if}
              </button>
            {/each}
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

    <!-- AGENT THOUGHTS VIEW -->
    {:else if activeSection === 'agent-thoughts'}
      <div class="section-view agent-thoughts-view">
        <AgentThoughtsPanel
          sessionId={currentSessionId ?? ''}
          maxHeight="100%"
          showHeader={true}
        />
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

  .delete-btn {
    opacity: 0;
    padding: 2px;
    background: transparent;
    border: none;
    color: #64748b;
    cursor: pointer;
    transition: opacity 0.12s, color 0.12s;
    flex-shrink: 0;
  }

  .session-row:hover .delete-btn { opacity: 1; }
  .delete-btn:hover { color: #f87171; }

  .nav-loading {
    padding: 6px 12px;
    color: #475569;
    font-size: 11px;
  }

  .sidebar-bottom { padding-bottom: 12px; }
  .sidebar-bottom .sidebar-divider { margin-bottom: 6px; }
  .settings-item { color: #475569; }
  .settings-item:hover { color: #94a3b8; }

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

  .input-hint {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #334155;
    margin: 0;
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

  /* Agent thoughts view — no extra padding, panel fills container */
  .agent-thoughts-view {
    padding: 16px;
    overflow: hidden;
  }

  .agent-thoughts-view > :global(.thoughts-panel) {
    height: 100%;
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

  /* Skills grid */
  .skills-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 12px;
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
</style>

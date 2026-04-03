<script lang="ts">
  import { createEventDispatcher, onMount, tick } from "svelte";
  import { fly, fade, slide } from "svelte/transition";
  import {
    Send,
    Loader,
    X,
    Bot,
    Plus,
    ChevronRight,
    ChevronDown,
    ArrowRightCircle,
    CheckCircle2,
    Clock,
    AlertCircle,
    Users,
    Building2,
    Mail,
    MailPlus,
    MessageSquarePlus,
    History,
    Trash2,
    Pencil,
    Check,
    MessageCircle,
    Power,
    Play,
  } from "lucide-svelte";
  import AgentModelSelector from "$lib/components/AgentModelSelector.svelte";
  import { chatApi, type ChatSession, type StoredChatMessage } from "$lib/api/chatApi";
  import {
    departmentChatStore,
    activeDelegatedTasks,
    DEPARTMENTS,
    type DepartmentId,
    type DepartmentInfo,
    type DelegatedTask,
  } from "$lib/stores/departmentChatStore";
  import { activeCanvasStore } from "$lib/stores/canvasStore";
  import { canvasContextService } from "$lib/services/canvasContextService";
  import { copilotKillSwitchService } from "$lib/services/copilotKillSwitchService";
  import { intentService, type CommandResponse, type PendingConfirmation } from "$lib/services/intentService";
  import { canvasContextStore } from "$lib/stores/canvas";
  import { serverHealthAlertEvent, getLastPersistedAlert, clearServerHealthAlert } from "$lib/stores/serverHealthAlerts";
  import { listAllAssets, type SharedAsset } from "$lib/api/sharedAssetsApi";
  import { API_CONFIG } from "$lib/config/api";
  import { marked } from 'marked';
  import DOMPurify from 'dompurify';

  // Render markdown safely — only for assistant messages
  function renderMarkdown(text: string): string {
    if (!text) return '';
    const html = marked.parse(text, { breaks: true, gfm: true }) as string;
    return DOMPurify.sanitize(html);
  }

  const dispatch = createEventDispatcher();

  // API base URL — use config to support remote servers
  const API_BASE = API_CONFIG.API_BASE;

  // Configurable conversation history limit (Story 5.5 - AC #4)
  const CONVERSATION_HISTORY_LIMIT = 10;

  // Chat history sidebar state
  let showChatHistory = $state(false);
  let sessions: ChatSession[] = $state([]);
  let currentSessionId: string | null = $state(null);
  let chatHistoryLoading = $state(false);
  let sessionActionLoading = $state(false);
  let renamingSessionId: string | null = $state(null);
  let renameSessionTitle = $state('');

  // Story 5.7: NL System Commands - pending confirmation state
  let pendingConfirmation: PendingConfirmation | null = $state(null);
  let confirmationLoading = $state(false);

  // Chat history grouped by time
  interface GroupedSessions {
    today: ChatSession[];
    yesterday: ChatSession[];
    last7Days: ChatSession[];
    last30Days: ChatSession[];
    older: ChatSession[];
  }

  let groupedSessions: GroupedSessions = $state({
    today: [],
    yesterday: [],
    last7Days: [],
    last30Days: [],
    older: [],
  });

  let sessionGroups = $derived([
    { label: 'Today', items: groupedSessions.today },
    { label: 'Yesterday', items: groupedSessions.yesterday },
    { label: 'Previous 7 Days', items: groupedSessions.last7Days },
    { label: 'Previous 30 Days', items: groupedSessions.last30Days },
    { label: 'Older', items: groupedSessions.older },
  ]);

  // Floor Manager state
  let message = $state("");
  let textareaElement: HTMLTextAreaElement = $state();
  let messagesContainer: HTMLDivElement = $state();
  let settingsOpen = false;

  // Mail compose state
  let showMailCompose = $state(false);
  let mailTo: DepartmentId = $state("development");
  let mailSubject = $state("");
  let mailBody = $state("");
  let mailPriority: "low" | "normal" | "high" | "urgent" = $state("normal");
  let mailType: "status" | "question" | "result" | "error" | "dispatch" = $state("dispatch");
  let sendingMail = $state(false);

  // FIXED: Store last message for retry functionality (Story 5.4)
  let lastFailedMessage = $state<string | null>(null);

  // Streaming state for Story 5.5
  let streamingContent = $state("");
  let isStreaming = $state(false);
  let cursorVisible = $state(true);
  let autoScroll = $state(true);
  let currentToolCall = $state<{ tool: string; status: "started" | "completed" } | null>(null);
  let fmModel = $state<string | null>(null);
  let fmProvider = $state<string | null>(null);

  // Copilot Kill Switch state (Story 5.6)
  let killSwitchActive = $state(false);
  let killSwitchLoading = $state(false);

  // ── Slash Commands ────────────────────────────────────────────────────────
  interface SlashCommand {
    command: string;
    description: string;
  }

  const SLASH_COMMANDS: SlashCommand[] = [
    { command: '/research', description: 'Start a research task' },
    { command: '/backtest', description: 'Run a backtest' },
    { command: '/scan',     description: 'Market scan' },
    { command: '/deploy',   description: 'Deploy an EA' },
    { command: '/report',   description: 'Generate a report' },
    { command: '/memory',   description: 'Query memory' },
  ];

  let showSlashMenu = $state(false);
  let slashQuery = $state('');
  let slashMenuIndex = $state(0);
  let slashMenuEl: HTMLDivElement = $state();

  let filteredSlashCommands = $derived(
    slashQuery
      ? SLASH_COMMANDS.filter(c => c.command.startsWith('/' + slashQuery))
      : SLASH_COMMANDS
  );

  // ── @ File Attachment ─────────────────────────────────────────────────────
  let showAssetMenu = $state(false);
  let assetQuery = $state('');
  let assetMenuIndex = $state(0);
  let allAssets = $state<SharedAsset[]>([]);
  let assetMenuLoading = $state(false);
  let attachedAssets = $state<SharedAsset[]>([]);

  let filteredAssets = $derived(
    assetQuery
      ? allAssets.filter(a =>
          a.name.toLowerCase().includes(assetQuery.toLowerCase()) ||
          a.type.toLowerCase().includes(assetQuery.toLowerCase())
        )
      : allAssets
  );

  // ── Workflow Templates (Story C1) ──────────────────────────────────────────
  interface WorkflowTemplate {
    id: string;
    name: string;
    description: string;
    trigger_message: string;
    departments: string[];
    estimated_duration: string;
  }

  let workflowsExpanded = $state(false);
  let workflows = $state<WorkflowTemplate[]>([]);
  let workflowsLoading = $state(false);

  async function loadWorkflows() {
    if (workflows.length > 0) return;
    workflowsLoading = true;
    try {
      const response = await fetch(`${API_BASE}/workflow-templates`);
      if (response.ok) {
        workflows = await response.json();
      }
    } catch (e) {
      console.error('Failed to load workflows:', e);
    } finally {
      workflowsLoading = false;
    }
  }

  async function triggerWorkflow(template: WorkflowTemplate) {
    // Send the trigger message to copilot chat
    message = template.trigger_message;
    workflowsExpanded = false;
    await sendMessage();
  }

  async function loadAllAssetsFlat() {
    if (allAssets.length > 0) return;
    assetMenuLoading = true;
    try {
      const grouped = await listAllAssets();
      const flat: SharedAsset[] = [];
      for (const type of Object.keys(grouped) as Array<keyof typeof grouped>) {
        flat.push(...grouped[type]);
      }
      allAssets = flat;
    } catch (e) {
      console.error('Failed to load assets for @ menu:', e);
    } finally {
      assetMenuLoading = false;
    }
  }

  function selectSlashCommand(cmd: SlashCommand) {
    // Replace the slash prefix the user typed with the full command + space
    const beforeSlash = message.slice(0, getSlashStart());
    message = beforeSlash + cmd.command + ' ';
    showSlashMenu = false;
    slashQuery = '';
    slashMenuIndex = 0;
    tick().then(() => {
      if (textareaElement) {
        textareaElement.focus();
        autoResize();
      }
    });
  }

  function selectAsset(asset: SharedAsset) {
    // Remove the @query from the input
    const atStart = getAtStart();
    const before = message.slice(0, atStart);
    const after = message.slice(atStart + 1 + assetQuery.length); // +1 for '@'
    message = before + after;
    // Attach
    if (!attachedAssets.find(a => a.id === asset.id)) {
      attachedAssets = [...attachedAssets, asset];
    }
    showAssetMenu = false;
    assetQuery = '';
    assetMenuIndex = 0;
    tick().then(() => {
      if (textareaElement) {
        textareaElement.focus();
        autoResize();
      }
    });
  }

  function removeAttachment(assetId: string) {
    attachedAssets = attachedAssets.filter(a => a.id !== assetId);
  }

  /** Returns the string index where the active `/` token starts. */
  function getSlashStart(): number {
    const val = message;
    const cursorPos = textareaElement?.selectionStart ?? val.length;
    const sub = val.slice(0, cursorPos);
    const lastSpace = sub.lastIndexOf(' ');
    return lastSpace === -1 ? 0 : lastSpace + 1;
  }

  /** Returns the string index where the active `@` token starts. */
  function getAtStart(): number {
    const val = message;
    const cursorPos = textareaElement?.selectionStart ?? val.length;
    const sub = val.slice(0, cursorPos);
    const lastAt = sub.lastIndexOf('@');
    return lastAt;
  }

  function handleInputChange() {
    autoResize();
    const val = message;
    const cursorPos = textareaElement?.selectionStart ?? val.length;
    const sub = val.slice(0, cursorPos);

    // Detect slash command trigger: `/` at start or after space
    const slashMatch = sub.match(/(^|\s)(\/(\w*))$/);
    if (slashMatch) {
      slashQuery = slashMatch[3]; // text after /
      showSlashMenu = filteredSlashCommands.length > 0;
      slashMenuIndex = 0;
      showAssetMenu = false;
    } else {
      showSlashMenu = false;
      slashQuery = '';
    }

    // Detect @ asset trigger
    const atMatch = sub.match(/(^|\s)@(\w*)$/);
    if (atMatch) {
      assetQuery = atMatch[2];
      showAssetMenu = true;
      assetMenuIndex = 0;
      showSlashMenu = false;
      loadAllAssetsFlat();
    } else {
      showAssetMenu = false;
      assetQuery = '';
    }
  }

  function handleDropdownKeydown(e: KeyboardEvent) {
    if (showSlashMenu) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        slashMenuIndex = (slashMenuIndex + 1) % filteredSlashCommands.length;
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        slashMenuIndex = (slashMenuIndex - 1 + filteredSlashCommands.length) % filteredSlashCommands.length;
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (filteredSlashCommands[slashMenuIndex]) {
          selectSlashCommand(filteredSlashCommands[slashMenuIndex]);
        }
      } else if (e.key === 'Escape') {
        showSlashMenu = false;
      }
      return;
    }
    if (showAssetMenu) {
      const total = filteredAssets.length;
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        assetMenuIndex = total ? (assetMenuIndex + 1) % total : 0;
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        assetMenuIndex = total ? (assetMenuIndex - 1 + total) % total : 0;
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (filteredAssets[assetMenuIndex]) {
          selectAsset(filteredAssets[assetMenuIndex]);
        }
      } else if (e.key === 'Escape') {
        showAssetMenu = false;
      }
      return;
    }
  }

  // Dismiss dropdowns on outside click
  function handleDocumentClick(e: MouseEvent) {
    const target = e.target as Node;
    if (slashMenuEl && !slashMenuEl.contains(target) && target !== textareaElement) {
      showSlashMenu = false;
    }
  }

  $effect(() => {
    if (showSlashMenu || showAssetMenu) {
      document.addEventListener('click', handleDocumentClick, true);
    } else {
      document.removeEventListener('click', handleDocumentClick, true);
    }
    return () => document.removeEventListener('click', handleDocumentClick, true);
  });

  // Initialize kill switch status on mount
  onMount(async () => {
    await checkKillSwitchStatus();
    await loadAgentModel();

    // Subscribe to server health threshold alerts (Story 10-5 AC3)
    // Injects a system message into the chat when a metric crosses its threshold.
    // Also restores any missed alert that fired while this panel was unmounted.
    let lastAlertId: string | null = null;

    function injectAlertMessage(alert: { id: string; message: string; timestamp: Date }) {
      lastAlertId = alert.id;
      messages = [
        ...messages,
        {
          id: `health-alert-${alert.id}`,
          role: 'system' as const,
          content: `[SERVER ALERT] ${alert.message}`,
          timestamp: alert.timestamp,
        }
      ];
      tick().then(scrollToBottom);
      clearServerHealthAlert(); // clear persisted so we don't re-show on next mount
    }

    // Restore missed alert from sessionStorage (fired while panel was unmounted)
    const missedAlert = getLastPersistedAlert();
    if (missedAlert) {
      injectAlertMessage(missedAlert);
    }

    const unsubHealthAlert = serverHealthAlertEvent.subscribe((alert) => {
      if (alert && alert.id !== lastAlertId) {
        injectAlertMessage(alert);
      }
    });

    return () => {
      unsubHealthAlert();
    };
  });

  // Cursor blink effect (600ms)
  $effect(() => {
    if (isStreaming) {
      const interval = setInterval(() => {
        cursorVisible = !cursorVisible;
      }, 600);
      return () => clearInterval(interval);
    }
  });

  // Messages with Floor Manager
  interface FloorManagerMessage {
    id: string;
    role: "user" | "floor_manager" | "system";
    content: string;
    timestamp: Date;
    delegation?: {
      departmentId: DepartmentId;
      taskId: string;
      status: DelegatedTask["status"];
    };
    error?: string;
    retry?: boolean;
    isStreaming?: boolean;  // Track if message is still streaming
    thinking?: string;  // Extended thinking content
    toolCall?: {
      tool: string;
      status: "started" | "completed";
    };
  }

  // Floor Manager greeting
  const floorManagerGreeting = "Hello! I'm the Floor Manager. I coordinate tasks across all departments - Analysis, Research, Risk, Execution, and Portfolio. I can delegate tasks, check status, or answer questions about the trading floor. How can I help?";

  let greeting = floorManagerGreeting;
  let agentName = "Floor Manager";
  let placeholderText = "Ask the Floor Manager...";
  // Use the session-backed chat API so live streaming and saved history share one source of truth.
  const apiEndpoint = `${API_BASE}/chat/floor-manager/message`;
  let messages = [
    {
      id: "fm_welcome",
      role: "floor_manager" as const,
      content: greeting,
      timestamp: new Date(),
    },
  ];

  // Department info for display
  const departmentList = Object.values(DEPARTMENTS);

  // Subscribed values
  let activeTasks = $derived($activeDelegatedTasks);
  let isLoading = $derived($departmentChatStore.isLoading);

  // Group sessions by time
  function groupSessionsByTime(sessionList: ChatSession[]): GroupedSessions {
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    const last7Days = new Date(today);
    last7Days.setDate(last7Days.getDate() - 7);
    const last30Days = new Date(today);
    last30Days.setDate(last30Days.getDate() - 30);

    const grouped: GroupedSessions = {
      today: [],
      yesterday: [],
      last7Days: [],
      last30Days: [],
      older: [],
    };

    sessionList.forEach((session) => {
      const sessionDate = new Date(session.created_at);
      if (sessionDate >= today) {
        grouped.today.push(session);
      } else if (sessionDate >= yesterday) {
        grouped.yesterday.push(session);
      } else if (sessionDate >= last7Days) {
        grouped.last7Days.push(session);
      } else if (sessionDate >= last30Days) {
        grouped.last30Days.push(session);
      } else {
        grouped.older.push(session);
      }
    });

    return grouped;
  }

  function sortSessionsByUpdated(sessionList: ChatSession[]): ChatSession[] {
    return [...sessionList].sort(
      (a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
    );
  }

  function applySessions(sessionList: ChatSession[]) {
    sessions = sortSessionsByUpdated(sessionList);
    groupedSessions = groupSessionsByTime(sessions);
  }

  function resetMessagesToGreeting() {
    messages = [
      {
        id: "fm_welcome",
        role: "floor_manager" as const,
        content: greeting,
        timestamp: new Date(),
      },
    ];
  }

  // Load chat sessions
  async function loadSessions() {
    chatHistoryLoading = true;
    try {
      const agentType = 'floor-manager';
      const fetched = await chatApi.listSessions(undefined, agentType);
      applySessions(fetched);
    } catch (e) {
      console.error('Failed to load sessions:', e);
    } finally {
      chatHistoryLoading = false;
    }
  }

  // Create new chat session
  async function createNewChat() {
    if (sessionActionLoading) return;
    sessionActionLoading = true;
    try {
      const agentType = 'floor-manager';
      const now = new Date();
      const newSession = await chatApi.createSession({
        agentType,
        agentId: 'floor-manager',
        userId: 'default_user',
        title: `Chat ${now.toLocaleString()}`,
        context: {
          canvas: 'workshop',
          session_type: 'interactive_session',
        },
      });
      currentSessionId = newSession.id;
      resetMessagesToGreeting();
      applySessions([newSession, ...sessions.filter((session) => session.id !== newSession.id)]);
      await loadSessions();
    } catch (e) {
      console.error('Failed to create new chat:', e);
    } finally {
      sessionActionLoading = false;
    }
  }

  // Load session messages
  async function loadSession(sessionId: string) {
    try {
      currentSessionId = sessionId;
      renamingSessionId = null;
      renameSessionTitle = '';
      const stored: StoredChatMessage[] = await chatApi.getSessionMessages(sessionId);
      messages = stored.map((msg) => ({
        id: msg.id,
        role: msg.role === 'assistant' ? 'floor_manager' : (msg.role as FloorManagerMessage["role"]),
        content: msg.content,
        timestamp: new Date(msg.created_at),
      }));
      showChatHistory = false;
    } catch (e) {
      console.error('Failed to load session:', e);
    }
  }

  // Delete session
  async function deleteSession(sessionId: string, event: MouseEvent) {
    event.stopPropagation();
    if (sessionActionLoading) return;
    sessionActionLoading = true;
    if (renamingSessionId === sessionId) {
      renamingSessionId = null;
      renameSessionTitle = '';
    }

    const nextSessions = sessions.filter((session) => session.id !== sessionId);
    applySessions(nextSessions);
    if (currentSessionId === sessionId) {
      currentSessionId = null;
      resetMessagesToGreeting();
    }

    try {
      await chatApi.deleteSession(sessionId);
      await loadSessions();
    } catch (e) {
      console.error('Failed to delete session:', e);
      await loadSessions();
    } finally {
      sessionActionLoading = false;
    }
  }

  function startSessionRename(session: ChatSession, event: MouseEvent) {
    event.stopPropagation();
    renamingSessionId = session.id;
    renameSessionTitle = (session.title || '').trim();
  }

  function cancelSessionRename(event?: Event) {
    event?.stopPropagation();
    renamingSessionId = null;
    renameSessionTitle = '';
  }

  async function saveSessionRename(sessionId: string, event?: Event) {
    event?.stopPropagation();
    const nextTitle = renameSessionTitle.trim();
    if (!nextTitle || sessionActionLoading) {
      cancelSessionRename();
      return;
    }

    sessionActionLoading = true;
    const previous = sessions;

    applySessions(
      sessions.map((session) =>
        session.id === sessionId ? { ...session, title: nextTitle } : session
      )
    );
    renamingSessionId = null;
    renameSessionTitle = '';

    try {
      await chatApi.updateSessionTitle(sessionId, { title: nextTitle });
      await loadSessions();
    } catch (e) {
      console.error('Failed to rename session:', e);
      applySessions(previous);
      await loadSessions();
    } finally {
      sessionActionLoading = false;
    }
  }

  async function clearAllSessions() {
    if (sessions.length === 0 || sessionActionLoading) return;

    sessionActionLoading = true;
    const sessionIds = sessions.map((session) => session.id);
    const previous = sessions;

    applySessions([]);
    currentSessionId = null;
    resetMessagesToGreeting();

    try {
      await Promise.all(sessionIds.map((sessionId) => chatApi.deleteSession(sessionId)));
    } catch (e) {
      console.error('Failed to clear session history:', e);
      applySessions(previous);
      await loadSessions();
    } finally {
      sessionActionLoading = false;
    }
  }

  // Toggle chat history sidebar
  function toggleChatHistory() {
    showChatHistory = !showChatHistory;
    if (showChatHistory) {
      loadSessions();
    }
  }

  // Generate message ID
  function generateId(): string {
    return `fm_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Auto-resize textarea
  function autoResize() {
    if (textareaElement) {
      textareaElement.style.height = "auto";
      const newHeight = Math.min(Math.max(textareaElement.scrollHeight, 60), 150);
      textareaElement.style.height = newHeight + "px";
    }
  }

  // Scroll handler to pause auto-scroll when user scrolls up (Story 5.5)
  function handleScroll() {
    if (!messagesContainer) return;
    const { scrollTop, scrollHeight, clientHeight } = messagesContainer;
    // If user scrolls up, pause auto-scroll
    autoScroll = (scrollHeight - scrollTop - clientHeight) < 50;
  }

  // Scroll to bottom function
  function scrollToBottom() {
    if (messagesContainer && autoScroll) {
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
  }

  async function loadAgentModel() {
    try {
      const res = await fetch(`${API_BASE}/agent-config/floor_manager/model`);
      if (res.ok) {
        const data = await res.json();
        fmModel = data?.model ?? fmModel;
        fmProvider = data?.provider ?? fmProvider;
      }
    } catch (e) {
      console.error("Failed to load floor_manager model config", e);
    }
  }

  function handleModelChange(event: CustomEvent<{ model: string; provider?: string }>) {
    fmModel = event.detail.model;
    fmProvider = event.detail.provider ?? fmProvider;
  }

  // Send message to Floor Manager with streaming support (Story 5.5)
  async function sendMessage() {
    if (!message.trim() || isLoading) return;

    // Build user content, appending any attached assets
    let userContent = message.trim();
    if (attachedAssets.length > 0) {
      const attachmentLines = attachedAssets
        .map(a => `[Attached: ${a.name} (${a.type})]`)
        .join('\n');
      userContent = userContent + '\n\n' + attachmentLines;
      attachedAssets = [];
    }
    message = "";

    // Add user message
    const userMessageId = generateId();
    messages = [
      ...messages,
      {
        id: userMessageId,
        role: "user",
        content: userContent,
        timestamp: new Date(),
      },
    ];

    await tick();
    scrollToBottom();

    // Prepare streaming message placeholder
    const assistantMessageId = generateId();
    let streamingMessage: FloorManagerMessage = {
      id: assistantMessageId,
      role: "floor_manager",
      content: "",
      timestamp: new Date(),
      isStreaming: true,
    };

    // Add empty streaming message
    messages = [...messages, streamingMessage];

    // Enable streaming state
    isStreaming = true;
    streamingContent = "";

    try {
      // Get current canvas context
      let currentCanvas = 'workshop';
      activeCanvasStore.subscribe(value => {
        currentCanvas = value;
      })();

      // Include conversation history (Story 5.4)
      const conversationHistory = messages
        .filter(m => m.role === "user" || m.role === "floor_manager")
        .slice(-CONVERSATION_HISTORY_LIMIT) // Last N messages for context
        .map(m => ({
          role: m.role === "floor_manager" ? "assistant" : m.role,
          content: m.content,
        }));

      // Use SSE streaming (Story 5.5)
      const response = await fetch(apiEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userContent,
          session_id: currentSessionId ?? undefined,
          context: {
            canvas_context: currentCanvas,
            model: fmModel,
            provider: fmProvider,
            session_type: 'interactive_session',
            workspace_contract: {
              version: 'manifest-v1',
              strategy: 'manifest-first',
              natural_resource_search: true,
            },
          },
          history: conversationHistory,
          stream: true,  // Enable streaming (Story 5.5)
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      // Handle streaming response
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let fullContent = "";
      let thinkingContent = "";
      let delegation: FloorManagerMessage["delegation"] | undefined;
      let lineBuffer = "";  // Buffer for partial SSE lines
      let resolvedSessionId = currentSessionId;

      // NFR-P3: Track first token time
      const streamStartTime = performance.now();
      let firstTokenTime: number | null = null;

      if (reader) {
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              // Process any remaining buffer
              if (lineBuffer.startsWith("data: ")) {
                try {
                  const data = JSON.parse(lineBuffer.slice(6));
                  // Handle remaining events...
                } catch {
                  // Ignore parse errors on final buffer
                }
              }
              break;
            }

            const chunk = decoder.decode(value, { stream: true });
            lineBuffer += chunk;
            const lines = lineBuffer.split("\n");
            // Keep incomplete last line in buffer
            lineBuffer = lines.pop() || "";

            for (const line of lines) {
              if (line.startsWith("data: ")) {
                try {
                  const data = JSON.parse(line.slice(6));

                  // Handle tool call events
                  if (data.type === "tool") {
                    currentToolCall = { tool: data.tool, status: data.status };
                    streamingMessage.toolCall = currentToolCall;
                    messages = [...messages];  // Trigger reactivity
                    continue;
                  }

                  // Handle thinking chunks
                  if (data.type === "thinking" && data.content !== undefined) {
                    thinkingContent += data.content;
                    streamingMessage.thinking = thinkingContent;
                    messages = [...messages];
                    continue;
                  }

                  // Handle content deltas
                  if (data.type === "content" && data.delta) {
                    // NFR-P3: Track first token time
                    if (firstTokenTime === null) {
                      firstTokenTime = performance.now() - streamStartTime;
                      console.debug(`[NFR-P3] First token: ${firstTokenTime.toFixed(0)}ms`);
                    }
                    fullContent += data.delta;
                    streamingMessage.content = fullContent;
                    messages = [...messages];  // Trigger reactivity
                    await tick();
                    scrollToBottom();
                    continue;
                  }

                  // Handle delegation events
                  if (data.type === "delegation") {
                    delegation = {
                      departmentId: data.department as DepartmentId,
                      taskId: data.task_id,
                      status: data.status,
                    };
                    streamingMessage.delegation = delegation;
                    messages = [...messages];
                    continue;
                  }

                  if (data.type === "done") {
                    if (typeof data.session_id === "string" && data.session_id) {
                      resolvedSessionId = data.session_id;
                    }
                    isStreaming = false;
                    streamingMessage.isStreaming = false;
                    currentToolCall = null;
                    messages = [...messages];
                    continue;
                  }

                  // Handle error events
                  if (data.type === "error") {
                    throw new Error(data.error);
                  }
                } catch (e) {
                  // Skip parse errors for incomplete chunks
                }
              }
            }
          }
        } finally {
          reader.releaseLock();
        }
      }

      // Mark streaming complete
      isStreaming = false;
      streamingMessage.isStreaming = false;
      streamingMessage.content = fullContent || "I've processed your request.";
      streamingMessage.delegation = delegation;
      if (thinkingContent) streamingMessage.thinking = thinkingContent;
      messages = [...messages];
      currentSessionId = resolvedSessionId;
      await loadSessions();

      await tick();
      scrollToBottom();
    } catch (error) {
      // Handle errors
      lastFailedMessage = userContent;
      messages = messages.map(m =>
        m.id === assistantMessageId
          ? {
              ...m,
              role: "system" as const,
              content: `Error: ${error instanceof Error ? error.message : "Failed to send message"}`,
              isStreaming: false,
              error: error instanceof Error ? error.message : "Failed to send message",
              retry: true,
            }
          : m
      );
      isStreaming = false;
    }
  }

  // FIXED: Retry last failed message (Story 5.4)
  async function retryLastMessage() {
    if (!lastFailedMessage || isLoading) return;

    const retryMsg = lastFailedMessage;
    lastFailedMessage = null;
    message = retryMsg;
    await sendMessage();
  }

  // Delegate to specific department
  async function delegateToDepartment(deptId: DepartmentId, task: string) {
    if (isLoading) return;

    // Add delegation message
    messages = [
      ...messages,
      {
        id: generateId(),
        role: "user",
        content: `Delegate to ${DEPARTMENTS[deptId].name}: ${task}`,
        timestamp: new Date(),
      },
    ];

    await tick();
    scrollToBottom();

    try {
      const result = await departmentChatStore.delegateTask(deptId, task);

      if (result) {
        messages = [
          ...messages,
          {
            id: generateId(),
            role: "floor_manager",
            content: `Task delegated to ${DEPARTMENTS[deptId].name}. ${result.result || "Processing..."}`,
            timestamp: new Date(),
            delegation: {
              departmentId: deptId,
              taskId: result.id,
              status: result.status,
            },
          },
        ];
      }
    } catch (error) {
      messages = [
        ...messages,
        {
          id: generateId(),
          role: "system",
          content: `Delegation failed: ${error instanceof Error ? error.message : "Unknown error"}`,
          timestamp: new Date(),
        },
      ];
    }

    await tick();
    scrollToBottom();
  }

  // Quick delegation actions
  const quickDelegations: Array<{ label: string; dept: DepartmentId; task: string }> = [
    { label: "Analyze Market", dept: "development", task: "Perform market analysis on current conditions" },
    { label: "Run Backtest", dept: "research", task: "Run backtest on active strategy" },
    { label: "Check Risk", dept: "risk", task: "Generate risk exposure report" },
    { label: "Order Status", dept: "trading", task: "Check status of all orders" },
    { label: "Portfolio Review", dept: "portfolio", task: "Generate portfolio review" },
  ];

  // Handle keyboard
  function handleKeydown(e: KeyboardEvent) {
    // Route arrow/enter/escape to dropdown handlers first
    if (showSlashMenu || showAssetMenu) {
      handleDropdownKeydown(e);
      return;
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  // Format timestamp
  function formatTime(date: Date): string {
    return new Date(date).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  // Get status icon
  function getStatusIcon(status: DelegatedTask["status"]): typeof CheckCircle2 {
    switch (status) {
      case "completed":
        return CheckCircle2;
      case "error":
        return AlertCircle;
      default:
        return Clock;
    }
  }

  // Get status color
  function getStatusColor(status: DelegatedTask["status"]): string {
    switch (status) {
      case "completed":
        return "#22c55e";
      case "error":
        return "#ef4444";
      default:
        return "#f59e0b";
    }
  }

  // Clear chat
  function clearChat() {
    currentSessionId = null;
    currentToolCall = null;
    pendingConfirmation = null;
    lastFailedMessage = null;
    attachedAssets = [];
    showMailCompose = false;
    messages = [
      {
        id: "fm_welcome",
        role: "floor_manager",
        content:
          "Chat cleared. I'm ready to help coordinate tasks across departments. What would you like to do?",
        timestamp: new Date(),
      },
    ];
  }

  // Open department chat
  function openDepartmentChat(deptId: DepartmentId) {
    dispatch("openDepartment", { departmentId: deptId });
  }

  // Close panel
  function closePanel() {
    dispatch("close");
  }

  // Send mail to department
  async function sendMailToDepartment() {
    if (!mailSubject.trim() || !mailBody.trim() || sendingMail) return;

    sendingMail = true;

    try {
      const response = await fetch(`${API_BASE}/departments/mail/send`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          from_dept: "floor_manager",
          to_dept: mailTo,
          type: mailType,
          subject: mailSubject.trim(),
          body: mailBody.trim(),
          priority: mailPriority,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();

      // Add confirmation message
      messages = [
        ...messages,
        {
          id: generateId(),
          role: "floor_manager",
          content: `Mail sent to ${DEPARTMENTS[mailTo].name}: "${mailSubject}"`,
          timestamp: new Date(),
        },
      ];

      // Reset form
      mailSubject = "";
      mailBody = "";
      showMailCompose = false;

      await tick();
      scrollToBottom();
    } catch (error) {
      messages = [
        ...messages,
        {
          id: generateId(),
          role: "system",
          content: `Failed to send mail: ${error instanceof Error ? error.message : "Unknown error"}`,
          timestamp: new Date(),
        },
      ];
    } finally {
      sendingMail = false;
    }
  }

  // Quick mail to department
  async function quickMailToDepartment(deptId: DepartmentId, subject: string, body: string) {
    try {
      const response = await fetch(`${API_BASE}/departments/mail/send`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          from_dept: "floor_manager",
          to_dept: deptId,
          type: "dispatch",
          subject,
          body,
          priority: "normal",
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      messages = [
        ...messages,
        {
          id: generateId(),
          role: "floor_manager",
          content: `Task dispatched to ${DEPARTMENTS[deptId].name}: "${subject}"`,
          timestamp: new Date(),
        },
      ];

      await tick();
      scrollToBottom();
    } catch (error) {
      messages = [
        ...messages,
        {
          id: generateId(),
          role: "system",
          content: `Failed to dispatch: ${error instanceof Error ? error.message : "Unknown error"}`,
          timestamp: new Date(),
        },
      ];
    }
  }

  // Copilot Kill Switch functions (Story 5.6)
  async function activateCopilotKillSwitch() {
    if (killSwitchLoading || killSwitchActive) return;

    killSwitchLoading = true;
    try {
      const result = await copilotKillSwitchService.activate('user');
      if (result.success) {
        killSwitchActive = true;
        // Add system message indicating suspension
        messages = [
          ...messages,
          {
            id: generateId(),
            role: 'system',
            content: '[SUSPENDED] Agent activity suspended - All AI tasks have been halted. Live trading is unaffected.',
            timestamp: new Date(),
            isKillSwitch: true,
          },
        ];
        await tick();
        scrollToBottom();
      }
    } catch (error) {
      console.error('Failed to activate kill switch:', error);
    } finally {
      killSwitchLoading = false;
    }
  }

  async function resumeCopilot() {
    if (killSwitchLoading || !killSwitchActive) return;

    killSwitchLoading = true;
    try {
      const result = await copilotKillSwitchService.resume();
      if (result.success) {
        killSwitchActive = false;
        // Add system message indicating resume
        messages = [
          ...messages,
          {
            id: generateId(),
            role: 'system',
            content: '[RESUMED] Agent activity resumed - AI tasks are now enabled.',
            timestamp: new Date(),
          },
        ];
        await tick();
        scrollToBottom();
      }
    } catch (error) {
      console.error('Failed to resume copilot:', error);
    } finally {
      killSwitchLoading = false;
    }
  }

  // Check kill switch status on mount
  async function checkKillSwitchStatus() {
    try {
      const status = await copilotKillSwitchService.getStatus();
      killSwitchActive = status.active;
    } catch (error) {
      console.error('Failed to check kill switch status:', error);
    }
  }

  // Story 5.7: Handle command confirmation
  async function confirmAction() {
    if (!pendingConfirmation || confirmationLoading) return;

    confirmationLoading = true;
    try {
      const response = await intentService.sendCommand(pendingConfirmation.message, true);
      await handleIntentResponse(response);
    } catch (error) {
      console.error('Failed to confirm action:', error);
    } finally {
      confirmationLoading = false;
      pendingConfirmation = null;
    }
  }

  function cancelConfirmation() {
    pendingConfirmation = null;
    // Add system message indicating cancellation
    messages = [
      ...messages,
      {
        id: generateId(),
        role: 'system',
        content: 'Action cancelled.',
        timestamp: new Date(),
      },
    ];
    tick().then(scrollToBottom);
  }

  // Story 5.7: Handle intent response (confirmation/clarification/success)
  async function handleIntentResponse(response: CommandResponse) {
    if (intentService.isConfirmationNeeded(response)) {
      // Show confirmation dialog
      pendingConfirmation = intentService.parseConfirmation(response);
    } else if (intentService.isClarificationNeeded(response)) {
      // Show clarification suggestions
      const suggestions = intentService.parseSuggestions(response);
      messages = [
        ...messages,
        {
          id: generateId(),
          role: 'system',
          content: response.message,
          suggestions: suggestions,
          timestamp: new Date(),
        },
      ];
      await tick();
      scrollToBottom();
    } else if (intentService.isSuccess(response)) {
      // Add success message
      messages = [
        ...messages,
        {
          id: generateId(),
          role: 'floor_manager',
          content: response.message,
          timestamp: new Date(),
        },
      ];
      await tick();
      scrollToBottom();
    } else if (intentService.isError(response)) {
      // Add error message
      messages = [
        ...messages,
        {
          id: generateId(),
          role: 'system',
          content: `Error: ${response.message}`,
          isError: true,
          timestamp: new Date(),
        },
      ];
      await tick();
      scrollToBottom();
    }
  }

  // Story 5.7: Initialize canvas context on mount
  function initializeCanvasContext() {
    const currentCanvas = $activeCanvasStore || 'workshop';
    canvasContextStore.setContext({
      canvas: currentCanvas,
      session_id: currentSessionId || generateId()
    });
  }
</script>

<div class="floor-manager-panel">
  <!-- Header -->
  <div class="panel-header">
    <div class="header-left">
      <div class="manager-icon">
        <Building2 size={18} />
      </div>
      <div class="manager-info">
        <span class="manager-name">{agentName}</span>
        <span class="manager-role">Trading Floor Coordinator</span>
      </div>
    </div>

    <div class="header-actions">
      <!-- Copilot Kill Switch (Story 5.6) -->
      {#if !killSwitchActive}
        <button
          class="icon-btn kill-switch-btn"
          title="Stop Agent Activity"
          onclick={activateCopilotKillSwitch}
          disabled={killSwitchLoading}
        >
          <Power size={14} />
        </button>
      {:else}
        <button
          class="icon-btn resume-btn"
          title="Resume Agent Activity"
          onclick={resumeCopilot}
          disabled={killSwitchLoading}
        >
          <Play size={14} />
        </button>
      {/if}
      <button class="icon-btn mail-btn" title="Send Mail to Department" onclick={() => showMailCompose = !showMailCompose}>
        <MailPlus size={14} />
      </button>
      <button class="icon-btn" title="Chat History" onclick={toggleChatHistory}>
        <History size={14} />
      </button>
      <button class="icon-btn" title="New Chat" onclick={createNewChat}>
        <MessageSquarePlus size={14} />
      </button>
      <button class="icon-btn" title="Close" onclick={closePanel}>
        <X size={14} />
      </button>
    </div>
  </div>

  <!-- Mail Compose Modal -->
  {#if showMailCompose}
    <div class="mail-compose-modal" transition:slide={{ y: -10 }}>
      <div class="mail-compose-header">
        <Mail size={14} />
        <span>Send Mail to Department</span>
        <button class="close-modal-btn" onclick={() => showMailCompose = false}>
          <X size={12} />
        </button>
      </div>

      <div class="mail-compose-body">
        <div class="form-row">
          <label>To Department:</label>
          <div class="dept-selector">
            {#each departmentList as dept}
              <button
                class="dept-option"
                class:selected={mailTo === dept.id}
                style="--dept-color: {dept.color}"
                onclick={() => mailTo = dept.id}
              >
                <span class="dept-dot"></span>
                {dept.name}
              </button>
            {/each}
          </div>
        </div>

        <div class="form-row">
          <label>Priority:</label>
          <div class="priority-selector">
            {#each [{v: 'low', c: '#6b7280'}, {v: 'normal', c: '#3b82f6'}, {v: 'high', c: '#f59e0b'}, {v: 'urgent', c: '#ef4444'}] as p}
              <button
                class="priority-option"
                class:selected={mailPriority === p.v}
                style="--priority-color: {p.c}"
                onclick={() => mailPriority = p.v}
              >
                {p.v}
              </button>
            {/each}
          </div>
        </div>

        <div class="form-row">
          <label>Type:</label>
          <select bind:value={mailType}>
            <option value="status">Status Update</option>
            <option value="question">Question</option>
            <option value="result">Result</option>
            <option value="dispatch" selected>Task Dispatch</option>
            <option value="error">Error Report</option>
          </select>
        </div>

        <div class="form-row">
          <label>Subject:</label>
          <input type="text" bind:value={mailSubject} placeholder="Enter subject..." />
        </div>

        <div class="form-row">
          <label>Message:</label>
          <textarea bind:value={mailBody} placeholder="Enter your message to the department head..." rows="4"></textarea>
        </div>
      </div>

      <div class="mail-compose-footer">
        <button class="cancel-btn" onclick={() => showMailCompose = false}>Cancel</button>
        <button
          class="send-mail-btn"
          onclick={sendMailToDepartment}
          disabled={!mailSubject.trim() || !mailBody.trim() || sendingMail}
        >
          {#if sendingMail}
            <Loader size={12} class="spinning" />
          {:else}
            <Send size={12} />
          {/if}
          Send Mail
        </button>
      </div>
    </div>
  {/if}

  <!-- Active Tasks Banner -->
  {#if activeTasks.length > 0}
    <div class="active-tasks-banner" transition:slide={{ y: -10 }}>
      <div class="tasks-header">
        <Clock size={12} />
        <span>{activeTasks.length} Active Task{activeTasks.length !== 1 ? 's' : ''}</span>
      </div>
      <div class="tasks-list">
        {#each activeTasks.slice(0, 3) as task}
          {@const SvelteComponent = getStatusIcon(task.status)}
          <div class="task-item">
            <div class="task-dept" style="background: {DEPARTMENTS[task.departmentId].color}20; color: {DEPARTMENTS[task.departmentId].color}">
              {DEPARTMENTS[task.departmentId].name}
            </div>
            <div class="task-status" style="color: {getStatusColor(task.status)}">
              <SvelteComponent size={10} />
            </div>
          </div>
        {/each}
        {#if activeTasks.length > 3}
          <div class="task-more">+{activeTasks.length - 3} more</div>
        {/if}
      </div>
    </div>
  {/if}

  <!-- Messages -->
  <div class="messages" bind:this={messagesContainer} onscroll={handleScroll}>
    {#each messages as msg}
      <div class="message {msg.role}">
        <div class="message-avatar">
          {#if msg.role === "user"}
            <span>Y</span>
          {:else if msg.role === "floor_manager"}
            <Building2 size={14} />
          {:else}
            <AlertCircle size={14} />
          {/if}
        </div>
        <div class="message-body" class:kill-switch-message={msg.isKillSwitch}>
          {#if msg.role === "floor_manager"}
            <div class="message-label">{agentName}</div>
          {:else if msg.role === "system"}
            <div class="message-label system">System</div>
          {/if}
          <!-- Extended thinking block -->
          {#if msg.thinking}
            <details class="thinking-block">
              <summary class="thinking-summary">
                <ChevronRight size={12} class="thinking-chevron" />
                Thinking
              </summary>
              <div class="thinking-content">{msg.thinking}</div>
            </details>
          {/if}
          <div class="message-text">
            {#if msg.role === 'floor_manager' || msg.role === 'assistant'}
              {@html renderMarkdown(msg.content)}
            {:else}
              {msg.content}
            {/if}
            <!-- Typing cursor (Story 5.5) -->
            {#if msg.isStreaming}
              <span class="typing-cursor" class:visible={cursorVisible}>|</span>
            {/if}
          </div>
          <!-- Tool call UI (Story 5.5) -->
          {#if msg.toolCall}
            <div class="tool-call" class:completed={msg.toolCall.status === "completed"}>
              {#if msg.toolCall.status === "started"}
                <span class="pulse-dot"></span>
                <span>Using: {msg.toolCall.tool}…</span>
              {:else}
                <CheckCircle2 size={12} />
                <span>Using: {msg.toolCall.tool}</span>
              {/if}
            </div>
          {/if}
          {#if msg.delegation}
            {@const SvelteComponent_1 = getStatusIcon(msg.delegation.status)}
            <div class="delegation-info">
              <ArrowRightCircle size={12} />
              <span>
                Delegated to <strong>{DEPARTMENTS[msg.delegation.departmentId].name}</strong>
              </span>
              <span class="delegation-status" style="color: {getStatusColor(msg.delegation.status)}">
                <SvelteComponent_1 size={10} />
                {msg.delegation.status}
              </span>
            </div>
          {/if}
          <!-- FIXED: Retry button for error messages (Story 5.4) -->
          {#if msg.retry}
            <button class="retry-button" onclick={retryLastMessage}>
              <Loader size={12} />
              Retry
            </button>
          {/if}
          <div class="message-time">{formatTime(msg.timestamp)}</div>
        </div>
      </div>
    {/each}

    {#if isLoading}
      <div class="message floor_manager">
        <div class="message-avatar">
          <Building2 size={14} />
        </div>
        <div class="message-body">
          <div class="typing-indicator">
            <Loader size={12} class="spinning" />
            <span>Coordinating...</span>
          </div>
        </div>
      </div>
    {/if}
  </div>

  <!-- Input -->
  <div class="input-area">
    <div class="input-wrapper">
      <div class="model-selector-inline">
        <AgentModelSelector
          agentId="floor_manager"
          currentModel={fmModel ?? 'opus'}
          on:modelchange={handleModelChange}
        />
      </div>
      <!-- Story 5.7: Confirmation Dialog -->
      {#if pendingConfirmation}
        <div class="confirmation-dialog" transition:slide={{ y: -10 }}>
          <div class="confirmation-content">
            <AlertCircle size={16} class="warning-icon" />
            <span class="confirmation-message">{pendingConfirmation.message}</span>
          </div>
          <div class="confirmation-actions">
            <button
              class="cancel-btn"
              onclick={cancelConfirmation}
              disabled={confirmationLoading}
            >
              Cancel
            </button>
            <button
              class="confirm-btn"
              onclick={confirmAction}
              disabled={confirmationLoading}
            >
              {#if confirmationLoading}
                <Loader size={14} class="spinning" />
              {:else}
                Confirm
              {/if}
            </button>
          </div>
        </div>
      {/if}

      <!-- Slash Commands Dropdown -->
      {#if showSlashMenu && filteredSlashCommands.length > 0}
        <div class="slash-menu" bind:this={slashMenuEl}>
          {#each filteredSlashCommands as cmd, i}
            <button
              class="slash-item"
              class:active={i === slashMenuIndex}
              onclick={() => selectSlashCommand(cmd)}
            >
              <span class="slash-cmd">{cmd.command}</span>
              <span class="slash-desc">{cmd.description}</span>
            </button>
          {/each}
        </div>
      {/if}

      <!-- @ Asset Dropdown -->
      {#if showAssetMenu}
        <div class="asset-menu">
          {#if assetMenuLoading}
            <div class="asset-menu-loading">
              <Loader size={12} class="spinning" />
              <span>Loading assets…</span>
            </div>
          {:else if filteredAssets.length === 0}
            <div class="asset-menu-empty">No assets found</div>
          {:else}
            {#each filteredAssets as asset, i}
              <button
                class="asset-item"
                class:active={i === assetMenuIndex}
                onclick={() => selectAsset(asset)}
              >
                <span class="asset-name">{asset.name}</span>
                <span class="asset-type-badge">{asset.type}</span>
              </button>
            {/each}
          {/if}
        </div>
      {/if}

      <!-- Workflow Templates (Story C1) -->
      <div class="workflows-section">
        <button
          class="workflows-toggle"
          onclick={() => { workflowsExpanded = !workflowsExpanded; if (!workflowsExpanded) loadWorkflows(); }}
        >
          <span class="workflows-toggle-icon" class:rotated={workflowsExpanded}>▶</span>
          <span>Workflows</span>
          {#if workflows.length > 0}
            <span class="workflows-count">{workflows.length}</span>
          {/if}
        </button>
        {#if workflowsExpanded}
          <div class="workflows-list">
            {#if workflowsLoading}
              <div class="workflows-loading">Loading workflows...</div>
            {:else}
              {#each workflows as wf}
                <button
                  class="workflow-item"
                  onclick={() => triggerWorkflow(wf)}
                  title={wf.description}
                >
                  <div class="workflow-name">{wf.name}</div>
                  <div class="workflow-meta">
                    <span class="workflow-duration">{wf.estimated_duration}</span>
                    <span class="workflow-depts">{wf.departments.join(', ')}</span>
                  </div>
                </button>
              {/each}
            {/if}
          </div>
        {/if}
      </div>

      <textarea
        bind:this={textareaElement}
        bind:value={message}
        onkeydown={handleKeydown}
        oninput={handleInputChange}
        placeholder={placeholderText}
        rows="1"
        disabled={isLoading}
      ></textarea>

      <!-- Attachment Chips -->
      {#if attachedAssets.length > 0}
        <div class="attachment-chips">
          {#each attachedAssets as asset}
            <span class="attachment-chip">
              <span class="chip-name">{asset.name}</span>
              <span class="chip-type">{asset.type}</span>
              <button class="chip-remove" onclick={() => removeAttachment(asset.id)} title="Remove attachment">
                <X size={10} />
              </button>
            </span>
          {/each}
        </div>
      {/if}
    </div>
    <div class="input-footer">
      <div class="char-count">{message.length} / 4000</div>
      <button
        class="send-btn"
        onclick={sendMessage}
        disabled={!message.trim() || isLoading}
      >
        {#if isLoading}
          <Loader size={16} class="spinning" />
        {:else}
          <Send size={16} />
        {/if}
      </button>
    </div>
  </div>
</div>

<!-- Chat History Sidebar -->
{#if showChatHistory}
  <div class="chat-history-sidebar" transition:fly={{ x: -300, duration: 200 }}>
    <div class="history-header">
      <span>Chat History</span>
      {#if sessions.length > 0}
        <button
          class="clear-history-btn"
          onclick={clearAllSessions}
          disabled={sessionActionLoading}
          title="Delete all chat sessions"
        >
          Clear history
        </button>
      {/if}
      <button class="close-history-btn" onclick={toggleChatHistory}>
        <X size={16} />
      </button>
    </div>
    <button class="new-chat-btn" onclick={createNewChat} disabled={sessionActionLoading}>
      <MessageSquarePlus size={16} />
      <span>New Chat</span>
    </button>
    <div class="history-list">
      {#if chatHistoryLoading}
        <div class="history-loading">Loading...</div>
      {:else}
        {#each sessionGroups as group}
          {#if group.items.length > 0}
            <div class="history-group">
              <div class="history-group-header">{group.label}</div>
              {#each group.items as session}
                <div
                  class="history-item"
                  class:active={currentSessionId === session.id}
                  onclick={() => loadSession(session.id)}
                  onkeydown={(e) => e.key === 'Enter' && loadSession(session.id)}
                  role="button"
                  tabindex="0"
                >
                  <MessageCircle size={14} />
                  {#if renamingSessionId === session.id}
                    <input
                      class="history-title-input"
                      bind:value={renameSessionTitle}
                      onclick={(e) => e.stopPropagation()}
                      onkeydown={(e) => {
                        if (e.key === 'Enter') {
                          void saveSessionRename(session.id, e);
                        } else if (e.key === 'Escape') {
                          cancelSessionRename(e);
                        }
                      }}
                    />
                    <button class="session-action-btn" title="Save title" onclick={(e) => saveSessionRename(session.id, e)}>
                      <Check size={12} />
                    </button>
                    <button class="session-action-btn" title="Cancel rename" onclick={(e) => cancelSessionRename(e)}>
                      <X size={12} />
                    </button>
                  {:else}
                    <span class="history-title">{session.title || 'New Conversation'}</span>
                    <button class="session-action-btn" title="Rename" onclick={(e) => startSessionRename(session, e)}>
                      <Pencil size={12} />
                    </button>
                    <button class="delete-session-btn" onclick={(e) => deleteSession(session.id, e)}>
                      <Trash2 size={12} />
                    </button>
                  {/if}
                </div>
              {/each}
            </div>
          {/if}
        {/each}
        {#if sessions.length === 0}
          <div class="history-empty">No chat history yet</div>
        {/if}
      {/if}
    </div>
  </div>
{/if}

<style>
  .floor-manager-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary, #0a0f1a);
    color: var(--text-primary, #e2e8f0);
    font-family: "Inter", sans-serif;
  }

  /* Header */
  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.625rem 0.75rem;
    background: var(--bg-secondary, #111827);
    border-bottom: 1px solid var(--border-color, #1e293b);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .manager-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 8px;
    color: white;
  }

  .manager-info {
    display: flex;
    flex-direction: column;
    gap: 0.125rem;
  }

  .manager-name {
    font-size: 0.875rem;
    font-weight: 600;
  }

  .manager-role {
    font-size: 0.625rem;
    color: var(--text-muted, #64748b);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .header-actions {
    display: flex;
    gap: 0.25rem;
  }

  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: transparent;
    border: none;
    border-radius: 4px;
    color: var(--text-muted, #64748b);
    cursor: pointer;
    transition: all 0.15s;
  }

  .icon-btn:hover {
    background: var(--bg-tertiary, #1e293b);
    color: var(--text-primary, #e2e8f0);
  }

  /* Copilot Kill Switch Button (Story 5.6) */
  .kill-switch-btn {
    color: #ef4444 !important;
  }

  .kill-switch-btn:hover {
    background: rgba(239, 68, 68, 0.2) !important;
    color: #ef4444 !important;
  }

  .resume-btn {
    color: #10b981 !important;
  }

  .resume-btn:hover {
    background: rgba(16, 185, 129, 0.2) !important;
    color: #10b981 !important;
  }

  /* Active Tasks Banner */
  .active-tasks-banner {
    padding: 0.5rem 0.75rem;
    background: rgba(245, 158, 11, 0.1);
    border-bottom: 1px solid rgba(245, 158, 11, 0.2);
  }

  .tasks-header {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    font-size: 0.6875rem;
    font-weight: 600;
    color: #f59e0b;
    margin-bottom: 0.375rem;
  }

  .tasks-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.375rem;
  }

  .task-item {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.25rem 0.5rem;
    background: var(--bg-tertiary, #1e293b);
    border-radius: 0.25rem;
  }

  .task-dept {
    font-size: 0.625rem;
    font-weight: 500;
    padding: 0.125rem 0.375rem;
    border-radius: 0.125rem;
  }

  .task-status {
    display: flex;
    align-items: center;
  }

  .task-more {
    font-size: 0.625rem;
    color: var(--text-muted, #64748b);
    padding: 0.25rem;
  }

  /* Messages */
  .messages {
    flex: 1;
    overflow-y: auto;
    padding: 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .message {
    display: flex;
    gap: 0.5rem;
    max-width: 90%;
  }

  .message.user {
    align-self: flex-end;
    flex-direction: row-reverse;
  }

  .message.floor_manager,
  .message.system {
    align-self: flex-start;
  }

  .message-avatar {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-tertiary, #1e293b);
    flex-shrink: 0;
    font-size: 0.75rem;
    font-weight: 600;
  }

  .message.user .message-avatar {
    background: var(--accent-primary, #3b82f6);
    color: white;
  }

  .message.floor_manager .message-avatar {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
  }

  .message.system .message-avatar {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .message-body {
    flex: 1;
    min-width: 0;
  }

  .message-label {
    font-size: 0.625rem;
    font-weight: 600;
    color: var(--accent-primary, #6366f1);
    text-transform: uppercase;
    margin-bottom: 0.25rem;
  }

  .message-label.system {
    color: #ef4444;
  }

  /* Story 5.6: Kill switch message styling - amber background */
  .kill-switch-message .message-text {
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 0.75rem;
    color: #fbbf24;
  }

  .message-text {
    padding: 0.5rem 0.75rem;
    border-radius: 0.75rem;
    font-size: 0.8125rem;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
  }

  /* Story 5.5: User messages - amber tint, right-aligned */
  .message.user .message-text {
    background: rgba(245, 158, 11, 0.15);  /* Amber tint */
    color: var(--text-primary, #f1f5f9);
    border-bottom-right-radius: 0.25rem;
    border: 1px solid rgba(245, 158, 11, 0.3);
  }

  /* Story 5.5: AI responses - cyan #00d4ff accent, left-aligned */
  .message.floor_manager .message-text {
    background: var(--bg-tertiary, #1e293b);
    border-bottom-left-radius: 0.25rem;
    border-left: 3px solid #00d4ff;  /* Cyan accent */
  }

  .message.system .message-text {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 0.5rem;
    font-size: 0.75rem;
  }

  .delegation-info {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    margin-top: 0.375rem;
    padding: 0.375rem 0.5rem;
    background: var(--bg-secondary, #111827);
    border-radius: 0.25rem;
    font-size: 0.6875rem;
  }

  .delegation-info strong {
    color: var(--accent-primary, #3b82f6);
  }

  .delegation-status {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    margin-left: auto;
    font-weight: 500;
  }

  /* Story 5.5: IBM Plex Mono 12px timestamps */
  .message-time {
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
    font-size: 0.75rem;  /* 12px */
    color: var(--text-muted, #64748b);
    margin-top: 0.25rem;
  }

  /* Story 5.5: Typing cursor with 600ms blink */
  .typing-cursor {
    display: inline-block;
    color: #00d4ff;
    font-weight: bold;
    opacity: 0;
    transition: opacity 0.1s;
  }

  .typing-cursor.visible {
    opacity: 1;
  }

  /* Extended thinking block */
  .thinking-block {
    margin-bottom: 0.375rem;
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 0.375rem;
    background: rgba(139, 92, 246, 0.06);
    overflow: hidden;
  }

  .thinking-summary {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.25rem 0.5rem;
    font-size: 0.6875rem;
    color: #a78bfa;
    cursor: pointer;
    user-select: none;
    list-style: none;
  }

  .thinking-summary::-webkit-details-marker {
    display: none;
  }

  .thinking-block[open] .thinking-chevron {
    transform: rotate(90deg);
  }

  .thinking-chevron {
    transition: transform 0.15s ease;
    flex-shrink: 0;
  }

  .thinking-content {
    padding: 0.375rem 0.5rem;
    font-size: 0.6875rem;
    color: rgba(167, 139, 250, 0.8);
    white-space: pre-wrap;
    word-break: break-word;
    border-top: 1px solid rgba(139, 92, 246, 0.15);
    max-height: 200px;
    overflow-y: auto;
  }

  /* Story 5.5: Tool call UI */
  .tool-call {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    margin-top: 0.375rem;
    padding: 0.25rem 0.5rem;
    background: rgba(99, 102, 241, 0.15);
    border-radius: 0.375rem;
    font-size: 0.6875rem;
    color: #a5b4fc;
  }

  .tool-call.completed {
    background: rgba(34, 197, 94, 0.15);
    color: #4ade80;
  }

  /* Pulsing dot animation */
  .pulse-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #a5b4fc;
    animation: pulse 1s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% {
      opacity: 1;
      transform: scale(1);
    }
    50% {
      opacity: 0.5;
      transform: scale(1.2);
    }
  }

  /* FIXED: Retry button styles (Story 5.4) */
  .retry-button {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    margin-top: 0.5rem;
    padding: 0.375rem 0.75rem;
    background: var(--accent-primary, #f59e0b);
    color: #000;
    border: none;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.2s;
  }

  .retry-button:hover {
    background: var(--accent-primary-hover, #d97706);
  }

  .retry-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .typing-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    color: var(--text-muted, #64748b);
    font-size: 0.75rem;
  }

  .spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  /* Department Shortcuts */
  .department-shortcuts {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    background: var(--bg-secondary, #111827);
    border-top: 1px solid var(--border-color, #1e293b);
    overflow-x: auto;
  }

  .shortcuts-label {
    font-size: 0.625rem;
    color: var(--text-muted, #64748b);
    flex-shrink: 0;
  }

  .shortcuts-list {
    display: flex;
    gap: 0.25rem;
  }

  .dept-shortcut {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.25rem 0.5rem;
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.25rem;
    color: var(--text-secondary, #94a3b8);
    font-size: 0.6875rem;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }

  .dept-shortcut:hover {
    border-color: var(--dept-color, #3b82f6);
    color: var(--text-primary, #e2e8f0);
  }

  .shortcut-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--dept-color, #3b82f6);
  }

  /* Quick Actions */
  .quick-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.375rem;
    padding: 0.5rem 0.75rem;
    border-top: 1px solid var(--border-color, #1e293b);
  }

  .quick-action-btn {
    padding: 0.375rem 0.625rem;
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.25rem;
    color: var(--text-secondary, #94a3b8);
    font-size: 0.6875rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  .quick-action-btn:hover:not(:disabled) {
    background: var(--bg-hover, #334155);
    border-color: var(--accent-primary, #3b82f6);
    color: var(--text-primary, #e2e8f0);
  }

  .quick-action-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  /* Input */
  .input-area {
    padding: 0.625rem 0.75rem;
    background: var(--bg-secondary, #111827);
    border-top: 1px solid var(--border-color, #1e293b);
  }

  .input-wrapper {
    position: relative;
  }

  .input-wrapper textarea {
    width: 100%;
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.5rem;
    padding: 0.5rem 0.75rem;
    color: var(--text-primary, #e2e8f0);
    font-size: 0.8125rem;
    font-family: inherit;
    resize: none;
    min-height: 40px;
    max-height: 120px;
    line-height: 1.4;
  }

  .model-selector-inline {
    margin-bottom: 0.5rem;
    flex-shrink: 0;
  }

  .input-wrapper textarea:focus {
    outline: none;
    border-color: var(--accent-primary, #3b82f6);
  }

  .input-wrapper textarea:disabled {
    opacity: 0.6;
  }

  /* Chat History Sidebar */
  .chat-history-sidebar {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 280px;
    background: var(--bg-secondary, #111827);
    border-right: 1px solid var(--border-color, #1e293b);
    display: flex;
    flex-direction: column;
    z-index: 100;
    overflow: hidden;
  }

  .history-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem;
    border-bottom: 1px solid var(--border-color, #1e293b);
    font-weight: 600;
    font-size: 0.875rem;
  }

  .close-history-btn {
    background: none;
    border: none;
    color: var(--text-muted, #64748b);
    cursor: pointer;
    padding: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .close-history-btn:hover {
    color: var(--text-primary, #e2e8f0);
  }

  .new-chat-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    margin: 0.75rem;
    padding: 0.625rem;
    background: var(--accent-primary, #3b82f6);
    border: none;
    border-radius: 0.375rem;
    color: white;
    cursor: pointer;
    font-size: 0.8125rem;
    font-weight: 500;
    transition: background 0.15s;
  }

  .new-chat-btn:hover {
    background: var(--accent-primary-hover, #2563eb);
  }

  .new-chat-btn:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }

  .clear-history-btn {
    margin-left: auto;
    margin-right: 0.5rem;
    padding: 0.2rem 0.45rem;
    border: 1px solid rgba(248, 113, 113, 0.32);
    background: rgba(239, 68, 68, 0.08);
    color: #fca5a5;
    border-radius: 0.25rem;
    font-size: 0.66rem;
    cursor: pointer;
    transition: background 0.15s ease, color 0.15s ease;
  }

  .clear-history-btn:hover:not(:disabled) {
    background: rgba(239, 68, 68, 0.15);
    color: #fecaca;
  }

  .clear-history-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .history-list {
    flex: 1;
    overflow-y: auto;
    padding: 0.5rem;
  }

  .history-loading,
  .history-empty {
    text-align: center;
    padding: 1rem;
    color: var(--text-muted, #64748b);
    font-size: 0.8125rem;
  }

  .history-group {
    margin-bottom: 1rem;
  }

  .history-group-header {
    font-size: 0.6875rem;
    font-weight: 600;
    color: var(--text-muted, #64748b);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.5rem 0.5rem 0.25rem;
  }

  .history-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    width: 100%;
    padding: 0.5rem;
    background: none;
    border: none;
    border-radius: 0.375rem;
    color: var(--text-secondary, #94a3b8);
    cursor: pointer;
    text-align: left;
    font-size: 0.8125rem;
    transition: all 0.15s;
  }

  .history-item:hover {
    background: var(--bg-hover, #1e293b);
    color: var(--text-primary, #e2e8f0);
  }

  .history-item.active {
    background: var(--bg-hover, #1e293b);
    color: var(--accent-primary, #3b82f6);
  }

  .history-title {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .history-title-input {
    flex: 1;
    min-width: 0;
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid var(--border-color, #334155);
    border-radius: 4px;
    padding: 2px 6px;
    color: var(--text-primary, #e2e8f0);
    font-size: 0.75rem;
  }

  .session-action-btn {
    opacity: 0;
    background: none;
    border: none;
    color: var(--text-muted, #64748b);
    cursor: pointer;
    padding: 2px;
    display: flex;
    align-items: center;
  }

  .history-item:hover .session-action-btn {
    opacity: 1;
  }

  .session-action-btn:hover {
    color: var(--text-primary, #e2e8f0);
  }

  .delete-session-btn {
    opacity: 0;
    background: none;
    border: none;
    color: var(--text-muted, #64748b);
    cursor: pointer;
    padding: 2px;
    display: flex;
    align-items: center;
  }

  .history-item:hover .delete-session-btn {
    opacity: 1;
  }

  .delete-session-btn:hover {
    color: #ef4444;
  }

  .input-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 0.5rem;
  }

  .char-count {
    font-size: 0.625rem;
    color: var(--text-muted, #64748b);
  }

  .send-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: var(--accent-primary, #3b82f6);
    border: none;
    border-radius: 0.375rem;
    color: white;
    cursor: pointer;
    transition: all 0.15s;
  }

  .send-btn:hover:not(:disabled) {
    background: var(--accent-hover, #2563eb);
  }

  .send-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  /* Mail Compose Modal */
  .mail-compose-modal {
    background: var(--bg-secondary, #111827);
    border-bottom: 1px solid var(--border-color, #1e293b);
  }

  .mail-compose-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    background: var(--bg-tertiary, #1e293b);
    border-bottom: 1px solid var(--border-color, #334155);
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--accent-primary, #3b82f6);
  }

  .close-modal-btn {
    margin-left: auto;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    background: transparent;
    border: none;
    border-radius: 4px;
    color: var(--text-muted, #64748b);
    cursor: pointer;
  }

  .close-modal-btn:hover {
    background: var(--bg-hover, #334155);
    color: var(--text-primary, #e2e8f0);
  }

  .mail-compose-body {
    padding: 0.625rem 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.625rem;
  }

  .form-row {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .form-row label {
    font-size: 0.625rem;
    font-weight: 600;
    color: var(--text-muted, #64748b);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .dept-selector {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
  }

  .dept-option {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.25rem 0.5rem;
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.25rem;
    color: var(--text-secondary, #94a3b8);
    font-size: 0.6875rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  .dept-option:hover {
    border-color: var(--dept-color, #3b82f6);
  }

  .dept-option.selected {
    border-color: var(--dept-color, #3b82f6);
    background: color-mix(in srgb, var(--dept-color) 20%, var(--color-bg-elevated));
    color: var(--text-primary, #e2e8f0);
  }

  .dept-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--dept-color, #3b82f6);
  }

  .priority-selector {
    display: flex;
    gap: 0.25rem;
  }

  .priority-option {
    padding: 0.25rem 0.625rem;
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.25rem;
    color: var(--text-secondary, #94a3b8);
    font-size: 0.625rem;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.15s;
  }

  .priority-option:hover {
    border-color: var(--priority-color, #3b82f6);
  }

  .priority-option.selected {
    border-color: var(--priority-color, #3b82f6);
    background: color-mix(in srgb, var(--priority-color) 20%, var(--color-bg-elevated));
    color: var(--priority-color);
  }

  .form-row select,
  .form-row input,
  .form-row textarea {
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.25rem;
    padding: 0.375rem 0.5rem;
    color: var(--text-primary, #e2e8f0);
    font-size: 0.75rem;
    font-family: inherit;
  }

  .form-row select:focus,
  .form-row input:focus,
  .form-row textarea:focus {
    outline: none;
    border-color: var(--accent-primary, #3b82f6);
  }

  .form-row textarea {
    resize: vertical;
    min-height: 80px;
  }

  .mail-compose-footer {
    display: flex;
    justify-content: flex-end;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    background: var(--bg-tertiary, #1e293b);
    border-top: 1px solid var(--border-color, #334155);
  }

  .cancel-btn {
    padding: 0.375rem 0.75rem;
    background: var(--bg-secondary, #111827);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.25rem;
    color: var(--text-secondary, #94a3b8);
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  .cancel-btn:hover {
    background: var(--bg-hover, #334155);
  }

  .send-mail-btn {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.375rem 0.75rem;
    background: var(--accent-primary, #3b82f6);
    border: none;
    border-radius: 0.25rem;
    color: white;
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  .send-mail-btn:hover:not(:disabled) {
    background: var(--accent-hover, #2563eb);
  }

  .send-mail-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .mail-btn {
    position: relative;
  }

  /* Story 5.7: Confirmation Dialog */
  .confirmation-dialog {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 8px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .confirmation-content {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .confirmation-content .warning-icon {
    color: #f59e0b;
    flex-shrink: 0;
  }

  .confirmation-message {
    color: #fbbf24;
    font-size: 14px;
    font-weight: 500;
  }

  .confirmation-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
  }

  .confirm-btn {
    background: #f59e0b;
    color: #000;
    border: none;
    border-radius: 6px;
    padding: 6px 16px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: background 0.2s;
  }

  .confirm-btn:hover:not(:disabled) {
    background: #d97706;
  }

  .confirm-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .confirmation-dialog .cancel-btn {
    background: transparent;
    color: #94a3b8;
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .confirmation-dialog .cancel-btn:hover {
    background: rgba(255, 255, 255, 0.05);
    color: #e2e8f0;
  }

  /* ── Slash Commands Dropdown ──────────────────────────────────────────── */
  .slash-menu {
    position: absolute;
    bottom: calc(100% + 4px);
    left: 0;
    right: 0;
    background: rgba(8, 13, 20, 0.95);
    border: 1px solid #00d4ff;
    border-radius: 0.5rem;
    overflow: hidden;
    z-index: 200;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.5);
    max-height: 220px;
    overflow-y: auto;
  }

  .slash-item {
    display: flex;
    align-items: center;
    gap: 0.625rem;
    width: 100%;
    padding: 0.5rem 0.75rem;
    background: transparent;
    border: none;
    cursor: pointer;
    text-align: left;
    transition: background 0.1s;
  }

  .slash-item:hover,
  .slash-item.active {
    background: rgba(0, 212, 255, 0.08);
  }

  .slash-cmd {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8125rem;
    color: #00d4ff;
    font-weight: 500;
    flex-shrink: 0;
    min-width: 90px;
  }

  .slash-desc {
    font-size: 0.75rem;
    color: #64748b;
  }

  /* ── @ Asset Dropdown ────────────────────────────────────────────────── */
  .asset-menu {
    position: absolute;
    bottom: calc(100% + 4px);
    left: 0;
    right: 0;
    background: rgba(8, 13, 20, 0.95);
    border: 1px solid #00d4ff;
    border-radius: 0.5rem;
    overflow: hidden;
    z-index: 200;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.5);
    max-height: 220px;
    overflow-y: auto;
  }

  .asset-menu-loading,
  .asset-menu-empty {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.625rem 0.75rem;
    font-size: 0.75rem;
    color: #64748b;
  }

  .asset-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    width: 100%;
    padding: 0.5rem 0.75rem;
    background: transparent;
    border: none;
    cursor: pointer;
    text-align: left;
    transition: background 0.1s;
  }

  .asset-item:hover,
  .asset-item.active {
    background: rgba(0, 212, 255, 0.08);
  }

  .asset-name {
    font-size: 0.8125rem;
    color: #e2e8f0;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .asset-type-badge {
    font-size: 0.625rem;
    padding: 0.125rem 0.375rem;
    border-radius: 0.25rem;
    background: rgba(0, 212, 255, 0.12);
    border: 1px solid rgba(0, 212, 255, 0.25);
    color: #00d4ff;
    flex-shrink: 0;
    font-family: 'IBM Plex Mono', monospace;
  }

  /* ── Attachment Chips ────────────────────────────────────────────────── */
  .attachment-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.375rem;
    margin-top: 0.375rem;
  }

  .attachment-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.2rem 0.5rem;
    background: rgba(0, 212, 255, 0.12);
    border: 1px solid rgba(0, 212, 255, 0.3);
    border-radius: 0.375rem;
    font-size: 0.6875rem;
    color: #e2e8f0;
  }

  .chip-name {
    font-weight: 500;
    max-width: 120px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .chip-type {
    font-size: 0.625rem;
    color: #00d4ff;
    font-family: 'IBM Plex Mono', monospace;
  }

  .chip-remove {
    display: flex;
    align-items: center;
    justify-content: center;
    background: none;
    border: none;
    color: #64748b;
    cursor: pointer;
    padding: 0;
    margin-left: 0.125rem;
    transition: color 0.15s;
  }

  .chip-remove:hover {
    color: #ef4444;
  }

  /* ── Workflow Templates Section (Story C1) ───────────────────────────────── */
  .workflows-section {
    margin-bottom: 0.5rem;
  }

  .workflows-toggle {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    width: 100%;
    padding: 0.375rem 0.5rem;
    background: rgba(0, 212, 255, 0.06);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 0.375rem;
    color: rgba(224, 224, 224, 0.6);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  .workflows-toggle:hover {
    background: rgba(0, 212, 255, 0.1);
    border-color: rgba(0, 212, 255, 0.3);
    color: #e0e0e0;
  }

  .workflows-toggle-icon {
    font-size: 0.625rem;
    transition: transform 0.2s;
  }

  .workflows-toggle-icon.rotated {
    transform: rotate(90deg);
  }

  .workflows-count {
    margin-left: auto;
    padding: 0.125rem 0.375rem;
    background: rgba(0, 212, 255, 0.15);
    border: 1px solid rgba(0, 212, 255, 0.3);
    border-radius: 10px;
    font-size: 0.625rem;
    color: #00d4ff;
  }

  .workflows-list {
    display: flex;
    flex-direction: column;
    gap: 0.375rem;
    margin-top: 0.5rem;
    padding: 0.5rem;
    background: rgba(8, 13, 20, 0.5);
    backdrop-filter: blur(12px) saturate(160%);
    -webkit-backdrop-filter: blur(12px) saturate(160%);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 0.5rem;
  }

  .workflows-loading {
    padding: 0.5rem;
    text-align: center;
    color: rgba(224, 224, 224, 0.4);
    font-size: 0.75rem;
  }

  .workflow-item {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    padding: 0.5rem 0.625rem;
    background: rgba(0, 212, 255, 0.05);
    border: 1px solid rgba(0, 212, 255, 0.12);
    border-radius: 0.375rem;
    cursor: pointer;
    text-align: left;
    transition: all 0.15s;
  }

  .workflow-item:hover {
    background: rgba(0, 212, 255, 0.12);
    border-color: rgba(0, 212, 255, 0.35);
    transform: translateY(-1px);
  }

  .workflow-name {
    font-size: 0.8125rem;
    font-weight: 500;
    color: #00d4ff;
  }

  .workflow-meta {
    display: flex;
    gap: 0.5rem;
    font-size: 0.625rem;
    color: rgba(224, 224, 224, 0.4);
  }

  .workflow-duration {
    font-family: 'IBM Plex Mono', monospace;
  }

  .workflow-depts {
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }
</style>

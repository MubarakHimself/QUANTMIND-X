<script lang="ts">
  /**
   * AgentThoughtsPanel — SSE-driven real-time agent thought feed.
   *
   * Connects to /api/agent-thoughts/stream, preloads history,
   * and renders a filterable ring-buffer of up to 200 thoughts.
   */
  import { onMount, onDestroy, tick } from 'svelte';
  import { Brain, Zap, Eye, Target, Trash2, Circle } from 'lucide-svelte';
  import { API_CONFIG } from '$lib/config/api';

  // ── Types ────────────────────────────────────────────────────────────────

  type ThoughtType = 'reasoning' | 'action' | 'observation' | 'decision';
  type Department =
    | 'research'
    | 'development'
    | 'risk'
    | 'trading'
    | 'portfolio'
    | 'floormanager'
    | 'all';

  interface Thought {
    id: string;
    type: 'thought';
    department: string;
    content: string;
    thought_type: ThoughtType;
    timestamp: string;
    session_id: string;
  }

  interface Props {
    sessionId?: string;
    maxHeight?: string;
    showHeader?: boolean;
  }

  // ── Props ────────────────────────────────────────────────────────────────

  const {
    sessionId = '',
    maxHeight = '400px',
    showHeader = true
  }: Props = $props();

  // ── Constants ────────────────────────────────────────────────────────────

  const API_BASE = API_CONFIG.API_BASE;
  const MAX_THOUGHTS = 200;
  const RECONNECT_DELAY_MS = 3000;

  const DEPARTMENTS: { label: string; value: Department }[] = [
    { label: 'All', value: 'all' },
    { label: 'Research', value: 'research' },
    { label: 'Development', value: 'development' },
    { label: 'Risk', value: 'risk' },
    { label: 'Trading', value: 'trading' },
    { label: 'Portfolio', value: 'portfolio' },
    { label: 'FloorManager', value: 'floormanager' }
  ];

  // ── State ────────────────────────────────────────────────────────────────

  let thoughts = $state<Thought[]>([]);
  let activeFilter = $state<Department>('all');
  let connected = $state(false);
  let feedEnd = $state<HTMLElement | null>(null);

  let eventSource: EventSource | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let destroyed = false;

  // ── Derived ──────────────────────────────────────────────────────────────

  const filteredThoughts = $derived(
    activeFilter === 'all'
      ? thoughts
      : thoughts.filter(
          t => t.department.toLowerCase() === activeFilter.toLowerCase()
        )
  );

  // ── Helpers ──────────────────────────────────────────────────────────────

  function addThought(t: Thought) {
    thoughts = thoughts.length >= MAX_THOUGHTS
      ? [...thoughts.slice(thoughts.length - MAX_THOUGHTS + 1), t]
      : [...thoughts, t];
  }

  function formatTime(iso: string): string {
    try {
      return new Date(iso).toLocaleTimeString('en-GB', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      });
    } catch {
      return '';
    }
  }

  // ── SSE connection ───────────────────────────────────────────────────────

  function connect() {
    if (destroyed) return;

    const sid = sessionId || 'default';
    const url = `${API_BASE}/agent-thoughts/stream?session_id=${encodeURIComponent(sid)}`;

    try {
      eventSource = new EventSource(url);

      eventSource.onopen = () => {
        connected = true;
      };

      eventSource.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data) as Thought;
          if (data.type === 'thought') {
            addThought({ ...data, id: data.id ?? crypto.randomUUID() });
            tick().then(() => {
              feedEnd?.scrollIntoView({ behavior: 'smooth' });
            });
          }
        } catch {
          // ignore malformed events
        }
      };

      eventSource.onerror = () => {
        connected = false;
        eventSource?.close();
        eventSource = null;
        scheduleReconnect();
      };
    } catch {
      connected = false;
      scheduleReconnect();
    }
  }

  function scheduleReconnect() {
    if (destroyed) return;
    reconnectTimer = setTimeout(() => {
      if (!destroyed) connect();
    }, RECONNECT_DELAY_MS);
  }

  function disconnect() {
    if (reconnectTimer !== null) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
    connected = false;
  }

  // ── History preload ──────────────────────────────────────────────────────

  async function loadHistory() {
    try {
      const sid = sessionId || 'default';
      const res = await fetch(
        `${API_BASE}/agent-thoughts/history?session_id=${encodeURIComponent(sid)}&limit=50`
      );
      if (!res.ok) return;
      const items: Thought[] = await res.json();
      // Seed thoughts without exceeding ring buffer
      const capped = items.slice(-MAX_THOUGHTS);
      thoughts = capped.map(t => ({ ...t, id: t.id ?? crypto.randomUUID() }));
      await tick();
      feedEnd?.scrollIntoView({ behavior: 'instant' });
    } catch {
      // history load is best-effort
    }
  }

  // ── Lifecycle ────────────────────────────────────────────────────────────

  onMount(async () => {
    await loadHistory();
    connect();
  });

  onDestroy(() => {
    destroyed = true;
    disconnect();
  });

  // ── Actions ──────────────────────────────────────────────────────────────

  function clearThoughts() {
    thoughts = [];
  }

  function setFilter(dept: Department) {
    activeFilter = dept;
  }
</script>

<div class="thoughts-panel" style:height={maxHeight}>

  <!-- Header -->
  {#if showHeader}
    <div class="panel-header">
      <div class="header-left">
        <Brain size={14} />
        <span class="panel-title">Agent Thoughts</span>
      </div>
      <div class="header-right">
        <button class="clear-btn" onclick={clearThoughts} title="Clear thoughts">
          <Trash2 size={12} />
        </button>
        <div class="status-dot" class:connected title={connected ? 'Connected' : 'Disconnected'}>
          <Circle size={8} />
        </div>
      </div>
    </div>
  {/if}

  <!-- Department filter pills -->
  <div class="filter-bar">
    {#each DEPARTMENTS as dept}
      <button
        class="dept-pill"
        class:active={activeFilter === dept.value}
        onclick={() => setFilter(dept.value)}
      >
        {dept.label}
      </button>
    {/each}
  </div>

  <!-- Thought feed -->
  <div class="feed">
    {#if filteredThoughts.length === 0}
      <div class="empty-state">
        <Brain size={28} />
        <span>No agent thoughts yet</span>
      </div>
    {:else}
      {#each filteredThoughts as thought (thought.id)}
        <div class="thought-item {thought.thought_type}">
          <div class="thought-meta">
            <span class="thought-icon">
              {#if thought.thought_type === 'reasoning'}
                <Brain size={11} />
              {:else if thought.thought_type === 'action'}
                <Zap size={11} />
              {:else if thought.thought_type === 'observation'}
                <Eye size={11} />
              {:else}
                <Target size={11} />
              {/if}
            </span>
            <span class="dept-badge">{thought.department}</span>
            <span class="type-label">{thought.thought_type}</span>
            <span class="timestamp">{formatTime(thought.timestamp)}</span>
          </div>
          <p class="thought-content">{thought.content}</p>
        </div>
      {/each}
      <div bind:this={feedEnd}></div>
    {/if}
  </div>

</div>

<style>
  /* ── Shell ────────────────────────────────────────────────────────────── */
  .thoughts-panel {
    display: flex;
    flex-direction: column;
    background: rgba(8, 13, 20, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    overflow: hidden;
    min-height: 0;
  }

  /* ── Header ──────────────────────────────────────────────────────────── */
  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    flex-shrink: 0;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 7px;
    color: #94a3b8;
  }

  .panel-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #64748b;
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .clear-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    background: transparent;
    border: none;
    color: #475569;
    cursor: pointer;
    padding: 3px;
    border-radius: 4px;
    transition: color 0.12s, background 0.12s;
  }

  .clear-btn:hover {
    color: #f87171;
    background: rgba(248, 113, 113, 0.08);
  }

  /* Status dot */
  .status-dot {
    display: flex;
    align-items: center;
    color: #475569;
    transition: color 0.2s;
  }

  .status-dot.connected {
    color: #00c896;
  }

  /* ── Filter bar ──────────────────────────────────────────────────────── */
  .filter-bar {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 7px 10px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    overflow-x: auto;
    flex-shrink: 0;
  }

  .filter-bar::-webkit-scrollbar { height: 3px; }
  .filter-bar::-webkit-scrollbar-track { background: transparent; }
  .filter-bar::-webkit-scrollbar-thumb {
    background: rgba(0, 212, 255, 0.12);
    border-radius: 2px;
  }

  .dept-pill {
    padding: 3px 9px;
    background: transparent;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    color: rgba(255, 255, 255, 0.35);
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    cursor: pointer;
    transition: background 0.12s, border-color 0.12s, color 0.12s;
    white-space: nowrap;
    flex-shrink: 0;
  }

  .dept-pill:hover {
    background: rgba(0, 212, 255, 0.07);
    border-color: rgba(0, 212, 255, 0.2);
    color: #94a3b8;
  }

  .dept-pill.active {
    background: rgba(0, 212, 255, 0.13);
    border-color: rgba(0, 212, 255, 0.35);
    color: #00d4ff;
  }

  /* ── Feed ────────────────────────────────────────────────────────────── */
  .feed {
    flex: 1;
    overflow-y: auto;
    min-height: 0;
  }

  .feed::-webkit-scrollbar { width: 4px; }
  .feed::-webkit-scrollbar-track { background: transparent; }
  .feed::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 2px;
  }

  /* ── Empty state ─────────────────────────────────────────────────────── */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    height: 100%;
    min-height: 120px;
    color: #334155;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }

  /* ── Thought item ────────────────────────────────────────────────────── */
  .thought-item {
    padding: 10px 14px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.5;
    border-left: 2px solid transparent;
    transition: background 0.1s;
  }

  .thought-item:hover {
    background: rgba(255, 255, 255, 0.02);
  }

  .thought-item.reasoning { border-left-color: #00d4ff; }
  .thought-item.action    { border-left-color: #f0a500; }
  .thought-item.observation { border-left-color: #00c896; }
  .thought-item.decision  { border-left-color: #ff3b3b; }

  /* ── Thought meta row ────────────────────────────────────────────────── */
  .thought-meta {
    display: flex;
    align-items: center;
    gap: 7px;
    margin-bottom: 5px;
  }

  .thought-icon {
    display: flex;
    align-items: center;
    flex-shrink: 0;
  }

  .thought-item.reasoning .thought-icon { color: #00d4ff; }
  .thought-item.action    .thought-icon { color: #f0a500; }
  .thought-item.observation .thought-icon { color: #00c896; }
  .thought-item.decision  .thought-icon { color: #ff3b3b; }

  .dept-badge {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 2px 6px;
    border-radius: 3px;
    background: rgba(255, 255, 255, 0.08);
    color: rgba(255, 255, 255, 0.5);
    flex-shrink: 0;
  }

  .type-label {
    font-size: 10px;
    color: #334155;
    text-transform: lowercase;
    flex-shrink: 0;
  }

  .timestamp {
    font-size: 10px;
    color: #334155;
    margin-left: auto;
    flex-shrink: 0;
  }

  /* ── Thought content ─────────────────────────────────────────────────── */
  .thought-content {
    margin: 0;
    color: #94a3b8;
    word-break: break-word;
  }
</style>

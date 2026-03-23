<script lang="ts">
  import {
    Bell,
    CheckCircle,
    AlertTriangle,
    XCircle,
    TrendingUp,
    Bot,
  } from 'lucide-svelte';
  import {
    notificationStore,
    unreadCount,
    markRead,
    markAllRead,
    clearAll,
  } from '$lib/stores/notifications';
  import type { AppNotification } from '$lib/stores/notifications';
  import { activeCanvasStore } from '$lib/stores/canvasStore';

  let { open, onClose }: { open: boolean; onClose: () => void } = $props();

  /** Reverse-chronological list */
  let notifications = $derived([...$notificationStore].sort(
    (a, b) => b.timestamp.getTime() - a.timestamp.getTime()
  ));

  function getIcon(type: AppNotification['type']) {
    switch (type) {
      case 'success': return CheckCircle;
      case 'warning': return AlertTriangle;
      case 'error':   return XCircle;
      case 'trade':   return TrendingUp;
      case 'agent':   return Bot;
      default:        return Bell;
    }
  }

  function getIconColor(type: AppNotification['type']): string {
    switch (type) {
      case 'success': return '#00c896';
      case 'warning': return '#f0a500';
      case 'error':   return '#ff3b3b';
      case 'trade':   return '#f0a500';
      case 'agent':   return '#00d4ff';
      default:        return 'rgba(255,255,255,0.45)';
    }
  }

  function relativeTime(date: Date): string {
    const diffMs = Date.now() - date.getTime();
    const mins = Math.floor(diffMs / 60_000);
    if (mins < 1) return 'just now';
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  }

  function handleRowClick(n: AppNotification) {
    markRead(n.id);
    if (n.canvasLink) {
      activeCanvasStore.setActiveCanvas(n.canvasLink);
    }
    onClose();
  }

  function handleOverlayClick() {
    onClose();
  }

  function handleMarkAll(e: MouseEvent) {
    e.stopPropagation();
    markAllRead();
  }

  function handleClearAll(e: MouseEvent) {
    e.stopPropagation();
    clearAll();
  }
</script>

{#if open}
  <!-- Transparent overlay to close on outside click -->
  <div class="overlay" onclick={handleOverlayClick} role="presentation"></div>

  <!-- Tray panel -->
  <div class="tray" role="dialog" aria-label="Notifications">
    <!-- Header -->
    <div class="tray-header">
      <div class="header-left">
        <span class="header-title">Notifications</span>
        {#if $unreadCount > 0}
          <span class="unread-badge">{$unreadCount}</span>
        {/if}
      </div>
      <div class="header-actions">
        <button class="hdr-btn mark-btn" onclick={handleMarkAll}>Mark all read</button>
        <button class="hdr-btn clear-btn" onclick={handleClearAll}>Clear all</button>
      </div>
    </div>

    <!-- Notification list -->
    <div class="tray-body">
      {#if notifications.length === 0}
        <div class="empty-state">
          <Bell size={18} strokeWidth={1.5} />
          <span>No notifications</span>
        </div>
      {:else}
        {#each notifications as notif (notif.id)}
          {@const IconComponent = getIcon(notif.type)}
          {@const iconColor = getIconColor(notif.type)}
          <button
            class="notif-row"
            class:unread={!notif.read}
            onclick={() => handleRowClick(notif)}
          >
            <div class="notif-icon" style="color: {iconColor}">
              <IconComponent size={14} strokeWidth={2} />
            </div>
            <div class="notif-content">
              <span class="notif-title">{notif.title}</span>
              {#if notif.body}
                <span class="notif-body">{notif.body}</span>
              {/if}
            </div>
            <div class="notif-time">{relativeTime(notif.timestamp)}</div>
          </button>
        {/each}
      {/if}
    </div>
  </div>
{/if}

<style>
  /* Overlay — sits behind the tray, captures outside clicks */
  .overlay {
    position: fixed;
    inset: 0;
    z-index: 499;
    background: transparent;
  }

  /* Tray panel */
  .tray {
    position: fixed;
    top: 48px;
    right: 14px;
    width: 340px;
    max-height: 480px;
    overflow-y: auto;
    z-index: 500;
    background: rgba(8, 13, 20, 0.92);
    backdrop-filter: blur(24px) saturate(160%);
    -webkit-backdrop-filter: blur(24px) saturate(160%);
    border: 1px solid rgba(0, 212, 255, 0.12);
    border-radius: 10px;
    display: flex;
    flex-direction: column;
    /* Thin scrollbar */
    scrollbar-width: thin;
    scrollbar-color: rgba(0, 212, 255, 0.15) transparent;
  }

  .tray::-webkit-scrollbar {
    width: 4px;
  }

  .tray::-webkit-scrollbar-thumb {
    background: rgba(0, 212, 255, 0.15);
    border-radius: 2px;
  }

  /* Header */
  .tray-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 12px 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    flex-shrink: 0;
    position: sticky;
    top: 0;
    background: rgba(8, 13, 20, 0.95);
    backdrop-filter: blur(24px);
    z-index: 1;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .header-title {
    font-family: var(--font-display, 'Syne', 'Space Grotesk', sans-serif);
    font-size: 13px;
    font-weight: 700;
    color: var(--color-text-primary, #e8eaf0);
    letter-spacing: 0.02em;
  }

  .unread-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 16px;
    height: 16px;
    padding: 0 4px;
    background: rgba(0, 212, 255, 0.18);
    color: #00d4ff;
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 10px;
    font-weight: 700;
    border-radius: 8px;
    border: 1px solid rgba(0, 212, 255, 0.3);
    line-height: 1;
  }

  .header-actions {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .hdr-btn {
    background: transparent;
    border: none;
    padding: 0;
    cursor: pointer;
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 11px;
    color: var(--color-text-muted, rgba(255,255,255,0.35));
    transition: color 0.12s ease;
  }

  .mark-btn:hover {
    color: #00d4ff;
  }

  .clear-btn:hover {
    color: #ff3b3b;
  }

  /* Body */
  .tray-body {
    display: flex;
    flex-direction: column;
  }

  /* Empty state */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 32px 16px;
    color: var(--color-text-muted, rgba(255,255,255,0.3));
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 12px;
  }

  /* Notification row */
  .notif-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 9px 12px;
    border: none;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    background: transparent;
    cursor: pointer;
    text-align: left;
    width: 100%;
    transition: background 0.1s ease;
  }

  .notif-row:last-child {
    border-bottom: none;
  }

  .notif-row.unread {
    background: rgba(0, 212, 255, 0.04);
  }

  .notif-row:hover {
    background: rgba(255, 255, 255, 0.04);
  }

  .notif-row.unread:hover {
    background: rgba(0, 212, 255, 0.07);
  }

  /* Icon column */
  .notif-icon {
    flex-shrink: 0;
    margin-top: 1px;
    display: flex;
    align-items: center;
  }

  /* Content column */
  .notif-content {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .notif-title {
    font-family: var(--font-nav, 'Space Grotesk', sans-serif);
    font-size: 13px;
    font-weight: 600;
    color: var(--color-text-primary, #e8eaf0);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.3;
  }

  .notif-body {
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 11px;
    color: var(--color-text-muted, rgba(255,255,255,0.4));
    line-height: 1.4;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
  }

  /* Timestamp column */
  .notif-time {
    flex-shrink: 0;
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 10px;
    color: var(--color-text-muted, rgba(255,255,255,0.3));
    margin-top: 2px;
    white-space: nowrap;
  }
</style>

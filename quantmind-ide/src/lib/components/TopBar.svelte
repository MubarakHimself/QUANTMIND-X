<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { Power, Settings, Minus, Square, X } from 'lucide-svelte';
  
  const dispatch = createEventDispatcher();
  
  let killSwitchArmed = false;
  
  async function toggleKillSwitch() {
    if (!killSwitchArmed) {
      // First click arms it
      killSwitchArmed = true;
      setTimeout(() => { killSwitchArmed = false; }, 5000); // Auto-disarm after 5s
    } else {
      // Second click triggers
      try {
        await fetch('http://localhost:8000/api/trading/kill', { method: 'POST' });
        dispatch('killTriggered');
      } catch (e) {
        console.error('Kill switch failed:', e);
      }
      killSwitchArmed = false;
    }
  }
  
  function openSettings() {
    dispatch('openSettings');
  }
</script>

<header class="top-bar">
  <div class="left-section">
    <div class="logo">
      <span class="logo-text">QuantMind</span>
    </div>
    
    <nav class="menu-bar">
      <button class="menu-item">File</button>
      <button class="menu-item">Edit</button>
      <button class="menu-item">View</button>
      <button class="menu-item">Tools</button>
      <button class="menu-item">Help</button>
    </nav>
  </div>
  
  <div class="right-section">
    <button 
      class="settings-btn" 
      on:click={openSettings}
      title="System Settings"
    >
      <Settings size={16} />
    </button>
    
    <button 
      class="kill-switch" 
      class:armed={killSwitchArmed}
      on:click={toggleKillSwitch}
      title={killSwitchArmed ? 'Click again to KILL ALL' : 'Emergency Kill Switch'}
    >
      <Power size={16} />
      <span>{killSwitchArmed ? 'CONFIRM KILL' : 'Kill Switch'}</span>
    </button>
    
    <div class="window-controls">
      <button class="window-btn minimize"><Minus size={14} /></button>
      <button class="window-btn maximize"><Square size={12} /></button>
      <button class="window-btn close"><X size={14} /></button>
    </div>
  </div>
</header>

<style>
  .top-bar {
    grid-area: topbar;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 36px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
    padding: 0 8px;
    -webkit-app-region: drag;
  }
  
  .left-section, .right-section {
    display: flex;
    align-items: center;
    gap: 8px;
    -webkit-app-region: no-drag;
  }
  
  .logo {
    display: flex;
    align-items: center;
    padding: 0 12px;
  }
  
  .logo-text {
    font-size: 13px;
    font-weight: 600;
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  
  .menu-bar {
    display: flex;
    gap: 2px;
  }
  
  .menu-item {
    padding: 4px 10px;
    background: transparent;
    border: none;
    border-radius: 4px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.1s ease;
  }
  
  .menu-item:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }
  
  .settings-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--text-muted);
    cursor: pointer;
  }
  
  .settings-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }
  
  .kill-switch {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    background: rgba(255, 60, 60, 0.1);
    border: 1px solid rgba(255, 60, 60, 0.3);
    border-radius: 6px;
    color: #ff6b6b;
    font-size: 11px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  
  .kill-switch:hover {
    background: rgba(255, 60, 60, 0.2);
    border-color: rgba(255, 60, 60, 0.5);
  }
  
  .kill-switch.armed {
    background: #ff3c3c;
    border-color: #ff3c3c;
    color: white;
    animation: pulse 0.5s ease infinite alternate;
  }
  
  @keyframes pulse {
    from { opacity: 1; }
    to { opacity: 0.7; }
  }
  
  .window-controls {
    display: flex;
    gap: 2px;
    margin-left: 8px;
  }
  
  .window-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: transparent;
    border: none;
    border-radius: 4px;
    color: var(--text-muted);
    cursor: pointer;
  }
  
  .window-btn:hover {
    background: var(--bg-tertiary);
  }
  
  .window-btn.close:hover {
    background: #ff3c3c;
    color: white;
  }
</style>

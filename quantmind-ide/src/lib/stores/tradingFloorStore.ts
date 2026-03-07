/**
 * Trading Floor Store
 *
 * State management for the Trading Floor visualization.
 * Uses Svelte stores for reactive state management.
 */

import { writable, derived, get } from 'svelte/store';

export type AgentState = {
  id: string;
  name: string;
  department: string;
  status: 'idle' | 'thinking' | 'typing' | 'reading' | 'walking' | 'spawning';
  position: { x: number; y: number };
  target: { x: number; y: number } | null;
  speechBubble?: {
    text: string;
    type: 'thinking' | 'result' | 'question' | 'error';
    duration: number;
  };
  subAgents: AgentState[];
  isExpanded: boolean;
}

export type DepartmentPersonality = {
  name: string;
  tagline: string;
  traits: string[];
  communication_style: string;
  strengths: string[];
  weaknesses: string[];
  color: string;
  icon: string;
}

export type DepartmentPosition = {
  x: number;
  y: number;
  width: number;
  height: number;
  personality?: DepartmentPersonality;
}

export type MailMessage = {
  id: string;
  fromDept: string;
  toDept: string;
  type: 'dispatch' | 'result' | 'question' | 'status';
  subject: string;
  startX: number;
  startY: number;
  progress: number;
  duration: number;
}

export type TradingFloorState = {
  totalTasks: number;
  activeTasks: number;
  completedTasks: number;
  pendingMail: number;
}

// Store state
const agents = writable(new Map<string, AgentState>());
const departments = writable(new Map<string, DepartmentPosition>());
const mailMessages = writable<MailMessage[]>([]);
const animatingMail = writable<MailMessage | null>(null);
const selectedAgent = writable<string | null>(null);
const floorStats = writable<TradingFloorState>({
  totalTasks: 0,
  activeTasks: 0,
  completedTasks: 0,
  pendingMail: 0,
});

// Initialize Floor Manager
const floorManager: AgentState = {
  id: 'floor-manager',
  name: 'Floor Manager',
  department: 'coordination',
  status: 'idle',
  position: { x: 450, y: 10 },
  target: null,
  subAgents: [],
  isExpanded: false,
};

agents.update((state) => {
  state.set(floorManager.id, floorManager);
  return state;
});

// Department personality data - matches backend personalities
const DEPARTMENT_PERSONALITIES: Record<string, DepartmentPersonality> = {
  development: {
    name: 'The Data Detective',
    tagline: 'Meticulous analysis reveals hidden truths',
    traits: ['analytical', 'detail-oriented', 'thorough', 'methodical'],
    communication_style: 'Precise and data-driven, citing specific metrics and indicators',
    strengths: ['Pattern recognition', 'Statistical analysis', 'Root cause discovery'],
    weaknesses: ['Analysis paralysis', 'May miss big picture', 'Over-reliance on historical data'],
    color: '#3b82f6',
    icon: 'search',
  },
  research: {
    name: 'The Innovation Pioneer',
    tagline: "Tomorrow's alpha is discovered today",
    traits: ['curious', 'exploratory', 'innovative', 'hypothesis-driven'],
    communication_style: 'Excited and forward-thinking, exploring what could be',
    strengths: ['Alpha discovery', 'Novel strategy development', 'Out-of-the-box thinking'],
    weaknesses: ['May pursue dead ends', 'Theoretical bias', 'Implementation gaps'],
    color: '#8b5cf6',
    icon: 'lightbulb',
  },
  risk: {
    name: 'The Guardian',
    tagline: 'Protecting capital through vigilance',
    traits: ['cautious', 'protective', 'vigilant', 'systematic'],
    communication_style: 'Alert and conservative, emphasizing downside protection',
    strengths: ['Risk assessment', 'Drawdown prevention', 'Capital preservation'],
    weaknesses: ['May block opportunities', 'Conservative bias', 'Analysis overhead'],
    color: '#ef4444',
    icon: 'shield',
  },
  trading: {
    name: 'The Precision Tactician',
    tagline: 'Precision in execution, speed in action',
    traits: ['decisive', 'efficient', 'action-oriented', 'reliable'],
    communication_style: 'Direct and action-focused, emphasizing execution quality',
    strengths: ['Order execution', 'Fill optimization', 'Trade management'],
    weaknesses: ['Limited strategic view', 'Reactive rather than proactive', 'Execution dependency'],
    color: '#f97316',
    icon: 'zap',
  },
  portfolio: {
    name: 'The Strategic Architect',
    tagline: 'Building wealth through balanced allocation',
    traits: ['holistic', 'balanced', 'long-term', 'strategic'],
    communication_style: 'Comprehensive and big-picture focused, emphasizing diversification',
    strengths: ['Portfolio optimization', 'Allocation decisions', 'Performance attribution'],
    weaknesses: ['May underreact to opportunities', 'Complex implementation', 'Rebalancing costs'],
    color: '#10b981',
    icon: 'pie-chart',
  },
};

// Initialize departments
const initDepartments = () => {
  const deptPositions: Record<string, DepartmentPosition> = {
    development: { x: 100, y: 80, width: 140, height: 100, personality: DEPARTMENT_PERSONALITIES.development },
    research: { x: 300, y: 80, width: 140, height: 100, personality: DEPARTMENT_PERSONALITIES.research },
    risk: { x: 500, y: 80, width: 140, height: 100, personality: DEPARTMENT_PERSONALITIES.risk },
    trading: { x: 700, y: 80, width: 140, height: 100, personality: DEPARTMENT_PERSONALITIES.trading },
    portfolio: { x: 900, y: 80, width: 140, height: 100, personality: DEPARTMENT_PERSONALITIES.portfolio },
  };

  departments.set(new Map(Object.entries(deptPositions)));
};

// Derived stores
export const agentList = derived(agents, ($agents) => Array.from($agents.values()));
export const departmentList = derived(departments, ($departments) => Array.from($departments.values()));
export const activeAgentCount = derived(
  [agents, selectedAgent],
  ($agents) => Array.from($agents.values()).filter(a => a.status !== 'idle').length
);

// Actions
export function updateAgentState(id: string, updates: Partial<AgentState>) {
  agents.update((state) => {
    const agent = state.get(id);
    if (agent) {
      const newAgent = { ...agent, ...updates };
      state.set(id, newAgent);

      // Update stats
      const activeCount = Array.from(state.values()).filter(a => a.status !== 'idle').length;
      floorStats.update((stats) => ({
        ...stats,
        totalTasks: state.size,
        activeTasks: activeCount,
      }));
    }
    return state;
  });
}

export function addSubAgent(parentId: string, subAgent: AgentState) {
  agents.update((state) => {
    const parent = state.get(parentId);
    if (parent) {
      const newSubAgents = [...parent.subAgents, subAgent];
      state.set(parentId, { ...parent, subAgents: newSubAgents });
      state.set(subAgent.id, subAgent);

      floorStats.update((stats) => ({
        ...stats,
        totalTasks: stats.totalTasks + 1,
        activeTasks: stats.activeTasks + 1,
      }));
    }
    return state;
  });
}

export function sendMail(message: MailMessage) {
  mailMessages.update((messages) => [...messages, message]);
  animatingMail.set(message);
  floorStats.update((stats) => ({
    ...stats,
    pendingMail: stats.pendingMail + 1,
  }));
}

export function completeMailAnimation() {
  animatingMail.set(null);
}

export function selectAgent(id: string) {
  selectedAgent.set(id);
}

export function clearSelection() {
  selectedAgent.set(null);
}

export function reset() {
  agents.set(new Map());
  departments.set(new Map());
  mailMessages.set([]);
  animatingMail.set(null);
  selectedAgent.set(null);
  floorStats.set({
    totalTasks: 0,
    activeTasks: 0,
    completedTasks: 0,
    pendingMail: 0,
  });

  initDepartments();

  // Re-initialize floor manager
  agents.update((state) => {
    state.set(floorManager.id, floorManager);
    return state;
  });
}

// Derived combined store for convenience
export const tradingFloorStore = derived(
  [agents, departments, floorStats],
  ([$agents, $departments, $floorStats]) => ({
    agents: $agents,
    departments: $departments,
    floorStats: $floorStats,
  })
);

// Export stores for direct access
export {
  agents,
  departments,
  mailMessages,
  animatingMail,
  selectedAgent,
  floorStats,
};

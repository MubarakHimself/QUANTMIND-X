// Legacy exports (deprecated - will be removed in future version)
// These use LangChain/LangGraph and are maintained for backward compatibility
export { agentManager, AgentManager, type AgentState, type AgentConfig, type AIProvider, PROVIDER_CONFIGS, getProviderForAgent } from './agentManager';
export { memoryManager, HybridMemoryManager } from './memoryManager';
export { createModel, createQuantMindXAgent, createAgentManager, createQuantMindXTools, MultiProviderAgentManager, type ProviderConfig } from './langchainAgent';

// V2 Claude Agent Client (new)
export {
  AgentClient,
  getAgentClient,
  runAgent,
  streamAgent,
  type AgentMessage,
  type AgentContext,
  type RunAgentRequest,
  type RunAgentResponse,
  type AgentTask,
  type ToolCall,
  type AgentResult,
  type WebSocketEventType,
  type WebSocketEvent,
  type ProgressData,
} from './claudeCodeAgent';

// Agent Stream Store (new)
export {
  agentStreamStore,
  activeStreams,
  completedStreams,
  latestTask,
  streamToStore,
  type TaskStreamState,
  type StreamStateMap,
} from './agentStreamStore';

// Skills System
export {
  skillRegistry,
  executeSkill,
  getSkillsAsTools,
  copilotSkills,
  analystSkills,
  quantcodeSkills,
  type Skill,
  type SkillSchema,
  type SkillResult,
  type SkillExample,
  type SkillContext
} from './skills';


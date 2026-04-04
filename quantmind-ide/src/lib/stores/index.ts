// Stores Export Index
export {
  chatStore,
  activeChat,
  activeMessages,
  activeContext,
  pinnedChats,
  unpinnedChats,
  agentChats,
  agentGreetings
} from './chatStore';

export {
  settingsStore,
  connectedMCPServers,
  enabledRules,
  workflowTemplates,
  permissionPresets
} from './settingsStore';

// Workflow Stores
export {
  workflowStore,
  activeWorkflows,
  workflowHistory,
  selectedWorkflow,
  hasActiveWorkflows,
  loading,
  error
} from './workflowStore';
export type {
  Workflow,
  WorkflowStep,
  WorkflowSummary,
  WorkflowStatus,
  StepStatus
} from './workflowStore';

// Memory Management Stores
export { memoryStore, memories, filteredMemories, selectedMemory, memoryStats, memoryLoading, memoryError, memoryFilters } from './memoryStore';
export { cronStore, cronJobs, enabledCronJobs, cronLoading, cronError } from './cronStore';
export { hooksStore, hooks, enabledHooks, hookLogs, hooksLoading } from './hooksStore';

// Re-export types from chatStore
export type {
  Chat,
  Message,
  ChatContext,
  FileReference,
  StrategyReference,
  BrokerReference,
  BacktestReference,
  AgentType
} from './chatStore';

// Re-export types from settingsTypes (via settingsStore)
export type {
  SettingsStoreState,
  GeneralSettings,
  APIKeys,
  MCPServer,
  Skill,
  AgentSkills,
  Rule,
  MemoryConfig,
  MemoryStats,
  Workflow,
  WorkflowStep,
  AgentPermissions
} from './settingsTypes';

// Re-export types from memory management stores
export type { MemoryEntry, MemoryFilters } from './memoryStore';
export type { CronJob } from './cronStore';
export type { Hook, HookLogEntry } from './hooksStore';

// Re-export Trading Floor types and store
export { tradingFloorStore } from './tradingFloorStore';
export type {
  AgentState,
  DepartmentPosition,
  MailMessage,
  TradingFloorState
} from './tradingFloorStore';

// Account Store
export { accountStore, accounts, activeAccount, accountLoading, accountError, connectedAccounts } from './accountStore';
export type { BrokerAccount } from './accountStore';

// Live Trading Store
export {
  wsConnected,
  wsError,
  activeBots,
  selectedBotId,
  botDetails,
  pnlFlash,
  isLoading,
  selectedBot,
  botCount,
  connectTradingWS,
  disconnectTradingWS,
  selectBot,
  loadBotDetails,
  fetchActiveBots,
  closePosition,
  closeAllPositions,
  closeLoading,
  closeError
} from './trading';
export type { Regime, BotStatus, BotDetail, TradingUpdate, PositionInfo, CloseResult, CloseAllResult } from './trading';

// Kill Switch Store
export {
  killSwitchState,
  killSwitchCountdown,
  selectedTier,
  showKillSwitchModal,
  showEmergencyCloseModal,
  killSwitchFired,
  killSwitchLoading,
  killSwitchError,
  killSwitchAriaLabel,
  TIER_DESCRIPTIONS,
  armKillSwitch,
  disarmKillSwitch,
  cancelKillSwitch,
  selectTier,
  triggerKillSwitch,
  confirmKillSwitch,
  fetchKillSwitchStatus
} from './kill-switch';
export type { KillSwitchState, KillSwitchTier } from './kill-switch';

// Node Health Store (Story 3-7: MorningDigestCard & Degraded Mode)
export {
  nodeHealthState,
  isContaboDegraded,
  isCloudzyConnected,
  systemDegraded,
  isCloudzyDegraded,
  checkNodeHealth,
  startHealthMonitoring,
  stopHealthMonitoring,
  updateLastKnownData,
  resetNodeHealth
} from './node-health';
export type { NodeStatus, NodeHealth, NodeHealthState } from './node-health';

// Risk Physics Sensor Store (Story 4-5: Risk Canvas Physics Sensor Tiles)
export {
  physicsSensorStore,
  isingData,
  lyapunovData,
  hmmData,
  kellyData,
  physicsLoading,
  physicsError,
  physicsLastUpdated
} from './risk';
export type {
  PhysicsIsingData,
  PhysicsLyapunovData,
  PhysicsHMMData,
  PhysicsKellyData,
  PhysicsSensorData,
  PhysicsSensorState
} from './risk';

// Risk Compliance Store (Story 4-6: Risk Canvas Compliance & Backtest Viewer)
export {
  complianceStore,
  complianceData,
  complianceLoading,
  complianceError,
  complianceLastUpdated,
  propFirmStore,
  propFirms,
  propFirmLoading,
  propFirmError,
  calendarGateStore,
  calendarGateData,
  calendarGateLoading,
  calendarGateError,
  calendarGateLastUpdated,
  backtestStore,
  backtestList,
  selectedBacktest,
  backtestLoading,
  backtestError
} from './risk';
export type {
  AccountTagCompliance,
  IslamicComplianceStatus,
  ComplianceData,
  PropFirm,
  CalendarEvent,
  CalendarBlackout,
  CalendarGateData,
  BacktestSummary,
  BacktestDetail
} from './risk';

// Canvas Store (Story 1-6-9: Canvas Routing Skeleton)
export { activeCanvasStore, CANVASES, CANVAS_SHORTCUTS } from './canvasStore';
export type { Canvas } from './canvasStore';

// Canvas Context Store (Story 5-3: Canvas Context System)
export { canvasContextStore } from './canvas';
export type { CanvasContext } from './canvas';

// Shared Assets Store (Story 6-7: Shared Assets Canvas)
export {
  sharedAssetsStore,
  currentAssets,
  assetsLoading,
  assetsError,
  selectedAsset,
  assetCounts,
  hasAssets,
  getAssetsByType,
  initSharedAssetsStore
} from './sharedAssets';
export type { AssetFilter, SharedAssetsState } from './sharedAssets';

// News Store (Story 6-6: Live News Feed Tile & News Canvas Integration)
export {
  newsStore,
  filteredNews,
  latestNews,
  highSeverityNews,
  newsLoading,
  newsError,
  currentNewsFilter,
  initNewsStore,
  cleanupNewsStore,
  getExposureCount
} from './news';
export type { NewsSeverity, NewsActionType, NewsFilter, NewsState, NewsAlertMessage } from './news';

// FlowForge Store (Story 11.5: FlowForge Canvas — Prefect Kanban & Node Graph)
export {
  flowForgeStore,
  workflows,
  workflowsByState,
  flowforgeSelectedWorkflow,
  selectedWorkflowForNodeGraph,
  showNodeGraph,
  flowforgeLoading,
  flowforgeError,
  KANBAN_COLUMNS
} from './flowforge';
export type { PrefectTask, PrefectWorkflow, WorkflowState, WorkflowsByState } from './flowforge';

// Theme Store (Story 11.7: Theme Presets & Wallpaper System)
export {
  theme,
  wallpaper,
  scanlines,
  THEME_PRESETS,
  getPreset,
  type ThemePreset,
  type ThemeConfig
} from './theme';

// SVSS Store (Story: SVSS REST API - VWAP, RVOL, Volume Profile, MFI)
export {
  svssStore,
  svssVWAP,
  svssRVOL,
  svssProfile,
  svssMFI,
  svssSymbols,
  svssSelectedSymbol,
  svssLoading,
  svssError,
  svssLastUpdated,
  selectedSymbolVWAP,
  selectedSymbolRVOL,
  getMFIZoneColor,
  getRVOLColor,
  getRVOLInterpretation
} from './svssStore';
export type {
  VWAPData,
  RVOLData,
  VolumeProfileData,
  MFIData,
  SVSSSummary,
  SVSSState
} from './svssStore';

// Inter-Session Cooldown Store (Story 16.3: Inter-Session Cooldown Panel)
export {
  cooldownStatus,
  cooldownStateEvent,
  cooldownLoading,
  cooldownError,
  isCooldownActive,
  cooldownCountdown,
  sessionTransition,
  cooldownProgressPercent,
  cooldownStepName,
  areActionsBlocked,
  isIntelligenceWindowActive,
  cooldownStatusColor,
  cooldownCurrentStep,
  COOLDOWN_STEP_INFO,
  fetchCooldownStatus,
  fetchCooldownStateEvent,
  startCooldownPolling,
  stopCooldownPolling
} from './cooldownStore';
export type {
  CooldownStatus,
  CooldownStateEvent,
  CooldownPhase
} from './cooldownStore';

// Correlation Sensor Store (Story 4-7: Correlation Sensor Dashboard)
export {
  correlationSensorStore,
  correlationData,
  correlationLoading,
  correlationError,
  correlationLastUpdated,
  correlationRegime,
  eigenvalueStatus,
  eigenvalueColor,
  regimeColor
} from './correlationSensor';
export type {
  CorrelationSensorData,
  CorrelationMatrixData,
  CorrelationSensorState
} from './correlationSensor';

// Session Kelly Store (Story 4.10: Session-Scoped Kelly Modifiers)
export { sessionKellyStore } from './sessionKellyStore';
export type {
  SessionKellyData,
  KellyCurrentState,
  KellyHistoryEntry,
  KellyHistoryState,
  KellyStatus
} from './sessionKellyStore';

// Bot Lifecycle Store (P2: Bot Lifecycle Stage Report Viewer)
export {
  lifecycleStore,
  selectedBotLifecycle,
  selectedStageReport,
  stageProgress,
  isInRecovery,
  failedQuestions,
  passedQuestions,
  lifecycleLoading,
  lifecycleError,
  lifecycleStats,
  lifecycleBotList,
  lifecycleTotalBots
} from './lifecycleStore';
export type {
  LifecycleStage,
  DeclineRecoveryStatus,
  QAAnswer,
  StageMetrics,
  StageReport,
  BotLifecycle,
  LifecycleStats,
  BotLifecycleSummary,
  LifecycleListResponse
} from './lifecycleStore';

// WFA Store (Walk-Forward Analysis Calibrator)
export { wfaStore } from './wfaStore';
export type {
  WfaCalibration,
  WfaHistoryEntry,
  WfaHistory
} from './wfaStore';

// Economic Calendar Store (P3: Economic Calendar Full View)
export {
  economicCalendarStore,
  economicCalendarData,
  economicCalendarEvents,
  economicCalendarBlackouts,
  economicCalendarLoading,
  economicCalendarError,
  economicCalendarLastUpdated,
  getImpactColor,
  getCurrencyBadge,
  getTimeUntilEvent,
  isBlackoutActive,
  getNextEvent
} from './economicCalendarStore';
export type {
  EconomicEvent,
  BlackoutWindow,
  EconomicCalendarData,
  EconomicCalendarState
} from './economicCalendarStore';

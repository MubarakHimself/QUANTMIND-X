/**
 * QuantMind IDE API Service
 * Centralized API calls for all UI components
 */

import { API_CONFIG } from './config/api';

const API_BASE = `${API_CONFIG.API_URL}/api`;

/**
 * Generic fetch wrapper with error handling and cookie-based auth.
 * Includes credentials: 'include' to send httpOnly session cookies.
 * Exported for use in components that cannot import from domain-specific API modules.
 */
export async function apiFetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE}${endpoint}`, {
        ...options,
        credentials: 'include',  // Required for httpOnly cookie-based auth
        headers: {
            'Content-Type': 'application/json',
            ...options?.headers
        }
    });

    if (!response.ok) {
        let errorMessage = `API Error: ${response.status} ${response.statusText}`;
        try {
            const errorData = await response.json();
            errorMessage = errorData.detail || errorData.message || errorMessage;
        } catch {
            // If JSON parsing fails, use the default error message
        }
        throw new Error(errorMessage);
    }

    return response.json();
}

// =============================================================================
// Strategy Endpoints
// =============================================================================

export interface StrategyFolder {
    id: string;
    name: string;
    status: 'pending' | 'processing' | 'ready' | 'primal' | 'quarantined';
    created_at: string;
    has_video_ingest: boolean;
    has_trd: boolean;
    has_ea: boolean;
    has_backtest: boolean;
}

export interface StrategyDetail {
    id: string;
    name: string;
    status: string;
    created_at: string;
    video_ingest?: { files: string[]; metadata?: any };
    trd?: { files: string[] };
    ea?: { files: string[] };
    backtests: Array<{ name: string; path: string; mode: string }>;
}

// =============================================================================
// Article Endpoints
// =============================================================================

export interface Article {
    id: string;
    name: string;
    path: string;
    size_bytes: number;
    created_at: string;
    modified_at: string;
}

export interface ArticleResponse {
    articles: Article[];
}

export interface ArticleContentResponse {
    content: string;
}

export interface ArticlePreviewResponse {
    preview: string;
}

// =============================================================================
// Articles API Functions
// =============================================================================

export async function getArticles(): Promise<ArticleResponse> {
    return apiFetch<ArticleResponse>('/articles');
}

export async function getArticleContent(articleId: string): Promise<ArticleContentResponse> {
    return apiFetch<ArticleContentResponse>(`/articles/${articleId}`);
}

export async function getArticlePreview(articleId: string): Promise<ArticlePreviewResponse> {
    return apiFetch<ArticlePreviewResponse>(`/articles/${articleId}/preview`);
}

export async function getStrategies(): Promise<StrategyFolder[]> {
    return apiFetch<StrategyFolder[]>('/strategies');
}

export async function getStrategy(id: string): Promise<StrategyDetail> {
    return apiFetch<StrategyDetail>(`/strategies/${id}`);
}

export async function createStrategy(name: string): Promise<{ id: string; name: string }> {
    return apiFetch('/strategies', {
        method: 'POST',
        body: JSON.stringify({ name })
    });
}

// =============================================================================
// Shared Assets Endpoints
// =============================================================================

export interface SharedAsset {
    id: string;
    name: string;
    type: string;
    path: string;
    description?: string;
    used_in: string[];
}

export async function getAssets(category?: string): Promise<SharedAsset[]> {
    const params = category ? `?category=${category}` : '';
    return apiFetch<SharedAsset[]>(`/assets${params}`);
}

export async function getAssetContent(assetId: string): Promise<{ content: string }> {
    return apiFetch(`/assets/${assetId}/content`);
}

// =============================================================================
// Knowledge Hub Endpoints
// =============================================================================

export interface KnowledgeItem {
    id: string;
    name: string;
    category: string;
    path: string;
    size_bytes: number;
    indexed: boolean;
}

export async function getKnowledge(category?: string): Promise<KnowledgeItem[]> {
    const params = category ? `?category=${category}` : '';
    return apiFetch<KnowledgeItem[]>(`/knowledge${params}`);
}

export async function getKnowledgeContent(itemId: string): Promise<{ content: string }> {
    return apiFetch(`/knowledge/${itemId}/content`);
}

// =============================================================================
// VideoIngest Processing Endpoints
// =============================================================================

export interface VideoIngestJob {
    job_id: string;
    status: string;
    strategy_folder: string;
    progress?: number;
}

export async function processVideoIngest(url: string, strategyName: string): Promise<VideoIngestJob> {
    return apiFetch('/video-ingest/process', {
        method: 'POST',
        body: JSON.stringify({ url, strategy_name: strategyName })
    });
}

export async function getVideoIngestJobStatus(jobId: string): Promise<VideoIngestJob> {
    return apiFetch(`/video-ingest/jobs/${jobId}`);
}

// =============================================================================
// Live Trading Endpoints
// =============================================================================

export interface Bot {
    id: string;
    name: string;
    state: string;
    tags: string[];
    symbol: string;
    account_id: string;
    pnl?: number;
}

export interface SystemStatus {
    connected: boolean;
    regime: string;
    kelly: number;
    active_bots: number;
    pnl_today: number;
}

export async function getBots(): Promise<Bot[]> {
    return apiFetch<Bot[]>('/trading/bots');
}

export async function controlBot(botId: string, action: 'pause' | 'resume' | 'quarantine' | 'kill'): Promise<any> {
    return apiFetch('/trading/bots/control', {
        method: 'POST',
        body: JSON.stringify({ bot_id: botId, action })
    });
}

export async function getSystemStatus(): Promise<SystemStatus> {
    return apiFetch<SystemStatus>('/trading/status');
}

export async function triggerKillSwitch(): Promise<{ success: boolean; message: string }> {
    return apiFetch('/trading/kill', { method: 'POST' });
}

// =============================================================================
// Fee Monitoring Endpoints
// =============================================================================

export interface FeeBreakdownItem {
    bot_id: string;
    trades: number;
    fees_paid: number;
    fee_pct: number;
}

export interface FeeMonitorData {
    daily_fees: number;
    daily_fee_burn_pct: number;
    kill_switch_active: boolean;
    fee_breakdown: FeeBreakdownItem[];
}

export async function getFeeMonitorData(): Promise<FeeMonitorData> {
    return apiFetch<FeeMonitorData>('/router/fee-monitor');
}

// =============================================================================
// Agent Chat Endpoint
// =============================================================================

export interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
    agent?: string;
}

export async function sendChatMessage(
    message: string,
    agent: string,
    model: string,
    context?: string[]
): Promise<{ response: string }> {
    return apiFetch('/chat', {
        method: 'POST',
        body: JSON.stringify({ message, agent, model, context })
    });
}

// =============================================================================
// File Operations
// =============================================================================

export async function getFileContent(path: string): Promise<{ content: string }> {
    return apiFetch(`/files/content?path=${encodeURIComponent(path)}`);
}

// =============================================================================
// Backtest Endpoints
// =============================================================================

export interface BacktestRequest {
    symbol: string;
    timeframe: string;
    variant: 'vanilla' | 'spiced' | 'vanilla_full' | 'spiced_full';
    start_date: string;
    end_date: string;
    strategy_code?: string;
    strategy_name?: string;
}

export interface BacktestResponse {
    backtest_id: string;
    status: string;
    message?: string;
}

export interface BacktestResult {
    backtest_id: string;
    final_balance: number;
    total_trades: number;
    win_rate?: number;
    sharpe_ratio?: number;
    drawdown?: number;
    return_pct?: number;
    duration_seconds?: number;
    results?: Record<string, unknown>;
}

export async function runBacktest(request: BacktestRequest): Promise<BacktestResponse> {
    return apiFetch<BacktestResponse>('/v1/backtest/run', {
        method: 'POST',
        body: JSON.stringify(request)
    });
}

export async function getBacktestResults(backtestId: string): Promise<BacktestResult> {
    return apiFetch<BacktestResult>(`/v1/backtest/results/${backtestId}`);
}

export async function getBacktestStatus(backtestId: string): Promise<{ status: string; progress: number }> {
    return apiFetch(`/v1/backtest/status/${backtestId}`);
}

// =============================================================================
// Status Band Endpoints
// =============================================================================

export interface SessionStatus {
  active: boolean;
  name: string;
}

export interface SessionsResponse {
  [key: string]: SessionStatus;
}

export interface MarketRegime {
  quality: number;
  trend: string;
  chaos: number;
  volatility: string;
}

export interface MarketResponse {
  regime: MarketRegime;
  symbols?: Array<{
    symbol: string;
    price: number;
    change: number;
    spread: number;
  }>;
}

export interface TradingMetrics {
  tick_latency_ms: number;
  active_bots: number;
  active_positions: number;
  daily_pnl: number;
  total_trades: number;
  win_rate: number;
}

export async function getAllSessions(): Promise<SessionsResponse> {
  return apiFetch<SessionsResponse>('/sessions/all');
}

export interface CurrentSessionInfo {
  session: string;
  utc_time: string;
  next_session: string | null;
  is_active: boolean;
  time_until_open: number | null;
  time_until_close: number | null;
  time_until_close_str: string | null;
}

export async function getCurrentSessionInfo(): Promise<CurrentSessionInfo> {
  return apiFetch<CurrentSessionInfo>('/sessions/current');
}

export async function getMarketState(): Promise<MarketResponse> {
  return apiFetch<MarketResponse>('/router/market');
}

export async function getTradingMetrics(): Promise<TradingMetrics> {
  return apiFetch<TradingMetrics>('/metrics/trading');
}

// =============================================================================
// Risk Settings Endpoints
// =============================================================================

export interface RiskSettings {
  houseMoneyEnabled: boolean;
  houseMoneyThreshold: number;
  dailyLossLimit: number;
  maxDrawdown: number;
  riskMode: 'fixed' | 'dynamic' | 'conservative';
  propFirmPreset: 'ftmo' | 'the5ers' | 'fundingpips' | 'custom';
  balanceZones: {
    danger: number;
    growth: number;
    scaling: number;
    guardian: number | typeof Infinity;
  };
  maxRiskPerTrade: number;
}

export async function getRiskSettings(): Promise<RiskSettings> {
  return apiFetch<RiskSettings>('/settings/risk');
}

export async function saveRiskSettings(settings: RiskSettings): Promise<void> {
  return apiFetch('/settings/risk', {
    method: 'POST',
    body: JSON.stringify(settings)
  });
}

// =============================================================================
// Router Settings Endpoints
// =============================================================================

export interface RouterSettings {
  active: boolean;
  mode: 'auction' | 'priority' | 'round-robin';
  auctionInterval: number;
}

export async function getRouterSettings(): Promise<RouterSettings> {
  return apiFetch<RouterSettings>('/router/state');
}

export async function saveRouterSettings(settings: RouterSettings): Promise<void> {
  return apiFetch('/router/settings', {
    method: 'POST',
    body: JSON.stringify(settings)
  });
}

// Export WebSocket client creation functions
export {
  createBacktestClient,
  createTradingClient,
  createWebSocketClient
} from './ws-client';

// =============================================================================
// Memory API Endpoints
// =============================================================================

// Re-export all memory-related functions
export * from './api/memory';

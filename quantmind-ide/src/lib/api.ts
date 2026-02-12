/**
 * QuantMind IDE API Service
 * Centralized API calls for all UI components
 */

const API_BASE = 'http://localhost:8000/api';

/**
 * Generic fetch wrapper with error handling
 */
async function apiFetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE}${endpoint}`, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...options?.headers
        }
    });

    if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
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
    has_nprd: boolean;
    has_trd: boolean;
    has_ea: boolean;
    has_backtest: boolean;
}

export interface StrategyDetail {
    id: string;
    name: string;
    status: string;
    created_at: string;
    nprd?: { files: string[]; metadata?: any };
    trd?: { files: string[] };
    ea?: { files: string[] };
    backtests: Array<{ name: string; path: string; mode: string }>;
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
// NPRD Processing Endpoints
// =============================================================================

export interface NPRDJob {
    job_id: string;
    status: string;
    strategy_folder: string;
    progress?: number;
}

export async function processNPRD(url: string, strategyName: string): Promise<NPRDJob> {
    return apiFetch('/nprd/process', {
        method: 'POST',
        body: JSON.stringify({ url, strategy_name: strategyName })
    });
}

export async function getNPRDJobStatus(jobId: string): Promise<NPRDJob> {
    return apiFetch(`/nprd/jobs/${jobId}`);
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

// Export WebSocket client creation functions
export {
    createBacktestClient,
    createTradingClient,
    createWebSocketClient
} from './ws-client';

import { API_CONFIG } from "$lib/config/api";

const API_BASE = API_CONFIG.API_BASE;

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export function fetchBacktestHistory(limit = 20) {
  return fetchJson<any[]>(`/analytics/backtests?limit=${limit}`);
}

export function fetchBacktestTrades(runId: string) {
  return fetchJson<any[]>(`/analytics/trades?run_id=${encodeURIComponent(runId)}`);
}

export function uploadKnowledgeFile(formData: FormData) {
  return fetch(`${API_BASE}/ide/knowledge/upload`, {
    method: "POST",
    body: formData,
  });
}

export function uploadKnowledgeNote(formData: FormData) {
  return fetch(`${API_BASE}/ide/knowledge/upload/note`, {
    method: "POST",
    body: formData,
  });
}

export function fetchKnowledgeContent(articleIdOrPath: string) {
  return fetchJson<{ content: string }>(`/knowledge/${encodeURIComponent(articleIdOrPath)}/content`);
}

export function fetchAssetContent(assetId: string) {
  return fetchJson<{ content: string }>(`/assets/${encodeURIComponent(assetId)}/content`);
}

export function fetchRelatedKnowledge(articleId: string, limit = 5) {
  return fetchJson<any[]>(`/knowledge/related?id=${encodeURIComponent(articleId)}&limit=${limit}`);
}

export function fetchKnowledgeIndex() {
  return fetchJson<any[]>(`/knowledge`);
}

export function fetchStrategiesIndex() {
  return fetchJson<any[]>(`/strategies`);
}

export function fetchTradingStatus() {
  return fetchJson<any>(`/trading/status`);
}

export function fetchTradingBots() {
  return fetchJson<any[]>(`/trading/bots`);
}

export function processVideoIngest(payload: { url: string; strategy_name: string }) {
  return fetchJson<{ job_id: string }>(`/videoIngest/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function controlTradingBot(payload: { bot_id: string; action: string }) {
  return fetchJson<any>(`/trading/bots/control`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function triggerTradingKillSwitch() {
  return fetchJson<any>(`/trading/kill`, {
    method: "POST",
  });
}

export function updateKnowledgeArticle(articleId: string, content: string) {
  return fetchJson<any>(`/knowledge/${encodeURIComponent(articleId)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  });
}

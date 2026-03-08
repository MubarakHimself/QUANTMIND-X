import { getJson, postJson, putJson } from '$lib/services/componentApi';

export interface HmmTrainRequest {
  model_type: string;
  symbol: string | null;
  timeframe: string | null;
  n_states: number;
  force_retrain: boolean;
}

export interface Mt5ConfigPayload {
  server: string;
  port: number;
  login: string;
  password: string;
  symbolMapping: string;
}

export interface BrokerCredentialsPayload {
  broker: string;
  credentials: Record<string, unknown>;
}

export interface VideoIngestStartPayload {
  url: string;
  strategy_name: string;
}

export function startHmmTraining<T>(payload: HmmTrainRequest): Promise<T> {
  return postJson<T>('/hmm/train', payload);
}

export function getHmmTrainingStatus<T>(jobId: string): Promise<T> {
  return getJson<T>(`/hmm/train/${jobId}/status`);
}

export function testMt5Connection<T>(payload: Mt5ConfigPayload): Promise<T> {
  return postJson<T>('/mt5/test', payload);
}

export function saveMt5Config<T>(payload: Mt5ConfigPayload): Promise<T> {
  return postJson<T>('/mt5/config', payload);
}

export function getRouterState<T>(): Promise<T> {
  return getJson<T>('/router/state');
}

export function toggleRouterState<T>(active: boolean): Promise<T> {
  return postJson<T>('/router/toggle', { active });
}

export function runRouterAuction<T>(): Promise<T> {
  return postJson<T>('/router/auction', {});
}

export function listEaTags<T>(): Promise<T> {
  return getJson<T>('/ea/tags');
}

export function addEaTag<T>(botName: string, tagId: string): Promise<T> {
  return postJson<T>(`/ea/tags/${encodeURIComponent(botName)}/add?tag=${encodeURIComponent(tagId)}`, {});
}

export function removeEaTag<T>(botName: string, tagId: string): Promise<T> {
  return postJson<T>(`/ea/tags/${encodeURIComponent(botName)}/remove?tag=${encodeURIComponent(tagId)}`, {});
}

export function listEaBots<T>(): Promise<T> {
  return getJson<T>('/ea/bots');
}

export function listEaReviews<T>(): Promise<T> {
  return getJson<T>('/ea/reviews');
}

export function startVideoIngest<T>(payload: VideoIngestStartPayload): Promise<T> {
  return postJson<T>('/video-ingest/start', payload);
}

export function getTrd<T>(id: string): Promise<T> {
  return getJson<T>(`/trd/${id}`);
}

export function createTrd<T>(payload: unknown): Promise<T> {
  return postJson<T>('/trd', payload);
}

export function updateTrd<T>(id: string, payload: unknown): Promise<T> {
  return putJson<T>(`/trd/${id}`, payload);
}

export function testTradingBroker<T>(payload: BrokerCredentialsPayload): Promise<T> {
  return postJson<T>('/trading/broker/test', payload);
}

export function connectTradingBroker<T>(payload: BrokerCredentialsPayload): Promise<T> {
  return postJson<T>('/trading/broker/connect', payload);
}

export function listKillSwitchBots<T>(): Promise<T> {
  return getJson<T>('/kill-switch/bots');
}

export function listKillSwitchHistory<T>(): Promise<T> {
  return getJson<T>('/kill-switch/history');
}

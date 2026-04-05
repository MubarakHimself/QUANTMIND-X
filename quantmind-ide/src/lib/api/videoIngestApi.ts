/**
 * Video ingest API client.
 */

import { apiFetch } from '$lib/api';

export interface VideoIngestJobResponse {
  job_id: string;
  job_ids?: string[];
  status: string;
  strategy_folder: string;
  message?: string;
}

export interface VideoIngestJobStatus {
  job_id: string;
  status: 'PENDING' | 'DOWNLOADING' | 'PROCESSING' | 'ANALYZING' | 'COMPLETED' | 'FAILED' | string;
  current_stage?: string;
  progress?: number;
  video_url?: string;
  created_at?: string;
  updated_at?: string;
  result_path?: string | null;
  error?: string;
  workflow_id?: string;
  workflow_status?: string;
  waiting_reason?: string | null;
  blocking_error?: string | null;
  blocking_error_detail?: string | null;
  strategy_id?: string;
  strategy_folder?: string;
  strategy_asset_id?: string;
  strategy_status?: string;
  strategy_family?: string;
  source_bucket?: string;
  has_video_ingest?: boolean;
}

export interface VideoIngestAuthStatus {
  openrouter?: boolean;
  gemini: boolean;
  qwen: boolean;
}

export interface VideoIngestProviderModel {
  id: string;
  name?: string;
  pricing?: string;
  capabilities?: string;
}

export interface VideoIngestProviderSummary {
  provider_type: string;
  display_name: string;
  is_active: boolean;
  primary_model?: string;
  selected_model?: VideoIngestProviderModel | null;
}

export async function submitVideoJob(
  url: string,
  strategyName?: string,
  isPlaylist?: boolean
): Promise<VideoIngestJobResponse> {
  return apiFetch<VideoIngestJobResponse>('/api/video-ingest/process', {
    method: 'POST',
    body: JSON.stringify({
      url,
      strategy_name: strategyName || 'video_ingest',
      is_playlist: isPlaylist || false
    })
  });
}

export async function getJobStatus(jobId: string): Promise<VideoIngestJobStatus> {
  return apiFetch<VideoIngestJobStatus>(`/api/video-ingest/jobs/${encodeURIComponent(jobId)}`);
}

export async function listVideoJobs(limit = 25, status?: string): Promise<VideoIngestJobStatus[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (status) params.set('status', status);
  return apiFetch<VideoIngestJobStatus[]>(`/api/video-ingest/jobs?${params.toString()}`);
}

export async function getAuthStatus(): Promise<VideoIngestAuthStatus> {
  return apiFetch<VideoIngestAuthStatus>('/api/video-ingest/auth-status');
}

export async function getPreferredVideoProvider(): Promise<VideoIngestProviderSummary | null> {
  const response = await apiFetch<{ providers: Array<Record<string, any>> }>('/api/providers');
  const configured = response.providers || [];
  const provider = configured.find((entry) => entry.provider_type === 'openrouter' && entry.is_active);
  if (!provider) return null;

  const primaryModel = provider.primary_model || provider.model_list?.[0]?.id || provider.model_list?.[0]?.model_id;
  const selectedModel = (provider.model_list || []).find((entry: Record<string, any>) =>
    (entry.id || entry.model_id || entry.name) === primaryModel
  ) || null;

  return {
    provider_type: provider.provider_type,
    display_name: provider.display_name || 'OpenRouter',
    is_active: Boolean(provider.is_active),
    primary_model: primaryModel,
    selected_model: selectedModel
      ? {
          id: selectedModel.id || selectedModel.model_id || selectedModel.name,
          name: selectedModel.name,
          pricing: selectedModel.pricing,
          capabilities: selectedModel.capabilities,
        }
      : null,
  };
}

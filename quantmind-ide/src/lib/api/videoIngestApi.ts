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
  error?: string;
}

export interface VideoIngestAuthStatus {
  openrouter?: boolean;
  gemini: boolean;
  qwen: boolean;
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

export async function getAuthStatus(): Promise<VideoIngestAuthStatus> {
  return apiFetch<VideoIngestAuthStatus>('/api/video-ingest/auth-status');
}

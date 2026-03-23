/**
 * Video Ingest API Client
 * Provides frontend API functions for YouTube video processing
 *
 * Connects to the backend endpoints defined in src/api/ide_video_ingest.py
 */

import { API_CONFIG } from '$lib/config/api';

// =============================================================================
// Types
// =============================================================================

/**
 * Response from submitting a video processing job
 */
export interface VideoIngestJobResponse {
  job_id: string;
  status: string;
  strategy_folder: string;
  message?: string;
}

/**
 * Job status information
 */
export interface VideoIngestJobStatus {
  job_id: string;
  status: 'processing' | 'completed' | 'failed' | string;
  current_stage?: string;
  progress?: number;
  error?: string;
}

/**
 * Authentication status for video ingest providers
 */
export interface VideoIngestAuthStatus {
  gemini: boolean;
  qwen: boolean;
}

// =============================================================================
// API Functions
// =============================================================================

/**
 * Generic fetch wrapper with error handling
 */
async function apiFetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_CONFIG.API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers
    }
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API Error: ${response.status} ${response.statusText} - ${errorText}`);
  }

  return response.json();
}

/**
 * Submit a video for processing
 *
 * @param url - YouTube URL to process
 * @param strategyName - Name for the strategy folder (optional)
 * @param isPlaylist - Whether URL is a playlist (optional)
 */
export async function submitVideoJob(
  url: string,
  strategyName?: string,
  isPlaylist?: boolean
): Promise<VideoIngestJobResponse> {
  return apiFetch<VideoIngestJobResponse>('/video-ingest/process', {
    method: 'POST',
    body: JSON.stringify({
      url,
      strategy_name: strategyName || 'video_ingest',
      is_playlist: isPlaylist || false
    })
  });
}

/**
 * Get the status of a video processing job
 *
 * @param jobId - The job ID to check
 */
export async function getJobStatus(jobId: string): Promise<VideoIngestJobStatus> {
  return apiFetch<VideoIngestJobStatus>(`/video-ingest/jobs/${encodeURIComponent(jobId)}`);
}

/**
 * Get authentication status for video ingest providers
 */
export async function getAuthStatus(): Promise<VideoIngestAuthStatus> {
  return apiFetch<VideoIngestAuthStatus>('/video-ingest/auth-status');
}
/**
 * Video Ingest State Store
 *
 * Manages state for YouTube video processing:
 * - Active jobs and their status
 * - Pipeline stage tracking
 * - Job submission and polling
 */

import { writable, derived, get } from 'svelte/store';
import {
  submitVideoJob,
  getJobStatus,
  getAuthStatus,
  type VideoIngestJobResponse,
  type VideoIngestJobStatus,
  type VideoIngestAuthStatus
} from '$lib/api/videoIngestApi';

// =============================================================================
// Types
// =============================================================================

/**
 * Pipeline stages for video processing
 */
export type PipelineStage =
  | 'PENDING'
  | 'DOWNLOADING'
  | 'TRANSCRIBING'
  | 'CHUNKING'
  | 'EMBEDDING'
  | 'INDEXING'
  | 'COMPLETED'
  | 'FAILED';

/**
 * Video job status enum
 */
export type VideoJobStatus = 'pending' | 'processing' | 'completed' | 'failed';

/**
 * Video job information
 */
export interface VideoJob {
  jobId: string;
  url: string;
  status: VideoJobStatus;
  currentStage: PipelineStage;
  progress: number;
  strategyFolder?: string;
  error?: string;
  submittedAt: Date;
  completedAt?: Date;
}

/**
 * Auth status for video ingest providers
 */
export interface VideoIngestAuth {
  openrouter: boolean;
  gemini: boolean;
  qwen: boolean;
  checked: boolean;
}

// =============================================================================
// Pipeline Stage Order
// =============================================================================

/**
 * Ordered list of pipeline stages for progress tracking
 */
export const PIPELINE_STAGES: PipelineStage[] = [
  'DOWNLOADING',
  'TRANSCRIBING',
  'CHUNKING',
  'EMBEDDING',
  'INDEXING',
  'COMPLETED'
];

/**
 * Map backend status to pipeline stage
 */
export function getPipelineStage(backendStatus: string): PipelineStage {
  const statusMap: Record<string, PipelineStage> = {
    pending: 'PENDING',
    processing: 'DOWNLOADING',
    completed: 'COMPLETED',
    failed: 'FAILED'
  };
  return statusMap[backendStatus.toLowerCase()] || 'PENDING';
}

/**
 * Calculate progress percentage based on current stage
 */
export function calculateProgress(stage: PipelineStage): number {
  const stageIndex = PIPELINE_STAGES.indexOf(stage);
  if (stageIndex === -1) return 0;
  return Math.round((stageIndex / (PIPELINE_STAGES.length - 1)) * 100);
}

// =============================================================================
// Stores
// =============================================================================

/**
 * Map of all video jobs by job ID
 */
export const videoJobs = writable<Map<string, VideoJob>>(new Map());

/**
 * Currently active job
 */
export const currentJob = writable<VideoJob | null>(null);

/**
 * Whether a job is currently being processed
 */
export const isProcessing = writable(false);

/**
 * Authentication status for video ingest providers
 */
export const authStatus = writable<VideoIngestAuth>({
  openrouter: false,
  gemini: false,
  qwen: false,
  checked: false
});

/**
 * Error message for the current operation
 */
export const errorMessage = writable<string | null>(null);

/**
 * Fires when a video ingest job completes successfully.
 * ResearchCanvas subscribes to this to trigger knowledge search refresh.
 */
export const completedJobEvent = writable<VideoJob | null>(null);

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Check if any job is currently processing
 */
export function hasProcessingJob(): boolean {
  const jobs = get(videoJobs);
  return Array.from(jobs.values()).some(job => job.status === 'processing');
}

/**
 * Submit a new video processing job
 */
export async function submitJob(
  url: string,
  strategyName?: string,
  isPlaylist?: boolean
): Promise<VideoJob | null> {
  isProcessing.set(true);
  errorMessage.set(null);

  try {
    const response: VideoIngestJobResponse = await submitVideoJob(url, strategyName, isPlaylist);

    const newJob: VideoJob = {
      jobId: response.job_id,
      url,
      status: 'processing',
      currentStage: getPipelineStage(response.status),
      progress: 0,
      strategyFolder: response.strategy_folder,
      submittedAt: new Date()
    };

    videoJobs.update(jobs => {
      jobs.set(newJob.jobId, newJob);
      return jobs;
    });

    currentJob.set(newJob);
    isProcessing.set(true);

    // Start polling for status
    pollJobStatus(newJob.jobId);

    return newJob;
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Failed to submit job';
    errorMessage.set(message);
    isProcessing.set(false);
    return null;
  }
}

/**
 * Poll job status at regular intervals
 */
let pollingIntervals: Map<string, number> = new Map();

export async function pollJobStatus(jobId: string): Promise<void> {
  // Clear any existing polling for this job
  stopPolling(jobId);

  const poll = async () => {
    try {
      const status: VideoIngestJobStatus = await getJobStatus(jobId);

      videoJobs.update(jobs => {
        const job = jobs.get(jobId);
        if (!job) return jobs;

        // Use current_stage from backend when available for accurate per-step progress
        const stage: PipelineStage = status.current_stage
          ? (status.current_stage.toUpperCase() as PipelineStage)
          : getPipelineStage(status.status);
        const updatedJob: VideoJob = {
          ...job,
          status: status.status as VideoJobStatus,
          currentStage: stage,
          progress: calculateProgress(stage),
          error: status.error,
          completedAt: status.status === 'completed' || status.status === 'failed' ? new Date() : undefined
        };

        jobs.set(jobId, updatedJob);

        // Update current job if it's the one being polled
        const current = get(currentJob);
        if (current?.jobId === jobId) {
          currentJob.set(updatedJob);
        }

        // Stop polling if job is complete or failed
        if (status.status === 'completed' || status.status === 'failed') {
          isProcessing.set(false);
          stopPolling(jobId);
          if (status.status === 'completed') {
            completedJobEvent.set(updatedJob);
          }
        }

        return jobs;
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to get job status';
      errorMessage.set(message);
      stopPolling(jobId);
    }
  };

  // Poll immediately
  await poll();

  // Then poll every 3 seconds
  const intervalId = window.setInterval(poll, 3000);
  pollingIntervals.set(jobId, intervalId);
}

/**
 * Stop polling for a specific job
 */
export function stopPolling(jobId: string): void {
  const intervalId = pollingIntervals.get(jobId);
  if (intervalId) {
    window.clearInterval(intervalId);
    pollingIntervals.delete(jobId);
  }
}

/**
 * Clear completed jobs from the store
 */
export function clearCompletedJobs(): void {
  videoJobs.update(jobs => {
    const updated = new Map<string, VideoJob>();
    jobs.forEach((job, id) => {
      if (job.status === 'processing' || job.status === 'pending') {
        updated.set(id, job);
      }
    });
    return updated;
  });

  const current = get(currentJob);
  if (current && (current.status === 'completed' || current.status === 'failed')) {
    currentJob.set(null);
  }
}

/**
 * Load authentication status
 */
export async function loadAuthStatus(): Promise<void> {
  try {
    const status: VideoIngestAuthStatus = await getAuthStatus();
    authStatus.set({
      openrouter: status.openrouter ?? false,
      gemini: status.gemini,
      qwen: status.qwen,
      checked: true
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Failed to get auth status';
    errorMessage.set(message);
    authStatus.set({
      openrouter: false,
      gemini: false,
      qwen: false,
      checked: true
    });
  }
}

/**
 * Validate YouTube URL
 */
export function isValidYouTubeUrl(url: string): boolean {
  const patterns = [
    /^https?:\/\/(www\.)?youtube\.com\/watch\?v=[\w-]+$/,
    /^https?:\/\/(www\.)?youtube\.com\/playlist\?list=[\w-]+$/,
    /^https?:\/\/youtu\.be\/[\w-]+$/
  ];
  return patterns.some(pattern => pattern.test(url));
}

/**
 * Clear error message
 */
export function clearError(): void {
  errorMessage.set(null);
}

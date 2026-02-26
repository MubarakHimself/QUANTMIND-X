import { writable, derived } from 'svelte/store';

export interface CronJob {
  id: string;
  name: string;
  enabled: boolean;
  schedule: string; // cron expression
  command: string;
  lastRun?: string;
  nextRun?: string;
  status: 'idle' | 'running' | 'success' | 'failed';
  lastStatus?: 'success' | 'failed';
  executionTime?: number; // ms
  description?: string;
}

function createCronStore() {
  const initialState = {
    jobs: [] as CronJob[],
    loading: false,
    error: null as string | null,
    selectedJob: null as CronJob | null
  };

  const { subscribe, update, set } = writable(initialState);

  return {
    subscribe,

    setJobs: (jobs: CronJob[]) => update(state => ({ ...state, jobs })),

    updateJob: (id: string, updates: Partial<CronJob>) => update(state => ({
      ...state,
      jobs: state.jobs.map(job =>
        job.id === id ? { ...job, ...updates } : job
      )
    })),

    toggleJob: (id: string) => update(state => ({
      ...state,
      jobs: state.jobs.map(job =>
        job.id === id ? { ...job, enabled: !job.enabled } : job
      )
    })),

    addJob: (job: CronJob) => update(state => ({
      ...state,
      jobs: [...state.jobs, job]
    })),

    removeJob: (id: string) => update(state => ({
      ...state,
      jobs: state.jobs.filter(job => job.id !== id)
    })),

    setSelectedJob: (job: CronJob | null) => update(state => ({
      ...state,
      selectedJob: job
    })),

    setLoading: (loading: boolean) => update(state => ({ ...state, loading })),

    setError: (error: string | null) => update(state => ({ ...state, error })),

    reset: () => set(initialState)
  };
}

export const cronStore = createCronStore();

// Derived stores
export const cronJobs = derived(cronStore, $store => $store.jobs);
export const enabledCronJobs = derived(cronStore, $store =>
  $store.jobs.filter(job => job.enabled)
);
export const cronLoading = derived(cronStore, $store => $store.loading);
export const cronError = derived(cronStore, $store => $store.error);

/**
 * Batch Processing Service
 *
 * Provides frontend interface to the batch processing API endpoints.
 * Allows submitting and monitoring batch operations.
 */

const API_BASE = '/api/batch';

// Types
export interface BatchSubmitRequest {
  payloads: any[];
  priority?: 'LOW' | 'NORMAL' | 'HIGH' | 'CRITICAL';
  metadata?: Record<string, any>;
  callback_type?: string;
}

export interface BatchSubmitResponse {
  batch_id: string;
  item_count: number;
  status: string;
  submitted_at: number;
}

export interface BatchStatusResponse {
  batch_id: string;
  status: string;
  total_items: number;
  completed_count: number;
  failed_count: number;
  queue_size: number;
  active_items: number;
}

export interface BatchItemStatusResponse {
  item_id: string;
  status: string;
  result?: any;
  error?: string;
  created_at?: number;
  started_at?: number;
  completed_at?: number;
}

export interface BatchResultResponse {
  batch_id: string;
  total: number;
  successful: number;
  failed: number;
  results: any[];
  errors: Array<{ id: string; error: string }>;
  duration: number;
  metadata: Record<string, any>;
}

export interface BatchStatsResponse {
  queue_size: number;
  active_items: number;
  total_batches: number;
  running: boolean;
  rate_limit_rps: number;
  burst_size: number;
  max_concurrent: number;
}

export interface BatchListItem {
  batch_id: string;
  status: string;
  total_items: number;
  completed_count: number;
  failed_count: number;
  created_at: number;
}

export const batchService = {
  /**
   * Start the batch processor
   */
  async start(): Promise<{ status: string; message: string }> {
    const response = await fetch(`${API_BASE}/start`, { method: 'POST' });
    return response.json();
  },

  /**
   * Stop the batch processor
   */
  async stop(): Promise<{ status: string; message: string }> {
    const response = await fetch(`${API_BASE}/stop`, { method: 'POST' });
    return response.json();
  },

  /**
   * Submit a batch of items for processing
   */
  async submitBatch(request: BatchSubmitRequest): Promise<BatchSubmitResponse> {
    const response = await fetch(`${API_BASE}/submit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to submit batch');
    }
    return response.json();
  },

  /**
   * Submit a single item for processing
   */
  async submitSingle(
    payload: any,
    priority: 'LOW' | 'NORMAL' | 'HIGH' | 'CRITICAL' = 'NORMAL',
    metadata?: Record<string, any>
  ): Promise<{ item_id: string; status: string; submitted_at: number }> {
    const response = await fetch(`${API_BASE}/submit-single?priority=${priority}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to submit item');
    }
    return response.json();
  },

  /**
   * Get status of a batch
   */
  async getBatchStatus(batchId: string): Promise<BatchStatusResponse> {
    const response = await fetch(`${API_BASE}/status/${batchId}`);
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Batch not found');
    }
    return response.json();
  },

  /**
   * Get status of a specific item
   */
  async getItemStatus(itemId: string): Promise<BatchItemStatusResponse> {
    const response = await fetch(`${API_BASE}/item/${itemId}`);
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Item not found');
    }
    return response.json();
  },

  /**
   * Get aggregated results for a batch
   */
  async getBatchResults(batchId: string): Promise<BatchResultResponse> {
    const response = await fetch(`${API_BASE}/results/${batchId}`);
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Batch not found');
    }
    return response.json();
  },

  /**
   * Get batch processor statistics
   */
  async getStats(): Promise<BatchStatsResponse> {
    const response = await fetch(`${API_BASE}/stats`);
    return response.json();
  },

  /**
   * Cancel a pending item
   */
  async cancelItem(itemId: string): Promise<{ status: string; item_id: string }> {
    const response = await fetch(`${API_BASE}/cancel`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ item_id: itemId }),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Item not found or already processed');
    }
    return response.json();
  },

  /**
   * Register a callback type for custom processing
   */
  async registerCallback(
    callbackType: string,
    callbackCode: string
  ): Promise<{ status: string; callback_type: string }> {
    const response = await fetch(`${API_BASE}/register-callback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ callback_type: callbackType, callback_code: callbackCode }),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to register callback');
    }
    return response.json();
  },

  /**
   * List all registered callback types
   */
  async listCallbacks(): Promise<{ callbacks: string[] }> {
    const response = await fetch(`${API_BASE}/callbacks`);
    return response.json();
  },

  /**
   * Delete a batch
   */
  async deleteBatch(batchId: string): Promise<{ status: string; batch_id: string }> {
    const response = await fetch(`${API_BASE}/batch/${batchId}`, { method: 'DELETE' });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Batch not found');
    }
    return response.json();
  },

  /**
   * List all batches
   */
  async listBatches(): Promise<{ batches: BatchListItem[]; total: number }> {
    const response = await fetch(`${API_BASE}/batches`);
    return response.json();
  },

  /**
   * Poll batch status until completed or failed
   */
  async waitForBatch(
    batchId: string,
    intervalMs: number = 1000,
    maxAttempts: number = 60
  ): Promise<BatchResultResponse> {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const status = await this.getBatchStatus(batchId);

      if (status.status === 'completed' || status.status === 'failed') {
        return this.getBatchResults(batchId);
      }

      if (status.status === 'cancelled') {
        throw new Error('Batch was cancelled');
      }

      await new Promise((resolve) => setTimeout(resolve, intervalMs));
    }

    throw new Error('Batch polling timeout');
  },
};

export default batchService;

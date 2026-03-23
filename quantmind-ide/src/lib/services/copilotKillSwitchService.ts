/**
 * Copilot Kill Switch Service
 *
 * Frontend service for controlling the Copilot/agent kill switch.
 * Independent from trading kill switch.
 */

import { API_CONFIG } from '$lib/config/api';

export interface CopilotKillSwitchStatus {
  active: boolean;
  suspended_at_utc: string | null;
  activated_by: string | null;
  terminated_tasks_count: number;
}

export interface CopilotKillSwitchResponse {
  success: boolean;
  suspended_at_utc?: string;
  activated_by?: string;
  terminated_tasks?: string[];
  already_active?: boolean;
}

export interface CopilotKillSwitchResumeResponse {
  success: boolean;
  resumed_at_utc?: string;
  not_active?: boolean;
}

class CopilotKillSwitchService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_CONFIG.API_URL || 'http://localhost:8000';
  }

  /**
   * Activate the copilot kill switch
   */
  async activate(activator: string = 'user'): Promise<CopilotKillSwitchResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/copilot/kill-switch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ activator }),
      });

      if (!response.ok) {
        throw new Error(`Failed to activate kill switch: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error activating copilot kill switch:', error);
      throw error;
    }
  }

  /**
   * Resume copilot - reactivate the agent system
   */
  async resume(): Promise<CopilotKillSwitchResumeResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/copilot/kill-switch/resume`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to resume copilot: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error resuming copilot:', error);
      throw error;
    }
  }

  /**
   * Get current kill switch status
   */
  async getStatus(): Promise<CopilotKillSwitchStatus> {
    try {
      const response = await fetch(`${this.baseUrl}/api/copilot/kill-switch/status`);

      if (!response.ok) {
        throw new Error(`Failed to get kill switch status: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error getting copilot kill switch status:', error);
      throw error;
    }
  }

  /**
   * Get kill switch activation history
   */
  async getHistory(): Promise<any[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/copilot/kill-switch/history`);

      if (!response.ok) {
        throw new Error(`Failed to get kill switch history: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error getting copilot kill switch history:', error);
      throw error;
    }
  }
}

// Singleton instance
export const copilotKillSwitchService = new CopilotKillSwitchService();

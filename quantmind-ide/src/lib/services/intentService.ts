// Intent service for handling NL commands in the frontend
// Story 5.7: NL System Commands & Context-Aware Canvas Binding

import { canvasContextStore, type CanvasContext } from '$lib/stores/canvas';

export interface CommandRequest {
  message: string;
  canvas_context: CanvasContext;
  confirmed?: boolean;
}

export interface CommandResponse {
  type: 'confirmation_needed' | 'clarification_needed' | 'success' | 'error' | 'general_query';
  message: string;
  intent?: string;
  entities?: string[];
  suggestions?: string[];
  classification?: {
    intent: string;
    confidence: number;
  };
  positions?: any[];
  regime?: string;
  account?: any;
  action?: string;
  symbols?: string[];
}

export interface PendingConfirmation {
  message: string;
  intent: string;
  entities: string[];
}

class IntentService {
  private baseUrl = '/api/floor-manager';

  async sendCommand(message: string, confirmed = false): Promise<CommandResponse> {
    const canvasContext = await this.getCanvasContext();

    const request: CommandRequest = {
      message,
      canvas_context: canvasContext,
      confirmed
    };

    const response = await fetch(`${this.baseUrl}/command`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      return {
        type: 'error',
        message: error.detail || 'Failed to process command'
      };
    }

    return response.json();
  }

  private getCanvasContext(): CanvasContext {
    let context: CanvasContext = { canvas: 'workshop', session_id: '' };

    canvasContextStore.subscribe(value => {
      context = value;
    })();

    return context;
  }

  isConfirmationNeeded(response: CommandResponse): boolean {
    return response.type === 'confirmation_needed';
  }

  isClarificationNeeded(response: CommandResponse): boolean {
    return response.type === 'clarification_needed';
  }

  isError(response: CommandResponse): boolean {
    return response.type === 'error';
  }

  isSuccess(response: CommandResponse): boolean {
    return response.type === 'success';
  }

  parseSuggestions(response: CommandResponse): string[] {
    return response.suggestions || [];
  }

  parseConfirmation(response: CommandResponse): PendingConfirmation | null {
    if (response.type !== 'confirmation_needed') return null;

    return {
      message: response.message,
      intent: response.intent || '',
      entities: response.entities || []
    };
  }
}

export const intentService = new IntentService();

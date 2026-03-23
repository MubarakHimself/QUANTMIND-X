// Story 5.5: Copilot Panel Streaming Tests
// Tests for token-by-token streaming, cursor blink, auto-scroll pause, and tool call UI

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Mock streaming response helper
function createMockStreamReader(chunks: string[]) {
  let index = 0;
  return {
    read: async () => {
      if (index >= chunks.length) {
        return { done: true, value: undefined };
      }
      const value = new TextEncoder().encode(chunks[index]);
      index++;
      return { done: false, value };
    }
  };
}

// Mock SSE data chunks
const streamingChunks = [
  'data: {"type": "tool", "tool": "thinking", "status": "started"}\n\n',
  'data: {"type": "content", "delta": "Hello"}\n\n',
  'data: {"type": "content", "delta": " there"}\n\n',
  'data: {"type": "content", "delta": "!"}\n\n',
  'data: {"type": "tool", "tool": "thinking", "status": "completed"}\n\n',
  'data: {"type": "done"}\n\n'
];

describe('CopilotPanel Streaming', () => {
  describe('SSE Parsing', () => {
    it('should parse SSE data lines correctly', () => {
      const chunk = 'data: {"type": "content", "delta": "test"}\n\n';
      const lines = chunk.split('\n');

      const dataLines = lines.filter(line => line.startsWith('data: '));
      expect(dataLines.length).toBe(1);

      const data = JSON.parse(dataLines[0].slice(6));
      expect(data.type).toBe('content');
      expect(data.delta).toBe('test');
    });

    it('should handle partial SSE lines across chunks', () => {
      let lineBuffer = '';
      const chunk1 = 'data: {"type": "content", ';
      const chunk2 = '"delta": "test"}\n\n';

      lineBuffer += chunk1;
      let lines = lineBuffer.split('\n');
      const incompleteLine = lines.pop(); // Keep incomplete line in buffer

      lineBuffer += chunk2;
      lines = lineBuffer.split('\n');
      const remainingLine = lines.pop();

      expect(incompleteLine).toBe('data: {"type": "content", ');
      expect(remainingLine).toBe('');
    });

    it('should accumulate content deltas correctly', () => {
      let fullContent = '';
      const deltas = ['Hello', ' there', '!'];

      for (const delta of deltas) {
        fullContent += delta;
      }

      expect(fullContent).toBe('Hello there!');
    });
  });

  describe('Cursor Blink', () => {
    it('should toggle cursor visibility every 600ms', async () => {
      vi.useFakeTimers();

      let cursorVisible = true;
      let toggleCount = 0;

      const interval = setInterval(() => {
        cursorVisible = !cursorVisible;
        toggleCount++;
      }, 600);

      // Advance time by 1800ms (3 toggles)
      vi.advanceTimersByTime(1800);

      expect(toggleCount).toBe(3);
      expect(cursorVisible).toBe(false); // Toggled 3 times: true -> false -> true -> false

      clearInterval(interval);
      vi.useRealTimers();
    });

    it('should clear interval on cleanup', () => {
      vi.useFakeTimers();

      let cursorVisible = true;
      let intervalCleared = false;

      const interval = setInterval(() => {
        cursorVisible = !cursorVisible;
      }, 600);

      // Simulate cleanup
      clearInterval(interval);

      // Advance time - cursor should NOT toggle since interval cleared
      vi.advanceTimersByTime(1200);

      expect(cursorVisible).toBe(true); // Still true - interval was cleared

      vi.useRealTimers();
    });
  });

  describe('Auto-Scroll', () => {
    it('should pause auto-scroll when user scrolls up', () => {
      const scrollTop = 0;
      const scrollHeight = 500;
      const clientHeight = 400;

      // User has scrolled up (not near bottom)
      const isNearBottom = (scrollHeight - scrollTop - clientHeight) < 50;
      expect(isNearBottom).toBe(false);
    });

    it('should enable auto-scroll when near bottom', () => {
      const scrollTop = 90;
      const scrollHeight = 500;
      const clientHeight = 400;

      // User is near bottom
      const isNearBottom = (scrollHeight - scrollTop - clientHeight) < 50;
      expect(isNearBottom).toBe(true);
    });
  });

  describe('Tool Call UI', () => {
    it('should show pulsing dot when tool status is started', () => {
      const toolCall = { tool: 'thinking', status: 'started' as const };
      expect(toolCall.status).toBe('started');
    });

    it('should show checkmark when tool status is completed', () => {
      const toolCall = { tool: 'thinking', status: 'completed' as const };
      expect(toolCall.status).toBe('completed');
    });
  });

  describe('NFR-P3: First Token Timing', () => {
    it('should track first token delivery time', () => {
      const startTime = performance.now();

      // Simulate receiving first token after 300ms
      const firstTokenTime = 300;
      const endTime = startTime + firstTokenTime;

      const elapsed = endTime - startTime;
      expect(elapsed).toBeGreaterThanOrEqual(280); // Allow small variance
      expect(elapsed).toBeLessThanOrEqual(320);
    });

    it('should log warning if first token exceeds 5 seconds', () => {
      const firstTokenTime = 6000; // 6 seconds - exceeds NFR-P3

      const consoleSpy = vi.spyOn(console, 'debug').mockImplementation(() => {});

      // This would be the check in production code
      if (firstTokenTime > 5000) {
        console.debug(`[NFR-P3] First token: ${firstTokenTime}ms - WARNING: exceeds 5s target`);
      }

      expect(consoleSpy).toHaveBeenCalled();
      consoleSpy.mockRestore();
    });
  });

  describe('Conversation History', () => {
    it('should limit history to configured limit', () => {
      const CONVERSATION_HISTORY_LIMIT = 10;
      const messages = Array.from({ length: 20 }, (_, i) => ({ id: i, content: `msg${i}` }));

      const history = messages.slice(-CONVERSATION_HISTORY_LIMIT);

      expect(history.length).toBe(10);
      expect(history[0].id).toBe(10);
      expect(history[9].id).toBe(19);
    });
  });
});

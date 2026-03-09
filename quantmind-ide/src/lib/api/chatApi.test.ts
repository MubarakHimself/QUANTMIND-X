import { describe, it, expect, vi } from 'vitest';
import { chatApi } from './chatApi';

describe('chatApi', () => {
  it('should have createSession method', () => {
    expect(typeof chatApi.createSession).toBe('function');
  });

  it('should have getSession method', () => {
    expect(typeof chatApi.getSession).toBe('function');
  });

  it('should have sendMessage method', () => {
    expect(typeof chatApi.sendMessage).toBe('function');
  });
});

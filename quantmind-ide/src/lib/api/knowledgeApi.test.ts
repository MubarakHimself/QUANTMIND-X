import { describe, it, expect, vi } from 'vitest';
import { searchKnowledge, getKnowledgeSources, SOURCE_BADGE_COLORS, SOURCE_FILTERS, type SourceFilter } from './knowledgeApi';

describe('knowledgeApi', () => {
  // Mock fetch globally
  global.fetch = vi.fn();

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('getKnowledgeSources', () => {
    it('should fetch knowledge sources successfully', async () => {
      const mockSources = [
        { id: 'articles', type: 'articles', status: 'online', document_count: 150 },
        { id: 'books', type: 'books', status: 'online', document_count: 75 },
        { id: 'logs', type: 'logs', status: 'offline', document_count: 0 }
      ];

      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: async () => mockSources
      });

      const result = await getKnowledgeSources();

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/knowledge/sources',
        expect.objectContaining({
          headers: expect.objectContaining({ 'Content-Type': 'application/json' })
        })
      );
      expect(result).toEqual(mockSources);
    });

    it('should throw error on non-ok response', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        text: async () => 'Server error'
      });

      await expect(getKnowledgeSources()).rejects.toThrow('API Error: 500 Internal Server Error');
    });
  });

  describe('searchKnowledge', () => {
    it('should search knowledge base with query', async () => {
      const mockResponse = {
        results: [
          {
            source_type: 'articles',
            title: 'Test Article',
            excerpt: 'This is a test excerpt...',
            relevance_score: 0.85,
            provenance: {
              source_url: 'https://example.com/article',
              source_type: 'articles',
              indexed_at_utc: '2026-03-15T10:00:00Z'
            }
          }
        ],
        total: 1,
        query: 'test query',
        warnings: []
      };

      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      });

      const result = await searchKnowledge('test query');

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/knowledge/search',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            query: 'test query',
            sources: null,
            limit: 10
          })
        })
      );
      expect(result).toEqual(mockResponse);
    });

    it('should pass sources filter when provided', async () => {
      const mockResponse = {
        results: [],
        total: 0,
        query: 'test',
        warnings: []
      };

      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      });

      await searchKnowledge('test', ['articles', 'books'], 5);

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/knowledge/search',
        expect.objectContaining({
          body: JSON.stringify({
            query: 'test',
            sources: ['articles', 'books'],
            limit: 5
          })
        })
      );
    });

    it('should include warnings in response', async () => {
      const mockResponse = {
        results: [],
        total: 0,
        query: 'test',
        warnings: ['logs index is offline']
      };

      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      });

      const result = await searchKnowledge('test');
      expect(result.warnings).toContain('logs index is offline');
    });
  });

  describe('SOURCE_BADGE_COLORS', () => {
    it('should have correct colors for each source type', () => {
      expect(SOURCE_BADGE_COLORS.articles).toBe('#00d4ff');
      expect(SOURCE_BADGE_COLORS.books).toBe('#00c896');
      expect(SOURCE_BADGE_COLORS.logs).toBe('#f0a500');
      expect(SOURCE_BADGE_COLORS.personal).toBe('#a78bfa');
    });
  });

  describe('SOURCE_FILTERS', () => {
    it('should have all required filter options', () => {
      const filterValues = SOURCE_FILTERS.map(f => f.value);
      expect(filterValues).toContain('all');
      expect(filterValues).toContain('articles');
      expect(filterValues).toContain('books');
      expect(filterValues).toContain('logs');
      expect(filterValues).toContain('personal');
    });

    it('should have labels for each filter', () => {
      SOURCE_FILTERS.forEach(filter => {
        expect(filter.label).toBeDefined();
        expect(typeof filter.label).toBe('string');
      });
    });
  });
});

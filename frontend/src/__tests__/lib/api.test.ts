/**
 * Tests for API client functions
 */
import { fetchHealth, fetchStats, fetchChat } from '@/lib/api'

// Mock fetch
global.fetch = jest.fn()

describe('API Client', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  describe('fetchHealth', () => {
    it('returns health status', async () => {
      const mockResponse = {
        status: 'healthy',
        vector_store: 'connected',
        collection_count: 100,
        rag_service: 'ready',
      }

      ;(fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      })

      const result = await fetchHealth()
      expect(result.status).toBe('healthy')
      expect(result.collection_count).toBe(100)
    })
  })

  describe('fetchStats', () => {
    it('returns statistics', async () => {
      const mockResponse = {
        total_documents: 100,
        collection_name: 'dbs_documents',
        status: 'active',
      }

      ;(fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      })

      const result = await fetchStats()
      expect(result.total_documents).toBe(100)
    })
  })

  describe('fetchChat', () => {
    it('sends chat request and returns response', async () => {
      const mockResponse = {
        response: 'Test response',
        sources: ['https://example.com'],
        context: [],
        model: 'gpt-4-turbo-preview',
        tokens_used: 100,
        timestamp: new Date().toISOString(),
      }

      ;(fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      })

      const result = await fetchChat({ query: 'test query' })
      expect(result.response).toBe('Test response')
      expect(result.sources).toHaveLength(1)
    })

    it('handles errors gracefully', async () => {
      ;(fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        text: async () => 'Error message',
      })

      await expect(fetchChat({ query: 'test' })).rejects.toThrow('Error message')
    })
  })
})


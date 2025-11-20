/**
 * Tests for ChatPanel component
 */
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ChatPanel } from '@/components/chat/ChatPanel'

// Mock the API
jest.mock('@/lib/api', () => ({
  fetchChat: jest.fn(),
  fetchChatStream: jest.fn(),
}))

describe('ChatPanel', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    localStorage.clear()
  })

  it('renders chat interface', () => {
    render(<ChatPanel />)
    expect(screen.getByText('Ask DBS')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('Ask me anything about DBS…')).toBeInTheDocument()
  })

  it('displays initial message when no history', () => {
    render(<ChatPanel />)
    expect(screen.getByText(/Start the conversation/)).toBeInTheDocument()
  })

  it('sends message when user submits', async () => {
    const { fetchChatStream } = require('@/lib/api')
    fetchChatStream.mockImplementation(async function* () {
      yield 'Test response'
    })

    render(<ChatPanel />)
    const input = screen.getByPlaceholderText('Ask me anything about DBS…')
    const sendButton = screen.getByText('Send')

    await userEvent.type(input, 'test query')
    await userEvent.click(sendButton)

    await waitFor(() => {
      expect(screen.getByText('test query')).toBeInTheDocument()
    })
  })
})


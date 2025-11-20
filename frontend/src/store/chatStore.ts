"use client";

import { create } from "zustand";
import type { ChatMessage, ConversationHistoryItem } from "@/types/chat";

type ChatState = {
  messages: ChatMessage[];
  setMessages: (messages: ChatMessage[]) => void;
  addMessage: (message: ChatMessage) => void;
  reset: () => void;
};

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  setMessages: (messages) => set({ messages }),
  addMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),
  reset: () => set({ messages: [] }),
}));

export const toConversationHistory = (
  messages: ChatMessage[]
): ConversationHistoryItem[] =>
  messages.map(({ role, content }) => ({ role, content }));


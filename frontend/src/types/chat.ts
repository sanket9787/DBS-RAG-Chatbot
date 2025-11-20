export type Role = "user" | "assistant";

export type ChatMessage = {
  role: Role;
  content: string;
  timestamp: string;
  sources?: string[];
};

export type ConversationHistoryItem = {
  role: Role;
  content: string;
};


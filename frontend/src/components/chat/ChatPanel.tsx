"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { fetchChat } from "@/lib/api";
import type { ChatMessage } from "@/types/chat";
import { useChatStore, toConversationHistory } from "@/store/chatStore";

const STORAGE_KEY = "dbs-chat-messages";

export function ChatPanel() {
  const { messages, addMessage, setMessages, reset } = useChatStore();
  const store = useChatStore;
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endOfMessagesRef = useRef<HTMLDivElement | null>(null);
  const handleReset = () => {
    reset();
    localStorage.removeItem(STORAGE_KEY);
    setError(null);
  };


  useEffect(() => {
    try {
      const cached = localStorage.getItem(STORAGE_KEY);
      if (cached) {
        const parsed = JSON.parse(cached);
        if (Array.isArray(parsed)) {
          setMessages(parsed);
        }
      }
    } catch (err) {
      console.warn("Failed to hydrate chat history", err);
    }
  }, [setMessages]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    endOfMessagesRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;
    const userMessage: ChatMessage = {
      role: "user",
      content: input.trim(),
      timestamp: new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
    };
    addMessage(userMessage);
    const queryText = input.trim();
    setInput("");
    setError(null);
    setLoading(true);

    const history = toConversationHistory([...messages, userMessage]);

    // Create placeholder assistant message for streaming
    const assistantMessage: ChatMessage = {
      role: "assistant",
      content: "",
      timestamp: new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
      sources: [],
    };
    addMessage(assistantMessage);

    try {
      // Use streaming
      const { fetchChatStream } = await import("@/lib/api");
      let fullContent = "";
      let sources: string[] = [];

      for await (const chunk of fetchChatStream({
        query: queryText,
        conversation_history: history,
      })) {
        fullContent += chunk;
        // Update messages with streaming content
        const currentMessages = store.getState().messages;
        const updatedMessages = currentMessages.map((msg, idx) => {
          if (idx === currentMessages.length - 1 && msg.role === "assistant") {
            return { ...msg, content: fullContent };
          }
          return msg;
        });
        setMessages(updatedMessages);
        // Auto-scroll during streaming
        setTimeout(() => {
          endOfMessagesRef.current?.scrollIntoView({ behavior: "smooth" });
        }, 0);
      }

      // After streaming completes, fetch final response for sources
      try {
        const finalResponse = await fetchChat({
          query: queryText,
          conversation_history: history,
        });
        sources = finalResponse.sources || [];
      } catch (e) {
        // If fetching final response fails, continue with empty sources
        console.warn("Could not fetch final response for sources", e);
      }

      // Final update with sources
      const finalMessages = store.getState().messages;
      const updatedFinalMessages = finalMessages.map((msg, idx) => {
        if (idx === finalMessages.length - 1 && msg.role === "assistant") {
          return { ...msg, content: fullContent, sources: sources };
        }
        return msg;
      });
      setMessages(updatedFinalMessages);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong. Please try again."
      );
      // Remove the empty assistant message on error
      const errorMessages = [...messages, userMessage];
      setMessages(errorMessages);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-[520px] flex-col rounded-3xl border border-white/10 bg-gradient-to-br from-slate-900/90 via-slate-900 to-[#020617] p-6 shadow-2xl shadow-sky-500/10">
      <div className="flex flex-col gap-2 border-b border-white/10 pb-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
            Live Conversation
          </p>
          <h2 className="text-2xl font-semibold text-white">Ask DBS</h2>
        </div>
        <div className="flex items-center gap-3 text-xs text-slate-300">
          <span className="inline-flex items-center gap-2 rounded-full border border-white/15 px-4 py-1">
            <span
              className={`h-2 w-2 rounded-full ${
                loading ? "bg-amber-400" : "bg-emerald-400"
              }`}
            />
            {loading ? "Generating…" : "Ready"}
          </span>
          {messages.length > 0 && (
            <button
              type="button"
              onClick={handleReset}
              className="rounded-full border border-white/15 px-4 py-1 text-slate-300 transition hover:border-white/30 hover:text-white"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      <div className="mt-6 flex-1 space-y-4 overflow-y-auto pr-2">
        {messages.length === 0 ? (
          <div className="space-y-3 rounded-3xl border border-dashed border-white/15 bg-white/5 p-5 text-sm text-slate-300">
            <p>
              Start the conversation by asking about a course, admission requirement, or student support service.
            </p>
          </div>
        ) : null}
        {messages.map((message, index) => (
          <article
            key={`${message.role}-${index}`}
            className={`flex flex-col gap-2 rounded-3xl border border-white/5 px-5 py-4 ${
              message.role === "assistant"
                ? "bg-white/5 text-slate-100"
                : "bg-sky-500/10 text-slate-200"
            }`}
          >
            <div className="flex items-center justify-between text-xs uppercase tracking-wide text-slate-400">
              <span>{message.role === "assistant" ? "Assistant" : "You"}</span>
              <span>{message.timestamp}</span>
            </div>
            <div className="prose prose-invert max-w-none text-base leading-relaxed prose-headings:mt-2 prose-headings:text-white prose-strong:text-white prose-p:my-2 prose-ul:list-disc prose-ul:pl-6">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            </div>
            {message.sources && message.sources.length > 0 && (
              <div className="flex flex-wrap gap-3 text-xs text-sky-300">
                {Array.from(new Set(message.sources.filter(s => s && s.trim() !== ''))).map((source, idx) => (
                  <a
                    key={`${source}-${idx}`}
                    href={source}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1 rounded-full border border-sky-500/30 px-3 py-1 hover:text-sky-200"
                  >
                    Source {idx + 1} ↗
                  </a>
                ))}
              </div>
            )}
          </article>
        ))}
        {error && (
          <div className="rounded-2xl border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
            {error}
          </div>
        )}
        <div ref={endOfMessagesRef} />
      </div>

      <div className="mt-6 rounded-3xl border border-white/10 bg-white/5 p-4">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask me anything about DBS…"
          className="min-h-[80px] w-full resize-none rounded-2xl bg-slate-950/50 p-4 text-sm text-white outline-none focus:ring-2 focus:ring-sky-500"
          disabled={loading}
        />
        <div className="mt-3 flex items-center justify-end">
          <button
            onClick={handleSend}
            disabled={loading}
            className="rounded-full bg-sky-500 px-5 py-2 text-sm font-semibold text-white transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading ? "Sending…" : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}

